import hashlib
import logging
import os
import time
import typing
from contextlib import suppress, contextmanager
from enum import Enum
from sys import version_info as v
from tempfile import NamedTemporaryFile

import dill
import skein
import tensorflow as tf
from skein.exceptions import SkeinError
from skein.model import FinalStatus, ApplicationReport

from ._criteo import get_default_env
from ._internal import (
    dump_fn,
    zip_inplace,
    PyEnv,
    StaticDefaultDict
)

__all__ = [
    "Experiment",
    "run_on_yarn", "RunFailed", "TaskFlavor", "TaskSpec",
]

logger = logging.getLogger(__name__)

here = os.path.dirname(__file__)


class Experiment(typing.NamedTuple):
    estimator: tf.estimator.Estimator
    train_spec: tf.estimator.TrainSpec
    eval_spec: tf.estimator.EvalSpec
    # TODO: experiment name?

    @property
    def config(self) -> tf.estimator.RunConfig:
        return self.estimator.config


ExperimentFn = typing.Callable[[], Experiment]


class TaskFlavor(Enum):
    """The flavor values are YARN node label expressions.

    That is, a task with a CPU flavor can be scheduled on any node,
    whereas a task with a GPU flavor, only on the one labeled with
    ``"gpu"``.
    """
    CPU = ""     # Default.
    GPU = "gpu"


class TaskSpec(typing.NamedTuple):
    memory: int
    vcores: int
    instances: int = 1
    flavor: TaskFlavor = TaskFlavor.CPU


#: A "dummy" ``TaskSpec``.
TaskSpec.NONE = TaskSpec(0, 0, 0)


class RunFailed(Exception):
    """``run_on_yarn`` failed."""


def run_on_yarn(
    experiment_fn: ExperimentFn,
    task_specs: typing.Dict[str, TaskSpec],
    *,
    python: str = f"{v.major}.{v.minor}.{v.micro}",
    pip_packages: typing.List[str] = None,
    files: typing.Dict[str, str] = None,
    env: typing.Dict[str, str] = None,
    queue: str = "default",
    name_nodes: typing.List[str] = None
) -> None:
    """Run an experiment on YARN.

    The implementation allocates a service with the requested number
    of instances for each distributed TensorFlow task type. Each
    instance runs ``_dispatch_task`` which roughly does the following.

    1. Reserve a TCP port and communicate the resulting socket address
       (host/port pair) to other instances using the "init" barrier.
    2. Spawn ``train_and_evaluate`` in a separate thread.
    3. Synchronize the "ps" tasks on the "stop" barrier.
       The barrier compensates for the fact that "ps" tasks never
       terminate, and therefore should be killed once all other
       tasks are finished.

    Parameters
    ----------
    experiment_fn
        A function constructing the estimator alongside the train
        and eval specs.

    task_specs
        Resources to allocate for each task type. The keys
        must be a subset of ``"chief"``, ``"worker"``, ``"ps"``, and
        ``"evaluator"``. The minimal spec must contain at least
        ``"chief"``.

    python
        Python version in the MAJOR.MINOR.MICRO format. Defaults to the
        version of ``sys.executable``.

    pip_packages
        Python packages to install in the environment. The packages
        are installed via pip, therefore all of the following forms
        are supported::

            SomeProject>=1,<2
            git+https://github.com/org/SomeProject
            http://SomeProject.org/archives/SomeProject-1.0.4.tar.gz
            path/to/SomeProject

        See `Installing Packages <https://packaging.python.org/tutorials \
        /installing-packages>`_ for more examples.

    files
        Local files or directories to upload to the container.
        The keys are the target locations of the resources relative
        to the container root, while the values -- their
        corresponding local sources. Note that container root is
        appended to ``PYTHONPATH``. Therefore, any listed Python
        module a package is automatically importable.

    env
        Environment variables to forward to the containers.

    queue
        YARN queue to use.

    name_nodes
        A list of namenode URIs to acquire delegation tokens for.

    Raises
    ------
    RunFailed
        If the final status of the YARN application is ``"FAILED"``.
    """
    # TODO: compute num_ps from the model size and the number of
    # executors. See https://stackoverflow.com/a/46080567/262432.
    task_specs = StaticDefaultDict(task_specs, default=TaskSpec.NONE)
    _check_task_specs(task_specs)

    task_files = _maybe_zip_task_files(files or {})
    task_files[__package__] = zip_inplace(here, replace=True)
    with NamedTemporaryFile(suffix=".dill", delete=False) as file:
        dump_fn(experiment_fn, file.name)
        task_files["experiment_fn.dill"] = file.name

    task_env = {
        # XXX this is Criteo-specific. Remove once Lake updates the
        #     container environment. See LAKE-709.
        **get_default_env(),
        **(env or {}),
        # Make Python modules/packages passed via ``files`` importable.
        "PYTHONPATH": ".:" + (env or {}).get("PYTHONPATH", ""),
    }

    pyenvs = _make_pyenvs(python, pip_packages or [])
    services = {}
    for task_type, task_spec in list(task_specs.items()):
        task_command = (
            f"pyenv/bin/python -m tf_yarn._dispatch_task "
            f"--num-ps={task_specs['ps'].instances} "
            f"--num-workers={task_specs['worker'].instances} "
            "--experiment-fn=experiment_fn.dill"
        )

        services[task_type] = skein.Service(
            [task_command],
            skein.Resources(task_spec.memory, task_spec.vcores),
            max_restarts=0,
            instances=task_spec.instances,
            node_label=task_spec.flavor.value,
            files={
                **task_files,
                "pyenv": zip_inplace(pyenvs[task_spec.flavor].create())
            },
            env=task_env)

    spec = skein.ApplicationSpec(services, queue=queue, name_nodes=name_nodes)
    with skein.Client() as client:
        _submit_and_await_termination(client, spec)
        # TODO: report per-container status via KV.


def _check_task_specs(task_specs):
    all_task_types = {"chief", "worker", "ps", "evaluator"}
    if not task_specs.keys() <= all_task_types:
        raise ValueError(
            f"task_specs.keys() must be a subset of: {all_task_types}")

    if task_specs["chief"].instances != 1:
        raise ValueError("exactly one 'chief' task is required")
    if task_specs["evaluator"].instances > 1:
        raise ValueError("no more than one 'evaluator' task is allowed")
    if task_specs["worker"].instances > 0 and not task_specs["ps"].instances:
        raise ValueError(
            "task_specs must contain at least a single 'ps' task for "
            "multi-worker training")


def _maybe_zip_task_files(files):
    task_files = {}
    for target, source in files.items():
        assert target not in task_files
        if os.path.isdir(source):
            source = zip_inplace(source, replace=True)

        task_files[target] = source
    return task_files


def _make_pyenvs(python, pip_packages) -> typing.Dict[TaskFlavor, PyEnv]:
    fp = hashlib.md5(str(pip_packages).encode()).hexdigest()
    base_packages = [
        "dill==" + dill.__version__,
        "git+http://github.com/criteo-forks/skein"
    ]
    # TODO: use internal PyPI for CPU-optimized TF.
    return {
        TaskFlavor.CPU: PyEnv(
            f"py{python}-{fp}-cpu",
            python,
            pip_packages + base_packages + ["tensorflow==" + tf.__version__]
        ),
        TaskFlavor.GPU: PyEnv(
            f"py{python}-{fp}-gpu",
            python,
            pip_packages + base_packages + [
                "tensorflow-gpu==" + tf.__version__
            ]
        )
    }


def _format_app_report(report: ApplicationReport) -> str:
    attrs = [
        "queue",
        "start_time",
        "finish_time",
        "final_status",
        "tracking_url",
        "user"
    ]
    return os.linesep + os.linesep.join(
        f"{attr:>16}: {getattr(report, attr) or ''}" for attr in attrs)


@contextmanager
def _shutdown_on_exception(client: skein.Client, app_id: str):
    # Ensure SIGINT is not masked to enable kill on C-c.
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        yield
    except (KeyboardInterrupt, SystemExit):
        with suppress(SkeinError):
            client.kill_application(app_id)
        logger.error("Application killed on user request")
    except Exception:
        with suppress(SkeinError):
            client.connect(app_id).shutdown(FinalStatus.FAILED)
        logger.exception("Application shutdown due to an exception")
        raise


def _submit_and_await_termination(
    client: skein.Client,
    spec: skein.ApplicationSpec,
    poll_every_secs: int = 10
):
    app_id = client.submit(spec)
    with _shutdown_on_exception(client, app_id):
        state = None
        while True:
            report = client.application_report(app_id)
            logger.info(
                f"Application report for {app_id} (state: {report.state})")
            if state != report.state:
                logger.info(_format_app_report(report))

            if report.final_status != "undefined":
                if report.final_status == "failed":
                    raise RunFailed
                else:
                    break

            time.sleep(poll_every_secs)
            state = report.state
