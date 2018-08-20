import hashlib
import logging
import os
import time
import typing
from enum import Enum
from sys import version_info as v

import dill
import skein
import tensorflow as tf

from ._criteo import get_default_env
from ._internal import (
    encode_fn,
    zip_inplace,
    PyEnv,
    StaticDefaultDict
)

__all__ = [
    "Experiment",
    "run_on_yarn", "TaskFlavor", "TaskSpec",
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


def run_on_yarn(
    experiment_fn: ExperimentFn,
    task_specs: typing.Dict[str, TaskSpec],
    *,
    python: str = f"{v.major}.{v.minor}.{v.micro}",
    pip_packages: typing.List[str] = None,
    files: typing.Dict[str, str] = None,
    env: typing.Dict[str, str] = None,
    queue: str = "default",
    max_restarts: int = 0,
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

    max_restarts
        Number of times to retry a failed task. The cap is applied
        independently to each task type, that is ``max_restarts`` equal
        to 1 allows at most one failed task of each type. Set to -1 for
        unbounded retries.

    name_nodes
        A list of namenode URIs to acquire delegation tokens for.
    """
    # TODO: compute num_ps from the model size and the number of
    # executors. See https://stackoverflow.com/a/46080567/262432.
    task_specs = StaticDefaultDict(task_specs, default=TaskSpec.NONE)
    _check_task_specs(task_specs)

    task_files = {__package__: zip_inplace(here, replace=True)}
    for target, source in (files or {}).items():
        assert target not in task_files
        task_files[target] = (
            zip_inplace(source, replace=True)
            if os.path.isdir(source)
            else source
        )

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
        pyenv = pyenvs[task_spec.flavor]
        task_command = (
            f"{pyenv.name}/bin/python -m tf_skein._dispatch_task "
            f"--num-ps={task_specs['ps'].instances} "
            f"--num-workers={task_specs['worker'].instances} "
            "--experiment-fn=" + encode_fn(experiment_fn)
        )

        services[task_type] = skein.Service(
            [task_command],
            skein.Resources(task_spec.memory, task_spec.vcores),
            max_restarts=max_restarts,
            instances=task_spec.instances,
            node_label=task_spec.flavor.value,
            files={**task_files, pyenv.name: zip_inplace(pyenv.create())},
            env=task_env)

    spec = skein.ApplicationSpec(
        services,
        queue=queue,
        name_nodes=name_nodes)
    with skein.Client() as client:
        logger.info(f"Submitting experiment to {queue} queue")
        app_id = client.submit(spec)
        final_status = _await_termination(client, app_id)
        logger.info(
            f"Application {app_id} finished with status {final_status}")
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


def _await_termination(
    client: skein.Client,
    app_id: str,
    poll_every_secs: int = 10
) -> skein.core.FinalStatus:
    # Ensure SIGINT is not masked to enable kill on C-c.
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        while True:
            report = client.application_report(app_id)
            final_status = report.final_status
            # TODO: log status on each tick?
            if final_status != "undefined":
                return final_status

            time.sleep(poll_every_secs)
    except (KeyboardInterrupt, SystemExit):
        client.kill_application(app_id)
        return skein.core.FinalStatus.KILLED
