# Copyright 2018 Criteo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import logging
import uuid
import os
import time
import typing
import warnings
from contextlib import suppress, contextmanager
from enum import Enum
from sys import version_info as v, modules
from tempfile import NamedTemporaryFile, gettempdir
from threading import Thread

import dill
import skein
import tensorflow as tf
from skein.exceptions import SkeinError
from skein.model import FinalStatus, ApplicationReport

from ._internal import (
    dump_fn,
    iter_tasks,
    zip_path,
    StaticDefaultDict,
    create_and_pack_conda_env
)
from ._env import (
   gen_pyenv_from_existing_archive,
   gen_task_cmd
)


__all__ = [
    "Experiment",
    "run_on_yarn", "RunFailed", "NodeLabel", "TaskSpec",
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


class NodeLabel(Enum):
    """YARN node label expression.

    A task with a CPU label could be scheduled on any node, whereas
    a task with a GPU label, only on the one labeled with ``"gpu"``.
    """
    CPU = ""  # Default.
    GPU = "gpu"


class TaskSpec(typing.NamedTuple):
    memory: int
    vcores: int
    instances: int = 1
    label: NodeLabel = NodeLabel.CPU


TASK_SPEC_NONE = TaskSpec(0, 0, 0)


class RunFailed(Exception):
    """``run_on_yarn`` failed."""


def run_on_yarn(
    experiment_fn: ExperimentFn,
    task_specs: typing.Dict[str, TaskSpec],
    *,
    pyenv_zip_path: typing.Union[str, typing.Dict[NodeLabel, str]] = None,
    python: str = f"{v.major}.{v.minor}.{v.micro}",
    pip_packages: typing.List[str] = None,
    files: typing.Dict[str, str] = None,
    env: typing.Dict[str, str] = None,
    queue: str = "default",
    file_systems: typing.List[str] = None,
    log_conf_file: str = None
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

    pyenv_zip_path
        Path to an archive of a python environment to be deployed
        It can be a zip conda env or a pex archive
        In case of GPU/CPU cluster, provide a dictionnary with both
        environments.

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

    file_systems
        A list of namenode URIs to acquire delegation tokens for
        in addition to ``fs.defaultFS``.

    log_conf_file
        optional file with log config, setups logging by default with INFO verbosity,
        if you specify a file here don't forget to also ship it to the containers via files arg

    Raises
    ------
    RunFailed
        If the final status of the YARN application is ``"FAILED"``.
    """
    if not pyenv_zip_path:
        warnings.warn(
            "Auto generation of conda environment is deprecated and will be removed in "
            "version 0.2.0, consider creating to pack yourself your environment and "
            "use the pyenv_zip_path argument")
        conda_envs = _make_conda_envs(python, pip_packages or [])
        pyenvs = {node_label: gen_pyenv_from_existing_archive(zipped_path)
                  for node_label, zipped_path in conda_envs.items()}
    elif isinstance(pyenv_zip_path, str):
        pyenvs = {NodeLabel.CPU: gen_pyenv_from_existing_archive(pyenv_zip_path)}
    else:
        pyenvs = {label: gen_pyenv_from_existing_archive(env_zip_path)
                  for label, env_zip_path in pyenv_zip_path.items()}

    # TODO: compute num_ps from the model size and the number of
    # executors. See https://stackoverflow.com/a/46080567/262432.
    task_specs = StaticDefaultDict(task_specs, default=TASK_SPEC_NONE)
    _check_task_specs(task_specs)

    task_files = _maybe_zip_task_files(files or {})
    task_files[__package__] = zip_path(here, replace=True)
    with NamedTemporaryFile(suffix=".dill", delete=False) as file:
        dump_fn(experiment_fn, file.name)
        task_files["experiment_fn.dill"] = file.name

    task_env = {
        **(env or {}),
        # Make Python modules/packages passed via ``files`` importable.
        "PYTHONPATH": ".:" + (env or {}).get("PYTHONPATH", "",),
        "PEX_ROOT": os.path.join(gettempdir(), str(uuid.uuid4()))
    }

    num_ps = task_specs["ps"].instances
    num_workers = task_specs["worker"].instances
    services = {}
    for task_type, task_spec in list(task_specs.items()):
        pyenv = pyenvs[task_spec.label]
        services[task_type] = skein.Service(
            [gen_task_cmd(pyenv, num_ps, num_workers, log_conf_file)],
            skein.Resources(task_spec.memory, task_spec.vcores),
            max_restarts=0,
            instances=task_spec.instances,
            node_label=task_spec.label.value,
            files={
                **task_files,
                pyenv.dest_path: pyenv.path_to_archive
            },
            env=task_env)

    tasks = list(iter_tasks(num_workers, num_ps))
    if "evaluator" in task_specs:
        tasks.append("evaluator:0")  # Not part of the cluster.
    spec = skein.ApplicationSpec(
        services,
        queue=queue,
        file_systems=file_systems)
    with skein.Client() as client:
        _submit_and_await_termination(client, spec, tasks)


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
            source = zip_path(source, replace=True)

        task_files[target] = source
    return task_files


def _make_conda_envs(python, pip_packages) -> typing.Dict[NodeLabel, str]:
    fp = hashlib.md5(str(pip_packages).encode()).hexdigest()
    base_packages = [
        "dill==" + dill.__version__,
        "skein==" + skein.__version__
    ]
    # TODO: use internal PyPI for CPU-optimized TF.
    # TODO: make the user responsible for constructing this mapping.
    return {
        NodeLabel.CPU: create_and_pack_conda_env(
            f"py{python}-{fp}-cpu",
            python,
            pip_packages + base_packages + ["tensorflow==" + tf.__version__]
        ),
        NodeLabel.GPU: create_and_pack_conda_env(
            f"py{python}-{fp}-gpu",
            python,
            pip_packages + base_packages + [
                "tensorflow-gpu==" + tf.__version__
            ]
        )
    }


@contextmanager
def _shutdown_on_exception(app: skein.ApplicationClient):
    # Ensure SIGINT is not masked to enable kill on C-c.
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        yield
    except (KeyboardInterrupt, SystemExit):
        with suppress(SkeinError):
            app.shutdown(FinalStatus.KILLED)
        logger.error("Application killed on user request")
    except Exception:
        with suppress(SkeinError):
            app.shutdown(FinalStatus.FAILED)
        logger.exception("Application shutdown due to an exception")
        raise


def _submit_and_await_termination(
    client: skein.Client,
    spec: skein.ApplicationSpec,
    tasks: typing.List[str],
    poll_every_secs: int = 10
):
    app = client.submit_and_connect(spec)
    events: typing.Dict[str, typing.Dict[str, str]] = {task: {} for task in tasks}
    event_listener = Thread(target=_aggregate_events, args=(app.kv, events))
    event_listener.start()
    with _shutdown_on_exception(app):
        state = None
        while True:
            report = client.application_report(app.id)
            logger.info(
                f"Application report for {app.id} (state: {report.state})")
            if state != report.state:
                logger.info(_format_app_report(report))

            if report.final_status != "undefined":
                event_listener.join()
                logger.info(_format_run_summary(events))
                if report.final_status == "failed":
                    raise RunFailed
                else:
                    break

            time.sleep(poll_every_secs)
            state = report.state


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


def _aggregate_events(
    kv: skein.kv.KeyValueStore,
    events: typing.Dict[str, typing.Dict[str, str]]
) -> None:
    """
    Aggregate events from all dispatched tasks.

    The lifecycle of a task consists of three stages:
    * init which carries the reserved socket address,
    * start with no payload, and
    * stop with an optional formatted exception.
    """
    # ``ConnectionError`` indicates that the app has finished and
    # the AM is down.
    queue = kv.events(event_type="PUT")
    with suppress(skein.exceptions.ConnectionError), queue:
        for event in queue:
            task, stage = event.key.rsplit("/", 1)
            events[task][stage] = event.result.value.decode()


def _format_run_summary(
    events: typing.Dict[str, typing.Dict[str, str]]
) -> str:
    header = []
    details = []
    for task, stages in sorted(events.items()):
        if "stop" in stages:
            status = "FAILED" if stages["stop"] else "SUCCEEDED"
        elif stages:
            status = "KILLED"
        else:
            # No events -- container was never started.
            status = "REQUESTED"

        sock_addr = stages.get("init", "")
        exception = stages.get("stop", "")
        logs = stages.get("logs", "")
        header.append(f"{task:>16}  {sock_addr}  {status}  {logs}")
        if exception:
            details.append(f"Exception in task {task}:")
            details.append(exception)
    return (os.linesep + os.linesep.join(header)
            + os.linesep * (1 + bool(details))
            + os.linesep.join(details))
