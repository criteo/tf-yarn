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
from typing import (
    Dict,
    Optional,
    Tuple,
    NamedTuple,
    Callable,
    Union,
    List
)
import itertools
import warnings
from contextlib import suppress, contextmanager
from enum import Enum
from sys import version_info as v, modules
import tempfile
from threading import Thread
from datetime import timedelta

import dill
import json
import skein
import tensorflow as tf
from skein.exceptions import SkeinError
from skein.model import FinalStatus, ApplicationReport

from ._internal import (
    zip_path,
    StaticDefaultDict,
    create_and_pack_conda_env,
    iter_tasks
)
from ._env import (
   gen_pyenv_from_existing_archive,
   gen_task_cmd,
   PythonEnvDescription
)

__all__ = [
    "Experiment",
    "run_on_yarn", "RunFailed", "NodeLabel", "TaskSpec",
]

logger = logging.getLogger(__name__)

here = os.path.dirname(__file__)


class Experiment(NamedTuple):
    estimator: tf.estimator.Estimator
    train_spec: tf.estimator.TrainSpec
    eval_spec: tf.estimator.EvalSpec

    @property
    def config(self) -> tf.estimator.RunConfig:
        return self.estimator.config


ExperimentFn = Callable[[], Experiment]


class NodeLabel(Enum):
    """YARN node label expression.

    A task with a CPU label could be scheduled on any node, whereas
    a task with a GPU label, only on the one labeled with ``"gpu"``.
    """
    CPU = ""  # Default.
    GPU = "gpu"


class TaskSpec(NamedTuple):
    memory: int
    vcores: int
    instances: int = 1
    label: NodeLabel = NodeLabel.CPU


class SkeinCluster(NamedTuple):
    client: skein.Client
    app: skein.ApplicationClient
    tasks: List[Tuple[str, int]]


class RunFailed(Exception):
    """``run_on_yarn`` failed."""


class Metrics(NamedTuple):
    total_training_time: Optional[timedelta]
    total_eval_duration: Optional[timedelta]
    container_duration: Dict[str, Optional[timedelta]]
    train_eval_time_per_node: Dict[str, Optional[timedelta]]


GB = 2**10
MAX_MEMORY_CONTAINER = 48 * GB
MAX_VCORES_CONTAINER = 48
ALL_TASK_TYPES = {"chief", "worker", "ps", "evaluator"}


def _check_general_topology(task_specs: Dict[str, TaskSpec]) -> None:
    if not task_specs.keys() <= ALL_TASK_TYPES:
        raise ValueError(
            f"task_specs.keys() must be a subset of: {ALL_TASK_TYPES}")
    if task_specs["chief"].instances != 1:
        raise ValueError("exactly one 'chief' task is required")
    for task_type, spec in task_specs.items():
        if spec.memory > MAX_MEMORY_CONTAINER:
            raise ValueError(
                f"{task_type}: Can not demand more memory than "
                f"{MAX_MEMORY_CONTAINER} bytes for container")
        if spec.vcores > MAX_VCORES_CONTAINER:
            raise ValueError(
                f"{task_type}: Can not demand more vcores than "
                f"{MAX_VCORES_CONTAINER} for container")


def _check_ps_topology(task_specs: Dict[str, TaskSpec]) -> None:
    _check_general_topology(task_specs)
    if task_specs["evaluator"].instances > 1:
        raise ValueError("no more than one 'evaluator' task is allowed")
    if not task_specs["ps"].instances:
        raise ValueError(
            "task_specs must contain at least a single 'ps' task for "
            "multi-worker training")


def single_server_topology(
    memory: int = MAX_MEMORY_CONTAINER,
    vcores: int = MAX_VCORES_CONTAINER
) -> Dict[str, TaskSpec]:
    topology = {"chief": TaskSpec(memory=memory, vcores=vcores)}
    _check_general_topology(topology)
    return topology


def ps_strategy_topology(
    nb_workers: int = 2,
    nb_ps: int = 1,
    memory: int = MAX_MEMORY_CONTAINER,
    vcores: int = MAX_VCORES_CONTAINER
) -> Dict[str, TaskSpec]:
    # TODO: compute num_ps from the model size and the number of
    # executors. See https://stackoverflow.com/a/46080567/262432.
    topology = {
        "chief": TaskSpec(memory=memory, vcores=vcores),
        "evaluator": TaskSpec(memory=memory, vcores=vcores),
        "worker": TaskSpec(memory=memory, vcores=vcores, instances=nb_workers),
        "ps": TaskSpec(memory=memory, vcores=vcores, instances=nb_ps)
    }
    _check_ps_topology(topology)
    return topology


TASK_SPEC_NONE = ps_strategy_topology()


def _setup_pyenvs(
        pyenv_zip_path: Union[str, Dict[NodeLabel, str]] = None,
        python: str = f"{v.major}.{v.minor}.{v.micro}",
        pip_packages: List[str] = None
) -> Dict[NodeLabel, PythonEnvDescription]:
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
    return pyenvs


def _setup_task_env(
        tempdir: str,
        files: Dict[str, str] = None,
        env: Dict[str, str] = {}
):
    task_files = _maybe_zip_task_files(files or {}, tempdir)
    task_files[__package__] = zip_path(here, tempdir)

    libhdfs_opts = "-Xms64m -Xmx512m"
    if "LIBHDFS_OPTS" in env:
        libhdfs_opts = "{default} {env}".format(default=libhdfs_opts,
                                                env=env.get("LIBHDFS_OPTS"))

    task_env = {
        **env,
        "LIBHDFS_OPTS": libhdfs_opts,
        # Make Python modules/packages passed via ``files`` importable.
        "PYTHONPATH": ".:" + env.get("PYTHONPATH", ""),
        "PEX_ROOT": os.path.join("/tmp", str(uuid.uuid4()))
    }

    return task_files, task_env


def setup_skein_cluster(
    task_specs: Dict[str, TaskSpec] = TASK_SPEC_NONE,
    *,
    pyenv_zip_path: Union[str, Dict[NodeLabel, str]] = None,
    python: str = f"{v.major}.{v.minor}.{v.micro}",
    pip_packages: List[str] = None,
    files: Dict[str, str] = None,
    env: Dict[str, str] = {},
    queue: str = "default",
    file_systems: List[str] = None,
    log_conf_file: str = None
) -> SkeinCluster:
    """Request a cluster on YARN with Skein.

    The implementation allocates a service with the requested number
    of instances for each distributed TensorFlow task type. Each
    instance expects a serialized run_config to setup the tensorflow servers
    and an experiment function to execute.

    Parameters
    ----------
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
    """
    pyenvs = _setup_pyenvs(pyenv_zip_path, python, pip_packages)

    with tempfile.TemporaryDirectory() as tempdir:
        task_files, task_env = _setup_task_env(tempdir, files, env)
        services = {}
        for task_type, task_spec in list(task_specs.items()):
            pyenv = pyenvs[task_spec.label]
            services[task_type] = skein.Service(
                commands=[gen_task_cmd(pyenv, log_conf_file)],
                resources=skein.model.Resources(task_spec.memory, task_spec.vcores),
                max_restarts=0,
                instances=task_spec.instances,
                node_label=task_spec.label.value,
                files={
                    **task_files,
                    pyenv.dest_path: pyenv.path_to_archive
                },
                env=task_env)

        spec = skein.ApplicationSpec(
            services,
            queue=queue,
            file_systems=file_systems)
        client = skein.Client()
        app = client.submit_and_connect(spec)
        task_instances = [(task_type, spec.instances) for task_type, spec in task_specs.items()]
        # Note that evaluator is not part of the cluster
        cluster_instances = [t for t in task_instances if t[0] is not 'evaluator']
        app.kv['cluster_instances'] = json.dumps(cluster_instances).encode()

        return SkeinCluster(client, app, task_instances)


def run_on_cluster(
    experiment_fn: ExperimentFn,
    cluster: SkeinCluster
) -> None:
    """Run an experiment on YARN.

    Dispatches experiment_fn to the cluster and awaits termination
    of the train_and_evaluate execution.

    Parameters
    ----------
    experiment_fn
        A function constructing the estimator alongside the train
        and eval specs.

    cluster
        optional skein cluster. All parameters except experiment_fn will be ignored.

    Raises
    ------
    RunFailed
        If the final status of the YARN application is ``"FAILED"``.
    """
    # Attempt serialization early to avoid allocating unnecesary resources
    serialized_fn = dill.dumps(experiment_fn, recurse=True)
    with cluster.client:
        return _execute_and_await_termination(cluster, serialized_fn)


def run_on_yarn(
    experiment_fn: ExperimentFn,
    task_specs: Dict[str, TaskSpec] = None,
    *,
    pyenv_zip_path: Union[str, Dict[NodeLabel, str]] = None,
    python: str = f"{v.major}.{v.minor}.{v.micro}",
    pip_packages: List[str] = None,
    files: Dict[str, str] = None,
    env: Dict[str, str] = {},
    queue: str = "default",
    file_systems: List[str] = None,
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
    warnings.warn("This method is deprecated. Please use run_on_cluster and setup_skein_cluster")
    cluster = setup_skein_cluster(
        StaticDefaultDict(task_specs, default=TASK_SPEC_NONE),
        pyenv_zip_path=pyenv_zip_path,
        python=python,
        pip_packages=pip_packages,
        files=files,
        env=env,
        queue=queue,
        file_systems=file_systems,
        log_conf_file=log_conf_file
    )
    run_on_cluster(experiment_fn, cluster)


def _maybe_zip_task_files(files, tempdir):
    task_files = {}
    for target, source in files.items():
        assert target not in task_files
        if os.path.isdir(source):
            source = zip_path(source, tempdir)

        task_files[target] = source
    return task_files


def _make_conda_envs(python, pip_packages) -> Dict[NodeLabel, str]:
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


def _execute_and_await_termination(
    cluster: SkeinCluster,
    serialized_fn: bytes,
    poll_every_secs: int = 10
):
    events: Dict[str, Dict[str, str]] = {task: {} for task in iter_tasks(cluster.tasks)}
    event_listener = Thread(target=_aggregate_events, args=(cluster.app.kv, events))
    event_listener.start()
    cluster.app.kv['experiment_fn'] = serialized_fn
    with _shutdown_on_exception(cluster.app):
        state = None
        while True:
            report = cluster.client.application_report(cluster.app.id)
            logger.info(f"Application report for {cluster.app.id} (state: {report.state})")
            if state != report.state:
                logger.info(_format_app_report(report))

            if report.final_status != "undefined":
                event_listener.join()
                log_events, metrics = _handle_events(events)
                logger.info(log_events)
                if report.final_status == "failed":
                    raise RunFailed
                else:
                    break
            time.sleep(poll_every_secs)
            state = report.state

    return metrics


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
    events: Dict[str, Dict[str, str]]
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
            if "/" in event.key:
                task, stage = event.key.rsplit("/", 1)
                events[task][stage] = event.result.value.decode()


def _handle_events(
    events: Dict[str, Dict[str, str]]
) -> Tuple[str, Metrics]:
    def is_worker(task: str) -> bool:
        return task == 'worker'

    def is_evaluator(task: str) -> bool:
        return task == 'evaluator'

    def is_chief(task: str) -> bool:
        return task == 'chief'

    header = []
    details = []
    min_training_start_time = timedelta.max
    max_training_stop_time = timedelta.min
    min_eval_start_time = timedelta.max
    max_eval_stop_time = timedelta.min
    valid_training_time = True
    valid_eval_time = True
    container_duration: Dict[str, Optional[timedelta]] = dict()
    train_eval_time_per_node: Dict[str, Optional[timedelta]] = dict()
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

        container_duration[task] = None
        if 'container_start_time' in stages and 'container_stop_time' in stages:
            container_duration[task] = timedelta(seconds=(float(stages['container_stop_time'])
                                                          - float(stages['container_start_time'])))

        train_eval_time_per_node[task] = None
        task_type = task.split(':')[0]
        if 'train_eval_start_time' in stages and 'train_eval_stop_time' in stages and not exception:
            start_time = timedelta(seconds=float(stages['train_eval_start_time']))
            stop_time = timedelta(seconds=float(stages['train_eval_stop_time']))
            train_eval_time_per_node[task] = stop_time - start_time
            if is_worker(task_type) or is_chief(task_type):
                if start_time < min_training_start_time:
                    min_training_start_time = start_time
                if stop_time > max_training_stop_time:
                    max_training_stop_time = stop_time
            elif is_evaluator(task_type):
                if start_time < min_eval_start_time:
                    min_eval_start_time = start_time
                if stop_time > max_eval_stop_time:
                    max_eval_stop_time = stop_time
        else:
            if is_worker(task_type) or is_chief(task_type):
                valid_training_time = False
            elif is_evaluator(task_type):
                valid_eval_time = False

        header.append(f"{task:>16}  {sock_addr}  {status}  {logs}"
                      f"  Container duration: {container_duration[task]}"
                      f"  Training/evaluation duration : {train_eval_time_per_node[task]}")

        if exception:
            details.append(f"Exception in task {task}:")
            details.append(exception)

    training_time = max_training_stop_time - min_training_start_time\
        if valid_training_time and min_training_start_time < timedelta.max else None
    eval_time = max_eval_stop_time - min_eval_start_time\
        if valid_eval_time and min_eval_start_time < timedelta.max else None
    header.append(f'Training time = {training_time}')
    header.append(f'Evaluation time = {eval_time}')

    metrics = Metrics(
        training_time,
        eval_time,
        container_duration,
        train_eval_time_per_node
    )
    return ((os.linesep + os.linesep.join(header)
            + os.linesep * (1 + bool(details))
             + os.linesep.join(details)), metrics)
