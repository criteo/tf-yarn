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
import warnings
from contextlib import suppress, contextmanager
from sys import version_info as v
import tempfile
from threading import Thread
from datetime import timedelta

import dill
import json
import skein
import tensorflow as tf
from skein.exceptions import SkeinError
from skein.model import FinalStatus, ApplicationReport

from tf_yarn.topologies import (
    single_server_topology,
    ps_strategy_topology,
    TaskSpec, NodeLabel)
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
from .cluster import aggregate_spec
from tf_yarn.experiment import Experiment
from tf_yarn.evaluator_metrics import (
    add_monitor_to_experiment,
    EvaluatorMetricsLogger
)

__all__ = [
    "Experiment", "run_on_yarn", "RunFailed",
    "single_server_topology", "ps_strategy_topology",
    "setup_skein_cluster", "run_on_cluster"
]

KV_CLUSTER_INSTANCES = 'cluster_instances'
KV_EXPERIMENT_FN = 'experiment_fn'

logger = logging.getLogger(__name__)

here = os.path.dirname(__file__)

ExperimentFn = Callable[[], Experiment]


class SkeinCluster(NamedTuple):
    client: skein.Client
    app: skein.ApplicationClient
    tasks: List[Tuple[str, int]]
    cluster_spec: tf.train.ClusterSpec


class RunFailed(Exception):
    """``run_on_yarn`` failed."""


class Metrics(NamedTuple):
    total_training_time: Optional[timedelta]
    total_eval_duration: Optional[timedelta]
    container_duration: Dict[str, Optional[timedelta]]
    train_eval_time_per_node: Dict[str, Optional[timedelta]]


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

    _add_to_env(env, "LIBHDFS_OPTS", "-Xms64m -Xmx512m")

    task_env = {
        **env,
        # Make Python modules/packages passed via ``files`` importable.
        "PYTHONPATH": ".:" + env.get("PYTHONPATH", ""),
        "PEX_ROOT": os.path.join("/tmp", str(uuid.uuid4()))
    }

    return task_files, task_env


def _add_to_env(env: Dict[str, str], env_name: str, opts: str):
    if env_name in env:
        env[env_name] = f"{opts} {env.get(env_name)}"
    else:
        env[env_name] = f"{opts}"


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


def _setup_cluster_tasks(
    task_instances: List[Tuple[str, int]],
    app: skein.ApplicationClient
) -> tf.train.ClusterSpec:
    # Note that evaluator is not part of the cluster
    cluster_instances = [t for t in task_instances if t[0] is not 'evaluator']
    app.kv[KV_CLUSTER_INSTANCES] = json.dumps(cluster_instances).encode()
    return tf.train.ClusterSpec(aggregate_spec(app, list(iter_tasks(cluster_instances))))


class TFYarnExecutor():

    def __init__(
        self,
        pyenv_zip_path: Union[str, Dict[NodeLabel, str]] = None,
        python: str = f"{v.major}.{v.minor}.{v.micro}",
        pip_packages: List[str] = None,
        queue: str = "default",
        file_systems: List[str] = None
    ) -> None:
        """
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

        queue
            YARN queue to use.

        file_systems
            A list of namenode URIs to acquire delegation tokens for
            in addition to ``fs.defaultFS``.
        """

        self.pyenvs = _setup_pyenvs(pyenv_zip_path, python, pip_packages)
        self.queue = queue
        self.file_systems = file_systems

    def __enter__(self):
        skein.Client.start_global_daemon()
        return self

    def __exit__(self, *args):
        skein.Client.stop_global_daemon()

    def setup_skein_cluster(
        self,
        task_specs: Dict[str, TaskSpec] = TASK_SPEC_NONE,
        *,
        files: Dict[str, str] = None,
        env: Dict[str, str] = {},
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

        files
            Local files or directories to upload to the container.
            The keys are the target locations of the resources relative
            to the container root, while the values -- their
            corresponding local sources. Note that container root is
            appended to ``PYTHONPATH``. Therefore, any listed Python
            module a package is automatically importable.

        env
            Environment variables to forward to the containers.

        log_conf_file
            optional file with log config, setups logging by default with INFO verbosity,
            if you specify a file here don't forget to also ship it to the containers via files arg
        """
        os.environ["JAVA_TOOL_OPTIONS"] = \
            "-XX:ParallelGCThreads=1 -XX:CICompilerCount=2 "\
            f"{os.environ.get('JAVA_TOOL_OPTIONS', '')}"

        with tempfile.TemporaryDirectory() as tempdir:
            task_files, task_env = _setup_task_env(tempdir, files, env)
            services = {}
            for task_type, task_spec in list(task_specs.items()):
                pyenv = self.pyenvs[task_spec.label]
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
                queue=self.queue,
                file_systems=self.file_systems)
            try:
                client = skein.Client.from_global_daemon()
            except skein.exceptions.DaemonNotRunningError:
                client = skein.Client()
            app = client.submit_and_connect(spec)
            task_instances = [(task_type, spec.instances) for task_type, spec in task_specs.items()]
            cluster_spec = _setup_cluster_tasks(task_instances, app)

            return SkeinCluster(client, app, task_instances, cluster_spec)

    def run_on_cluster(
        self,
        experiment_fn: ExperimentFn,
        cluster: SkeinCluster,
        eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None
    ) -> Optional[Metrics]:
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
        def _new_experiment_fn():
            return add_monitor_to_experiment(experiment_fn())
        new_experiment_fn = _new_experiment_fn

        # Attempt serialization early to avoid allocating unnecesary resources
        serialized_fn = dill.dumps(new_experiment_fn, recurse=True)
        with cluster.client:
            return _execute_and_await_termination(
                cluster,
                serialized_fn,
                eval_monitor_log_thresholds
            )

    def run_on_yarn(
        self,
        experiment_fn: ExperimentFn,
        task_specs: Dict[str, TaskSpec] = None,
        *,
        files: Dict[str, str] = None,
        env: Dict[str, str] = {},
        log_conf_file: str = None,
        eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None
    ) -> Optional[Metrics]:
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

        files
            Local files or directories to upload to the container.
            The keys are the target locations of the resources relative
            to the container root, while the values -- their
            corresponding local sources. Note that container root is
            appended to ``PYTHONPATH``. Therefore, any listed Python
            module a package is automatically importable.

        env
            Environment variables to forward to the containers.

        log_conf_file
            optional file with log config, setups logging by default with INFO verbosity,
            if you specify a file here don't forget to also ship it to the containers via files arg

        eval_monitor_log_thresholds
            optional dictionnary of string to (float 1, float 2).
            Each couple (key, value) corresponds to an evaluation
            monitored metric and an associated range. The evaluation monitored metric
            is logged if it is in [float 1; float 2]. If the lower bound is None it is set to 0.
            If the upper bound is None, it is set to maximum value
            A monitored metric with no range is always logged. List of monitored metrics:
            'awake_time_ratio': 'Awake/idle ratio',
            'eval_step_mean_duration': 'Eval step mean duration (in sec)',
            'last_training_step': 'Training set of last checkpoint',
            'nb_eval_steps': 'Number of evaluation steps done'

        Raises
        ------
        RunFailed
            If the final status of the YARN application is ``"FAILED"``.
        """
        cluster = self.setup_skein_cluster(
            StaticDefaultDict(task_specs, default=TASK_SPEC_NONE),
            files=files,
            env=env,
            log_conf_file=log_conf_file
        )
        return self.run_on_cluster(experiment_fn, cluster, eval_monitor_log_thresholds)


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
    eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None,
    poll_every_secs: int = 10
) -> Optional[Metrics]:
    events: Dict[str, Dict[str, str]] = {task: {} for task in iter_tasks(cluster.tasks)}
    event_listener = Thread(target=_aggregate_events, args=(cluster.app.kv, events))
    event_listener.start()
    cluster.app.kv[KV_EXPERIMENT_FN] = serialized_fn
    eval_metrics_logger = EvaluatorMetricsLogger(
        [task for task in iter_tasks(cluster.tasks) if task.startswith('evaluator')],
        cluster.app,
        eval_monitor_log_thresholds
    )
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
            else:
                eval_metrics_logger.log()
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
