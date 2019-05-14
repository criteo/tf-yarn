import importlib
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
from contextlib import suppress, contextmanager
from functools import partial
import tempfile
from threading import Thread
from datetime import timedelta
import subprocess

import cloudpickle
import json
import skein
import tensorflow as tf
from skein.exceptions import SkeinError
from skein.model import FinalStatus, ApplicationReport, ACLs

from tf_yarn.topologies import (
    ps_strategy_topology,
    TaskSpec, NodeLabel)
from ._internal import (
    zip_path,
    StaticDefaultDict,
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
    EvaluatorMetricsLogger)
from tf_yarn.metrics import OneShotMetricsLogger
from tf_yarn.event import broadcast

__all__ = [
    "Experiment", "RunFailed", "run_on_yarn",
    "standalone_client_mode"
    "single_server_topology", "ps_strategy_topology",
    "get_safe_experiment_fn"
]

KV_CLUSTER_INSTANCES = 'cluster_instances'
KV_EXPERIMENT_FN = 'experiment_fn'
YARN_LOG_TRIES = 15
KV_TF_SESSION_CONFIG = 'tf_session_config'

logger = logging.getLogger(__name__)

here = os.path.dirname(__file__)

ExperimentFn = Callable[[], Experiment]


class SkeinCluster(NamedTuple):
    client: skein.Client
    app: skein.ApplicationClient
    tasks: List[Tuple[str, int]]
    cluster_spec: tf.train.ClusterSpec
    event_listener: Thread
    events: Dict[str, Dict[str, str]]


class RunFailed(Exception):
    """``run_on_yarn`` failed."""


class Metrics(NamedTuple):
    total_training_time: Optional[timedelta]
    total_eval_duration: Optional[timedelta]
    container_duration: Dict[str, Optional[timedelta]]
    train_eval_time_per_node: Dict[str, Optional[timedelta]]


TASK_SPEC_NONE = ps_strategy_topology()


def _setup_pyenvs(
        pyenv_zip_path: Union[str, Dict[NodeLabel, str]],
        standalone_client_mode: bool = False
) -> Dict[NodeLabel, PythonEnvDescription]:
    if isinstance(pyenv_zip_path, str):
        pyenvs = {NodeLabel.CPU: gen_pyenv_from_existing_archive(
            pyenv_zip_path,
            standalone_client_mode)}
    else:
        pyenvs = {label: gen_pyenv_from_existing_archive(env_zip_path, standalone_client_mode)
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


def _setup_cluster_tasks(
    task_instances: List[Tuple[str, int]],
    app: skein.ApplicationClient,
    standalone_client_mode: bool
) -> tf.train.ClusterSpec:
    tasks_not_in_cluster = ['evaluator']
    # In standalone client mode the chief is also not part of the cluster
    if standalone_client_mode:
        tasks_not_in_cluster.append('chief')
    cluster_instances = [t for t in task_instances if t[0] not in tasks_not_in_cluster]
    app.kv[KV_CLUSTER_INSTANCES] = json.dumps(cluster_instances).encode()
    return tf.train.ClusterSpec(aggregate_spec(app, list(iter_tasks(cluster_instances))))


def _setup_skein_cluster(
        pyenvs: Dict[NodeLabel, PythonEnvDescription],
        task_specs: Dict[str, TaskSpec] = TASK_SPEC_NONE,
        *,
        skein_client: skein.Client = None,
        files: Dict[str, str] = None,
        env: Dict[str, str] = {},
        queue: str = "default",
        acls: ACLs = None,
        file_systems: List[str] = None,
        log_conf_file: str = None,
        standalone_client_mode: bool = False
) -> SkeinCluster:
    os.environ["JAVA_TOOL_OPTIONS"] = \
        "-XX:ParallelGCThreads=1 -XX:CICompilerCount=2 "\
        f"{os.environ.get('JAVA_TOOL_OPTIONS', '')}"

    with tempfile.TemporaryDirectory() as tempdir:
        task_files, task_env = _setup_task_env(tempdir, files, env)
        services = {}
        for task_type, task_spec in list(task_specs.items()):
            pyenv = pyenvs[task_spec.label]
            service_env = task_env.copy()
            if task_spec.termination_timeout_seconds >= 0:
                _add_to_env(service_env, "SERVICE_TERMINATION_TIMEOUT_SECONDS",
                            str(task_spec.termination_timeout_seconds))
            services[task_type] = skein.Service(
                script=gen_task_cmd(pyenv, log_conf_file),
                resources=skein.model.Resources(task_spec.memory, task_spec.vcores),
                max_restarts=0,
                instances=task_spec.instances,
                node_label=task_spec.label.value,
                files={
                    **task_files,
                    pyenv.dest_path: pyenv.path_to_archive
                },
                env=service_env)

        spec = skein.ApplicationSpec(
            services,
            queue=queue,
            acls=acls,
            file_systems=file_systems
        )

        if skein_client is None:
            skein_client = skein.Client()

        task_instances = [(task_type, spec.instances) for task_type, spec in task_specs.items()]
        events: Dict[str, Dict[str, str]] = \
            {task: {} for task in iter_tasks(task_instances)}
        app = skein_client.submit_and_connect(spec)
        # Start a thread which collects all events posted by all tasks in kv store
        event_listener = Thread(target=_aggregate_events, args=(app.kv, events))
        event_listener.start()

        cluster_spec = _setup_cluster_tasks(task_instances, app, standalone_client_mode)

        return SkeinCluster(skein_client, app, task_instances, cluster_spec, event_listener, events)


def _run_on_cluster(
    experiment_fn: ExperimentFn,
    cluster: SkeinCluster,
    eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None,
    path_hdfs_logs: str = None
) -> Optional[Metrics]:
    def _new_experiment_fn():
        return add_monitor_to_experiment(experiment_fn())
    new_experiment_fn = _new_experiment_fn

    with _shutdown_on_exception(cluster.app, path_hdfs_logs):
        # Attempt serialization early to avoid allocating unnecesary resources
        serialized_fn = cloudpickle.dumps(new_experiment_fn)
        with cluster.client:
            return _execute_and_await_termination(
                cluster,
                serialized_fn,
                eval_monitor_log_thresholds
            )


def run_on_yarn(
    pyenv_zip_path: Union[str, Dict[NodeLabel, str]],
    experiment_fn: ExperimentFn,
    task_specs: Dict[str, TaskSpec] = None,
    *,
    skein_client: skein.Client = None,
    files: Dict[str, str] = None,
    env: Dict[str, str] = {},
    queue: str = "default",
    acls: ACLs = None,
    file_systems: List[str] = None,
    log_conf_file: str = None,
    eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None,
    path_to_log_hdfs: str = None,
    nb_retries: int = 0
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

    pyenv_zip_path
        Path to an archive of a python environment to be deployed
        It can be a zip conda env or a pex archive
        In case of GPU/CPU cluster, provide a dictionnary with both
        environments.

    experiment_fn
        A function constructing the estimator alongside the train
        and eval specs.

    skein_client
        Skein client used to submit yarn jobs

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

    queue
        YARN queue to use.

    acls
        Configures the application-level Access Control Lists (ACLs).
        Optional, defaults to no ACLs.

        See `ACLs <https://jcrist.github.io/skein/specification.html#id16>` for details.

    file_systems
        A list of namenode URIs to acquire delegation tokens for
        in addition to ``fs.defaultFS``.

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

    path_to_log_hdfs
        Optional path. If specified, tf-yarn will copy hadoop logs into this path

    nb_retries
        Number of times the yarn application is retried in case of failures

    Raises
    ------
    RunFailed
        If the final status of the YARN application is ``"FAILED"``.
    """
    if nb_retries < 0:
        raise ValueError(f'nb_retries must be greater or equal to 0. Got {nb_retries}')

    pyenvs = _setup_pyenvs(
        pyenv_zip_path,
        standalone_client_mode=False)

    n_tries_max = nb_retries + 1
    while True:
        try:
            cluster = _setup_skein_cluster(
                pyenvs=pyenvs,
                skein_client=skein_client,
                task_specs=StaticDefaultDict(task_specs, default=TASK_SPEC_NONE),
                files=files,
                env=env,
                queue=queue,
                acls=acls,
                file_systems=file_systems,
                log_conf_file=log_conf_file,
                standalone_client_mode=False
            )

            return _run_on_cluster(
                experiment_fn, cluster, eval_monitor_log_thresholds, path_to_log_hdfs)
        except Exception:
            n_tries_max -= 1
            if n_tries_max == 0:
                raise
            logger.exception(f"Retrying user application ... {n_tries_max} remaining attempts")

    # Necessary for type checking
    return None


@contextmanager
def standalone_client_mode(
        pyenv_zip_path: Union[str, Dict[NodeLabel, str]],
        task_specs: Dict[str, TaskSpec] = None,
        tf_session_config: Optional[tf.ConfigProto] = None,
        *,
        skein_client: skein.Client = None,
        files: Dict[str, str] = None,
        env: Dict[str, str] = {},
        queue: str = "default",
        acls: ACLs = None,
        file_systems: List[str] = None,
        log_conf_file: str = None):
    """
    https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow \
            /contrib/distribute/README.md#standalone-client-mode
    Standalone mode means starting tf server on the cluster,
    launching everything on the client and letting tf take care of the rest
    This is not limited to Estimator API, it also works with low level tf
    (see session_run_example.py)

    Parameters
    ----------

    pyenv_zip_path
        Path to an archive of a python environment to be deployed
        It can be a zip conda env or a pex archive
        In case of GPU/CPU cluster, provide a dictionnary with both
        environments.

    skein_client
        Skein client to submit yarn jobs

    task_specs
        Resources to allocate for each task type. The keys
        must be a subset of ``"chief"``, ``"worker"``, ``"ps"``, and
        ``"evaluator"``. The minimal spec must contain at least
        ``"chief"``.

    tf_session_config
        tf.ConfigProto to be provided to each started TFServer

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

    acls
        Configures the application-level Access Control Lists (ACLs).
        Optional, defaults to no ACLs.

        See `ACLs <https://jcrist.github.io/skein/specification.html#id16>` for details.

    file_systems
        A list of namenode URIs to acquire delegation tokens for
        in addition to ``fs.defaultFS``.

    log_conf_file
        optional file with log config, setups logging by default with INFO verbosity,
        if you specify a file here don't forget to also ship it to the containers via files arg
    """
    cluster = None
    try:
        pyenvs = _setup_pyenvs(
            pyenv_zip_path,
            standalone_client_mode=True)
        cluster = _setup_skein_cluster(
            pyenvs=pyenvs,
            skein_client=skein_client,
            task_specs=StaticDefaultDict(task_specs, default=TASK_SPEC_NONE),
            files=files,
            env=env,
            queue=queue,
            acls=acls,
            file_systems=file_systems,
            log_conf_file=log_conf_file,
            standalone_client_mode=True
        )
        _send_config_proto(cluster, tf_session_config)

        yield cluster.cluster_spec
    finally:
        if cluster:
            broadcast(cluster.app, "stop", "1")


def get_safe_experiment_fn(full_fn_name: str, *args):
    """
    tf-yarn serializes the provided experiment function with cloudpickle.dumps.
    This is good for interactive experiments but can sometimes fail
    because the function is not serializable.
    You can use this wrapper function
    if you ship your experiment function (via conda, pex) manually to the workers.

    full_fn_name
        the name of the function ( with the full path to package and module)
        i.e. tf_yarn.my_module.my_experiment_fn

    args
        arguments to be provided to this function

    """
    module_name, fn_name = full_fn_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    experiment_fn = getattr(module, fn_name)
    return partial(experiment_fn, *args)


def _send_config_proto(
        cluster: SkeinCluster,
        tf_session_config: tf.ConfigProto):
    serialized_fn = cloudpickle.dumps(tf_session_config)
    cluster.app.kv[KV_TF_SESSION_CONFIG] = serialized_fn


@contextmanager
def _shutdown_on_exception(app: skein.ApplicationClient,
                           path_to_log_hdfs: str = None):
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
    finally:
        if path_to_log_hdfs:
            with tf.gfile.GFile(f'{path_to_log_hdfs}/yarn_logs.txt', 'wb') as fd:
                fd.write(app_logs(app))


def _execute_and_await_termination(
    cluster: SkeinCluster,
    serialized_fn: bytes,
    eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None,
    poll_every_secs: int = 10
) -> Optional[Metrics]:
    cluster.app.kv[KV_EXPERIMENT_FN] = serialized_fn
    eval_metrics_logger = EvaluatorMetricsLogger(
        [task for task in iter_tasks(cluster.tasks) if task.startswith('evaluator')],
        cluster.app,
        eval_monitor_log_thresholds
    )
    one_shot_metrics_logger = OneShotMetricsLogger(
        cluster.app,
        {task: ['url'] for task in iter_tasks(cluster.tasks) if task.startswith('tensorboard')}
    )
    state = None
    while True:
        report = cluster.client.application_report(cluster.app.id)
        logger.info(
            f"Application report for {cluster.app.id} (state: {report.state})")
        if state != report.state:
            logger.info(_format_app_report(report))

        if report.final_status != "undefined":
            cluster.event_listener.join()
            log_events, metrics = _handle_events(cluster.events)
            logger.info(log_events)
            if report.final_status == "failed":
                raise RunFailed
            else:
                break
        else:
            eval_metrics_logger.log()
            one_shot_metrics_logger.log()
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


def app_logs(app: skein.ApplicationClient) -> str:
    command = ["yarn", "logs", "-applicationId", app.id]
    for ind in range(YARN_LOG_TRIES - 1):
        try:
            return subprocess.check_output(command).decode()
        except Exception:
            logger.warn(
                f"Cannot collect logs (attempt {ind}/{YARN_LOG_TRIES})",
                exc_info=True)
        time.sleep(1)
    return subprocess.check_output(command).decode()
