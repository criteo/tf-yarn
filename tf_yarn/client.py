
import importlib
import logging
import logging.config
import uuid
import os
import tempfile
import time

import cloudpickle
import json
import skein
import tensorflow as tf

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

from threading import Thread
from datetime import timedelta
from skein.exceptions import SkeinError
from skein.model import FinalStatus, ApplicationReport, ACLs

import cluster_pack

from tf_yarn import (
    _env,
    _internal,
    cluster,
    constants,
    metrics,
    evaluator_metrics,
    mlflow,
    tensorboard,
    event,
    experiment,
    keras_experiment,
    topologies
)

YARN_LOG_TRIES = 15

ExperimentFn = Callable[[], experiment.Experiment]

KerasExperimentFn = Callable[[], keras_experiment.KerasExperiment]


TASK_SPEC_NONE = topologies.single_server_topology()

logger = logging.getLogger(__name__)

here = os.path.dirname(__file__)


class SkeinCluster(NamedTuple):
    client: skein.Client
    app: skein.ApplicationClient
    tasks: List[Tuple[str, int]]
    event_listener: Thread
    events: Dict[str, Dict[str, str]]


class ContainerLogStatus(NamedTuple):
    log_urls: Dict[str, str] = dict()
    container_status: Dict[str, str] = dict()

    def by_container_id(self) -> Dict[str, Tuple[str, str]]:
        containers: Dict[str, Tuple[str, str]] = {}
        if len(self.log_urls.keys()) != len(self.container_status.keys()):
            logger.warning("logs_urls and container_status dicts have not the same length")
            return containers

        for task, url, status in zip(self.log_urls.keys(),
                                     self.log_urls.values(),
                                     self.container_status.values()):
            container_id = self._get_container_id(url)
            containers[container_id] = (task, status)

        return containers

    def _get_container_id(self, url: str) -> str:
        if not url:
            return ""
        url_components = url.split("/")
        if len(url_components) > 1:
            return url_components[-2]

        return ""


class RunFailed(Exception):
    """``run_on_yarn`` failed."""


def _setup_pyenvs(
        pyenv_zip_path: Union[str, Dict[topologies.NodeLabel, str]]
) -> Dict[topologies.NodeLabel, _env.PythonEnvDescription]:
    if isinstance(pyenv_zip_path, str):
        pyenv = _env.gen_pyenv_from_existing_archive(pyenv_zip_path)
        pyenvs = {
            topologies.NodeLabel.CPU: pyenv,
            topologies.NodeLabel.GPU: pyenv
        }
    else:
        pyenvs = {label: _env.gen_pyenv_from_existing_archive(env_zip_path)
                  for label, env_zip_path in pyenv_zip_path.items()}
    return pyenvs


def _setup_task_env(
        tempdir: str,
        files: Dict[str, str] = None,
        env: Dict[str, str] = {},
        n_try: int = 0
):
    task_files = _maybe_zip_task_files(files or {}, tempdir)
    task_files[__package__] = cluster_pack.zip_path(here, False, tempdir)

    _add_to_env(env, "LIBHDFS_OPTS", "-Xms64m -Xmx512m")

    env["TF_YARN_N_TRY"] = str(n_try)

    task_env = {
        **env,
        # Make Python modules/packages passed via ``files`` importable.
        "PYTHONPATH": ".:" + env.get("PYTHONPATH", ""),
        "PEX_ROOT": os.path.join("/tmp", str(uuid.uuid4()))
    }

    if mlflow.use_mlflow:
        task_env["MLFLOW_RUN_ID"] = mlflow.active_run_id()
        task_env["MLFLOW_TRACKING_URI"] = mlflow.get_tracking_uri()
        task_env["GIT_PYTHON_REFRESH"] = "quiet"

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
            source = cluster_pack.zip_path(source, False, tempdir)

        task_files[target] = source
    return task_files


def _setup_cluster_spec(
    task_instances: List[Tuple[str, int]],
    app: skein.ApplicationClient
) -> tf.train.ClusterSpec:
    tasks_not_in_cluster = ['evaluator', 'tensorboard']
    cluster_instances = [t for t in task_instances if t[0] not in tasks_not_in_cluster]
    app.kv[constants.KV_CLUSTER_INSTANCES] = json.dumps(cluster_instances).encode()
    return tf.train.ClusterSpec(
        cluster.aggregate_spec(app, list(_internal.iter_tasks(cluster_instances)))
    )


def _setup_skein_cluster(
        pyenvs: Dict[topologies.NodeLabel, _env.PythonEnvDescription],
        task_specs: Dict[str, topologies.TaskSpec] = TASK_SPEC_NONE,
        *,
        custom_task_module: Optional[str] = None,
        skein_client: skein.Client = None,
        files: Dict[str, str] = None,
        env: Dict[str, str] = {},
        queue: str = "default",
        acls: ACLs = None,
        file_systems: List[str] = None,
        name: str = "RunOnYarn",
        n_try: int = 0,
        pre_script_hook: Optional[str] = None
) -> SkeinCluster:
    os.environ["JAVA_TOOL_OPTIONS"] = \
        "-XX:ParallelGCThreads=1 -XX:CICompilerCount=2 "\
        f"{os.environ.get('JAVA_TOOL_OPTIONS', '')}"

    pre_script_hook = pre_script_hook if pre_script_hook else ""
    with tempfile.TemporaryDirectory() as tempdir:
        task_files, task_env = _setup_task_env(tempdir, files, env, n_try)
        services = {}
        for task_type, task_spec in list(task_specs.items()):
            pyenv = pyenvs[task_spec.label]
            service_env = task_env.copy()
            if task_spec.tb_termination_timeout_seconds >= 0:
                service_env["TB_TERMINATION_TIMEOUT_SECONDS"] = \
                    str(task_spec.tb_termination_timeout_seconds)
            if task_spec.tb_model_dir:
                service_env["TB_MODEL_DIR"] = str(task_spec.tb_model_dir)
            if task_spec.tb_extra_args:
                service_env["TB_EXTRA_ARGS"] = str(task_spec.tb_extra_args)

            services[task_type] = skein.Service(
                script=f'''
                            set -x
                            {pre_script_hook}
                            {_env.gen_task_cmd(
                                pyenv,
                                task_type,
                                custom_task_module)}
                        ''',
                resources=skein.model.Resources(task_spec.memory, task_spec.vcores),
                max_restarts=0,
                instances=task_spec.instances,
                node_label=task_spec.label.value,
                files={
                    **task_files,
                    pyenv.dest_path: pyenv.path_to_archive
                },
                env=service_env)

        # on the cluster we don't ask again for delegation tokens
        if "HADOOP_TOKEN_FILE_LOCATION" in os.environ:
            file_systems = None

        spec = skein.ApplicationSpec(
            services,
            queue=queue,
            acls=acls,
            file_systems=file_systems,
            name=name
        )

        if skein_client is None:
            skein_client = skein.Client()

        task_instances = [(task_type, spec.instances) for task_type, spec in task_specs.items()]
        events: Dict[str, Dict[str, str]] = \
            {task: {} for task in _internal.iter_tasks(task_instances)}
        app = skein_client.submit_and_connect(spec)

        # Start a thread which collects all events posted by all tasks in kv store
        event_listener = Thread(target=_aggregate_events, args=(app.kv, events))
        event_listener.start()

        return SkeinCluster(skein_client, app, task_instances, event_listener, events)


def _hook_name_already_exists(
        hook: tf.estimator.SessionRunHook,
        hooks: List[tf.estimator.SessionRunHook]) -> bool:
    hook_name = type(hook).__name__
    return len([h for h in hooks
                if type(h).__name__ == hook_name]) > 0


def _add_monitor_to_experiment(
    my_experiment: Union[experiment.Experiment, keras_experiment.KerasExperiment]
) -> Union[experiment.Experiment, keras_experiment.KerasExperiment]:
    if isinstance(my_experiment, experiment.Experiment):
        logger.info(f"configured training hooks: {my_experiment.train_spec.hooks}")

        training_hooks = list(my_experiment.train_spec.hooks)

        if my_experiment.config.log_step_count_steps is not None:
            steps_per_second_hook = metrics.StepPerSecondHook(
                every_n_steps=my_experiment.config.log_step_count_steps
            )
            if not _hook_name_already_exists(steps_per_second_hook, training_hooks):
                training_hooks.append(steps_per_second_hook)
            else:
                logger.warning("do not add StepPerSecondHook as there is already one configured")

        monitored_train_spec = my_experiment.train_spec._replace(
            hooks=training_hooks
        )

        monitored_eval_spec = my_experiment.eval_spec._replace(
            hooks=(evaluator_metrics.EvalMonitorHook(), *my_experiment.eval_spec.hooks)
        )

        my_experiment = my_experiment._replace(
            eval_spec=monitored_eval_spec, train_spec=monitored_train_spec)
    elif isinstance(my_experiment, keras_experiment.KerasExperiment):
        logger.warning("equivalent of StepPerSecondHook not yet implemented for KerasExperiment")
    else:
        raise ValueError("experiment must be an Experiment or a KerasExperiment")
    return my_experiment


def _add_monitor_to_keras_experiment(
    experiment: keras_experiment.KerasExperiment
) -> keras_experiment.KerasExperiment:
    return experiment


def _run_on_cluster(
    experiment_fn: Union[ExperimentFn, KerasExperimentFn],
    skein_cluster: SkeinCluster,
    eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None,
    n_try: int = 0
) -> Optional[metrics.Metrics]:
    def _new_experiment_fn():
        return _add_monitor_to_experiment(experiment_fn())

    new_experiment_fn = _new_experiment_fn

    # Attempt serialization early to avoid allocating unnecesary resources
    serialized_fn = cloudpickle.dumps(new_experiment_fn)
    with skein_cluster.client:
        return _execute_and_await_termination(
            skein_cluster,
            serialized_fn,
            eval_monitor_log_thresholds,
            n_try=n_try
        )


def _default_acls_all_access() -> skein.model.ACLs:
    return skein.model.ACLs(
        enable=True,
        ui_users=['*'],
        view_users=['*']
    )


def run_on_yarn(
    pyenv_zip_path: Union[str, Dict[topologies.NodeLabel, str]],
    experiment_fn: Union[ExperimentFn, KerasExperimentFn],
    task_specs: Dict[str, topologies.TaskSpec] = TASK_SPEC_NONE,
    *,
    skein_client: skein.Client = None,
    files: Dict[str, str] = None,
    env: Dict[str, str] = {},
    queue: str = "default",
    acls: ACLs = _default_acls_all_access(),
    file_systems: List[str] = None,
    eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None,
    nb_retries: int = 0,
    custom_task_module: Optional[str] = None,
    name: str = "RunOnYarn",
    pre_script_hook: Optional[str] = None
) -> Optional[metrics.Metrics]:
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
        Optional, defaults to ACLs all access.

        See `ACLs <https://jcrist.github.io/skein/specification.html#acls>` for details.

    file_systems
        A list of namenode URIs to acquire delegation tokens for
        in addition to ``fs.defaultFS``.

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

    nb_retries
        Number of times the yarn application is retried in case of failures

    custom_task_module
        Provide the full module name of a custom task that is executed on each worker
        None by default
        (Module will be invoked with python -m {custom_task_module} on the cluster)
        Only for advanced use cases, can be useful for example,
        to bypass/tweek the existing estimator.train_and_evaluate pattern

    name
        Name of the yarn application

    pre_script_hook
        bash command to prepare Hadoop environment

    Raises
    ------
    RunFailed
        If the final status of the YARN application is ``"FAILED"``.
    """
    if nb_retries < 0:
        raise ValueError(f'nb_retries must be greater or equal to 0. Got {nb_retries}')

    pyenvs = _setup_pyenvs(pyenv_zip_path)

    n_try = 0
    while True:
        try:
            skein_cluster = _setup_skein_cluster(
                pyenvs=pyenvs,
                task_specs=task_specs,
                skein_client=skein_client,
                files=files,
                env=env,
                queue=queue,
                acls=acls,
                file_systems=file_systems,
                name=name,
                n_try=n_try,
                custom_task_module=custom_task_module,
                pre_script_hook=pre_script_hook
            )
            with _shutdown_on_exception(skein_cluster.app):
                _setup_cluster_spec(skein_cluster.tasks, skein_cluster.app)

                return _run_on_cluster(
                    experiment_fn,
                    skein_cluster,
                    eval_monitor_log_thresholds,
                    n_try
                )
        except Exception:
            n_try += 1
            if n_try == nb_retries + 1:
                raise
            logger.exception(f"Retrying user application ... "
                             f"{nb_retries + 1 - n_try} remaining attempts")

    # Necessary for type checking
    return None


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

    def _safe_exp_fn():
        module = importlib.import_module(module_name)
        experiment_fn = getattr(module, fn_name)
        return experiment_fn(*args)

    return _safe_exp_fn


def _send_config_proto(
        skein_cluster: SkeinCluster,
        tf_session_config: tf.compat.v1.ConfigProto):
    serialized_fn = cloudpickle.dumps(tf_session_config)
    skein_cluster.app.kv[constants.KV_TF_SESSION_CONFIG] = serialized_fn


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
    skein_cluster: SkeinCluster,
    serialized_fn: bytes,
    eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None,
    n_try: int = 0,
    poll_every_secs: int = 10
) -> Optional[metrics.Metrics]:
    skein_cluster.app.kv[constants.KV_EXPERIMENT_FN] = serialized_fn
    eval_metrics_logger = evaluator_metrics.EvaluatorMetricsLogger(
        [task for task in _internal.iter_tasks(skein_cluster.tasks)
         if task.startswith('evaluator')],
        skein_cluster.app,
        eval_monitor_log_thresholds
    )

    tensorboard_url_event_name = tensorboard.url_event_name(
        _internal.iter_tasks(skein_cluster.tasks)
    )
    tensorboard_url_logger = metrics.OneShotMetricsLogger(
        skein_cluster.app,
        [(tensorboard_url_event_name, tensorboard.URL_EVENT_LABEL)]
        if tensorboard_url_event_name else [],
        n_try
    )

    state = None
    while True:
        report = skein_cluster.client.application_report(skein_cluster.app.id)
        logger.info(
            f"Application report for {skein_cluster.app.id} (state: {report.state})")
        if state != report.state:
            logger.info(_format_app_report(report))

        if report.final_status != "undefined":
            skein_cluster.event_listener.join()
            log_events, result_metrics, container_status = _handle_events(skein_cluster.events,
                                                                          n_try)
            logger.info(log_events)

            containers = container_status.by_container_id()
            # add one for AM container
            wait_for_nb_logs = sum([instances for task, instances in skein_cluster.tasks]) + 1
            logs = _get_app_logs(
                skein_cluster.client,
                skein_cluster.app,
                wait_for_nb_logs
            )
            _save_logs_to_mlflow(logs, containers, n_try)

            if report.final_status == "failed":
                raise RunFailed
            else:
                break
        else:
            eval_metrics_logger.log()
            tensorboard_url_logger.log()
        time.sleep(poll_every_secs)
        state = report.state

    result_metrics.log_mlflow(n_try)
    return result_metrics


def _save_logs_to_mlflow(logs: Optional[skein.model.ApplicationLogs],
                        containers: Dict[str, Tuple[str, str]],
                        n_try: int):
    if not logs:
        return

    for key, logs in logs.items():
        if key in containers:
            task, status = containers[key]
            filename = mlflow.format_key(f"{task}_{status}_{n_try}")
        else:
            filename = mlflow.format_key(f"{key}_{n_try}")
        mlflow.save_text_to_mlflow(logs, filename)


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
        for evt in queue:
            if "/" in evt.key:
                task, stage = evt.key.rsplit("/", 1)
                events[task][stage] = evt.result.value.decode()


def _handle_events(
    events: Dict[str, Dict[str, str]],
    n_try: int
) -> Tuple[str, metrics.Metrics, ContainerLogStatus]:
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
    container_log_urls: Dict[str, str] = dict()
    container_status: Dict[str, str] = dict()
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

        container_log_urls[task] = logs
        container_status[task] = status
        container_duration[task] = None
        if 'container_start_time' in stages and 'container_stop_time' in stages:
            container_duration[task] = timedelta(seconds=(float(stages['container_stop_time'])
                                                          - float(stages['container_start_time'])))

        train_eval_time_per_node[task] = None
        task_type = cluster.get_task_type(task)
        if 'train_eval_start_time' in stages and 'train_eval_stop_time' in stages and not exception:
            start_time = timedelta(seconds=float(stages['train_eval_start_time']))
            stop_time = timedelta(seconds=float(stages['train_eval_stop_time']))
            train_eval_time_per_node[task] = stop_time - start_time
            if cluster.is_worker(task_type) or cluster.is_chief(task_type):
                if start_time < min_training_start_time:
                    min_training_start_time = start_time
                if stop_time > max_training_stop_time:
                    max_training_stop_time = stop_time
            elif cluster.is_evaluator(task_type):
                if start_time < min_eval_start_time:
                    min_eval_start_time = start_time
                if stop_time > max_eval_stop_time:
                    max_eval_stop_time = stop_time
        else:
            if cluster.is_worker(task_type) or cluster.is_chief(task_type):
                valid_training_time = False
            elif cluster.is_evaluator(task_type):
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

    result_metrics = metrics.Metrics(
        training_time,
        eval_time,
        container_duration,
        train_eval_time_per_node
    )

    return ((os.linesep + os.linesep.join(header)
             + os.linesep * (1 + bool(details))
             + os.linesep.join(details)),
            result_metrics,
            ContainerLogStatus(container_log_urls, container_status))


def _get_app_logs(
    client: skein.Client,
    app: skein.ApplicationClient,
    wait_for_nb_logs: int
) -> Optional[skein.model.ApplicationLogs]:
    for ind in range(YARN_LOG_TRIES):
        try:
            logs = client.application_logs(app.id)
            nb_keys = len(logs.keys())
            logger.info(f"Got {nb_keys}/{wait_for_nb_logs} log files")
            if nb_keys == wait_for_nb_logs:
                return logs
        except Exception:
            logger.warn(
                f"Cannot collect logs (attempt {ind+1}/{YARN_LOG_TRIES})",
                exc_info=True)
        time.sleep(3)
    return None
