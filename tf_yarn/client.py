import importlib
import logging.config
import uuid
import os
import tempfile
import time
import json
from typing import (
    Dict,
    Optional,
    Tuple,
    NamedTuple,
    Callable,
    Union,
    List,
    Any
)
from contextlib import suppress, contextmanager
from threading import Thread
from datetime import timedelta

import cloudpickle
import skein
from cluster_pack import get_pyenv_usage_from_archive
from cluster_pack.packaging import PythonEnvDescription
from skein.exceptions import SkeinError
from skein.model import FinalStatus, ApplicationReport, ACLs
import cluster_pack

from tf_yarn import (
    _env,
    _internal,
    constants,
    metrics,
    evaluator_metrics,
    mlflow,
    topologies
)
from tf_yarn._task_commons import is_chief, is_evaluator, is_worker, catchtime
from tf_yarn import tensorboard
from tf_yarn._criteo import is_criteo
from tf_yarn.topologies import ContainerKey

YARN_LOG_TRIES = 15

logger = logging.getLogger(__name__)

here = os.path.dirname(__file__)

ExperimentFn = Union[Callable[[], Any], Callable[[int], Any]]


class SkeinCluster(NamedTuple):
    client: skein.Client
    app: skein.ApplicationClient
    tasks: List[Tuple[str, int, int]]
    event_listener: Thread
    events: Dict[ContainerKey, Dict[str, str]]


class ContainerLogStatus(NamedTuple):
    log_urls: Dict[ContainerKey, str] = dict()
    container_status: Dict[ContainerKey, str] = dict()

    def by_container_id(self) -> Dict[str, Tuple[ContainerKey, str]]:
        containers: Dict[str, Tuple[ContainerKey, str]] = {}
        if len(self.log_urls.keys()) != len(self.container_status.keys()):
            logger.warning("logs_urls and container_status dicts have not the same length")
            return containers

        for task_key, url, status in zip(self.log_urls.keys(),
                                         self.log_urls.values(),
                                         self.container_status.values()):
            container_id = self._get_container_id(url)
            containers[container_id] = (task_key, status)

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
) -> Dict[topologies.NodeLabel, PythonEnvDescription]:
    if isinstance(pyenv_zip_path, str):
        pyenv = get_pyenv_usage_from_archive(pyenv_zip_path)
        pyenvs = {
            topologies.NodeLabel.CPU: pyenv,
            topologies.NodeLabel.GPU: pyenv
        }
    else:
        pyenvs = {label: get_pyenv_usage_from_archive(env_zip_path)
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
        "PEX_ROOT": _get_pex_root(),
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


def _setup_to_use_cuda_archive(
        env: Dict[str, str],
        pre_script_hook: str,
        cuda_runtime_hdfs_path: str
) -> str:
    if "LD_LIBRARY_PATH" not in env:
        logger.warning("No LD_LIBRARY_PATH found in env. \
             Cuda archive won't be shipped and tensorflow will fallback to CPU usage.")
        return pre_script_hook
    cuda_archive_name = cuda_runtime_hdfs_path.split("/")[-1]
    cuda_script = f"hdfs dfs -get {cuda_runtime_hdfs_path}; \
         mkdir cuda; tar -xf {cuda_archive_name} -C ./cuda;"
    pre_script_hook = f"{cuda_script}{pre_script_hook}"
    return pre_script_hook


def _setup_cluster_spec(
        task_instances: List[Tuple[str, int, int]],
        app: skein.ApplicationClient
) -> None:
    tasks_not_in_cluster = ['evaluator', 'tensorboard']
    cluster_instances = [t for t in task_instances if t[0] not in tasks_not_in_cluster]
    app.kv[constants.KV_CLUSTER_INSTANCES] = json.dumps(cluster_instances).encode()


def _setup_skein_cluster(
        pyenvs: Dict[topologies.NodeLabel, PythonEnvDescription],
        task_specs: Dict[str, topologies.TaskSpec],
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
        pre_script_hook: Optional[str] = None,
        cuda_runtime_hdfs_path: Optional[str] = None
) -> SkeinCluster:
    os.environ["JAVA_TOOL_OPTIONS"] = \
        "-XX:ParallelGCThreads=1 -XX:CICompilerCount=2 " \
        f"{os.environ.get('JAVA_TOOL_OPTIONS', '')}"

    pre_script_hook = pre_script_hook if pre_script_hook else ""

    if cuda_runtime_hdfs_path:
        pre_script_hook = _setup_to_use_cuda_archive(env, pre_script_hook, cuda_runtime_hdfs_path)

    with tempfile.TemporaryDirectory() as tempdir:
        print('setting up task files and environment')
        with catchtime():
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

            print(f'setting up skein service "{task_type}')
            with catchtime():
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

        container_info = [(task_type, spec.instances, spec.nb_proc_per_worker)
                          for task_type, spec in task_specs.items()]
        events: Dict[ContainerKey, Dict[str, str]] = {
            task.to_container_key(): {} for task in _internal.iter_tasks(container_info)}
        print('submitting skein Application')
        with catchtime():
            app = skein_client.submit_and_connect(spec)

        # Start a thread which collects all events posted by all tasks in kv store
        event_listener = Thread(target=_aggregate_events, args=(app.kv, events), daemon=True)
        event_listener.start()

        return SkeinCluster(skein_client, app, container_info, event_listener, events)


def _run_on_cluster(
        experiment_fn: ExperimentFn,
        skein_cluster: SkeinCluster,
        eval_monitor_log_thresholds: Dict[str, Tuple[float, float]] = None,
        n_try: int = 0
) -> Optional[metrics.Metrics]:
    # Attempt serialization early to avoid allocating unnecesary resources
    print('Serializing experiment function ...')
    with catchtime():
        serialized_fn = cloudpickle.dumps(experiment_fn)
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
        experiment_fn: ExperimentFn,
        task_specs: Dict[str, topologies.TaskSpec],
        *,
        pyenv_zip_path: Union[str, Dict[topologies.NodeLabel, str]] = None,
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
        pre_script_hook: Optional[str] = None,
        cuda_runtime_hdfs_path: Optional[str] = None

) -> Optional[metrics.Metrics]:
    """Run an experiment on YARN.

    The implementation allocates a service with the requested number
    of instances for each distributed task type. Each
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
        environments. If none is provided, the current python environment
        will be packaged in a pex

    skein_client
        Skein client used to submit yarn jobs

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

    cuda_runtime_hdfs_path
        Path to the cuda runtime archive you want to ship on the cluster.
        This is usefull to run tensorflow learning on gpu.
        When using this option env must contain LD_LIBRARY_PATH key pointing to path
        where cuda libs will be once the archive is unpacked in ./cuda
        (e.g. 'cuda/usr/cuda-11.2/lib64' )

    Raises
    ------
    RunFailed
        If the final status of the YARN application is ``"FAILED"``.
    """
    updated_files = _add_editable_requirements(files)
    _pyenv_zip_path = pyenv_zip_path
    if _pyenv_zip_path is None:
        print("building and uploading venv")
        with catchtime():
            _pyenv_zip_path = cluster_pack.upload_env()[0]

    if nb_retries < 0:
        raise ValueError(f'nb_retries must be greater or equal to 0. Got {nb_retries}')

    pyenvs = _setup_pyenvs(_pyenv_zip_path)

    n_try = 0
    while True:
        print(f'Try: {n_try +1}')
        try:
            with catchtime():
                print('Setupping skein cluster')
                skein_cluster = _setup_skein_cluster(
                    pyenvs=pyenvs,
                    task_specs=task_specs,
                    skein_client=skein_client,
                    files=updated_files,
                    env=env,
                    queue=queue,
                    acls=acls,
                    file_systems=file_systems,
                    name=name,
                    n_try=n_try,
                    custom_task_module=custom_task_module,
                    pre_script_hook=pre_script_hook,
                    cuda_runtime_hdfs_path=cuda_runtime_hdfs_path
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


def _add_editable_requirements(files: Optional[Dict[str, str]]):
    editable_requirements = cluster_pack.get_editable_requirements()
    if files is None:
        files = dict()
    for dirname, path in editable_requirements.items():
        if dirname not in files:
            files[dirname] = path
    return files


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
    logger.info("posting serialized Experiment fcn to skein KV store")
    with catchtime():
        skein_cluster.app.kv[constants.KV_EXPERIMENT_FN] = serialized_fn
    eval_metrics_logger = evaluator_metrics.EvaluatorMetricsLogger(
        [task for task in _internal.iter_tasks(skein_cluster.tasks)
         if task.type == 'evaluator'],
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
    container_log_urls: Dict[str, str] = {}

    while True:
        report = skein_cluster.client.application_report(skein_cluster.app.id)
        logger.info(
            f"Application report for {skein_cluster.app.id} (state: {report.state})")
        if state != report.state:
            logger.info(_format_app_report(report))

        if state == "running":
            try:
                for key in skein_cluster.app.kv.keys():
                    if '/logs' in key and key not in container_log_urls.keys():
                        container_log_urls[key] = skein_cluster.app.kv.wait(key).decode()
            except Exception:
                pass  # accessing the kv if the app is not ready or closed raises, we just ignore

        if report.final_status != "undefined":
            skein_cluster.event_listener.join()
            log_events, result_metrics, container_status = _handle_events(skein_cluster.events,
                                                                          n_try)
            logger.info(log_events)
            logger.info('container logs urls:')
            for v in container_log_urls.values():
                logger.info(v)

            containers = container_status.by_container_id()
            # add one for AM container
            wait_for_nb_logs = sum([instances for task, instances, _ in skein_cluster.tasks]) + 1
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
                         containers: Dict[str, Tuple[ContainerKey, str]],
                         n_try: int):
    if not logs:
        return

    for key, logs in logs.items():
        if key in containers:
            task, status = containers[key]
            filename = mlflow.format_key(f"{task.to_kv_str()}_{status}_{n_try}")
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
        events: Dict[ContainerKey, Dict[str, str]]
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
                task_key_str, stage = evt.key.rsplit("/", 1)
                task = ContainerKey.from_kv_str(task_key_str)
                try:
                    events[task][stage] = evt.result.value.decode()
                except KeyError:
                    print(f'faulty key: {task}')
                    print(events)


def _handle_events(
        events: Dict[ContainerKey, Dict[str, str]],
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
    container_duration: Dict[ContainerKey, Optional[timedelta]] = dict()
    train_eval_time_per_node: Dict[ContainerKey, Optional[timedelta]] = dict()
    container_log_urls: Dict[ContainerKey, str] = dict()
    container_status: Dict[ContainerKey, str] = dict()
    for task_key, stages in sorted(events.items()):
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

        container_log_urls[task_key] = logs
        container_status[task_key] = status
        container_duration[task_key] = None
        if 'container_start_time' in stages and 'container_stop_time' in stages:
            container_duration[task_key] = timedelta(
                seconds=(float(stages['container_stop_time']) - float(stages['container_start_time']
                                                                      )))

        train_eval_time_per_node[task_key] = None

        if 'train_eval_start_time' in stages and 'train_eval_stop_time' in stages and not exception:
            start_time = timedelta(seconds=float(stages['train_eval_start_time']))
            stop_time = timedelta(seconds=float(stages['train_eval_stop_time']))
            train_eval_time_per_node[task_key] = stop_time - start_time
            if is_worker(task_key.type) or is_chief(task_key.type):
                if start_time < min_training_start_time:
                    min_training_start_time = start_time
                if stop_time > max_training_stop_time:
                    max_training_stop_time = stop_time
            elif is_evaluator(task_key.type):
                if start_time < min_eval_start_time:
                    min_eval_start_time = start_time
                if stop_time > max_eval_stop_time:
                    max_eval_stop_time = stop_time
        else:
            if is_worker(task_key.type) or is_chief(task_key.type):
                valid_training_time = False
            elif is_evaluator(task_key.type):
                valid_eval_time = False

        header.append(f"{task_key.to_kv_str():>16}  {sock_addr}  {status}  {logs}"
                      f"  Container duration: {container_duration[task_key]}"
                      f"  Training/evaluation duration : {train_eval_time_per_node[task_key]}")

        if exception:
            details.append(f"Exception in task {task_key.to_kv_str()}:")
            details.append(exception)

    training_time = max_training_stop_time - min_training_start_time \
        if valid_training_time and min_training_start_time < timedelta.max else None
    eval_time = max_eval_stop_time - min_eval_start_time \
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
                f"Cannot collect logs (attempt {ind + 1}/{YARN_LOG_TRIES})",
                exc_info=True)
        time.sleep(3)
    return None


def _get_pex_root() -> str:
    if "PEX_ROOT" in os.environ:
        return os.environ["PEX_ROOT"]
    random_str = str(uuid.uuid4())
    if is_criteo():
        # Use current working directory because /tmp is a shared folder with limited space
        return f"./{random_str}"
    else:
        return os.path.join("/tmp", random_str)
