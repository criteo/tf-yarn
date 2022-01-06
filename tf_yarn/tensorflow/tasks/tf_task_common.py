import logging
import sys
from typing import Union, List, Optional, Tuple, Dict
import re

import skein
import tensorflow as tf

from tf_yarn import event
from tf_yarn.tensorflow import Experiment, KerasExperiment
from tf_yarn._internal import MonitoredThread
from tf_yarn._task_commons import (
    _setup_container_logs, _get_cluster_tasks, get_task, get_task_description
)
from tf_yarn.tensorflow import cluster


_logger = logging.getLogger(__name__)


def _prepare_container(
    host_port: Tuple[str, int]
) -> Tuple[skein.ApplicationClient, Dict[str, List[str]], List[str]]:
    """Keep socket open while preparing container """
    client = skein.ApplicationClient.from_current()
    _setup_container_logs(client)
    cluster_tasks = _get_cluster_tasks(client)
    cluster_spec = cluster.start_cluster(host_port, client, cluster_tasks)
    return client, cluster_spec, cluster_tasks


def _log_sys_info() -> None:
    _logger.info(f"Python {sys.version}")
    _logger.info(f"Skein {skein.__version__}")
    _logger.info(f"TensorFlow {tf.version.GIT_VERSION} {tf.version.VERSION}")


def _gen_monitored_train_and_evaluate(client: skein.ApplicationClient):
    task = get_task()

    def train_and_evaluate(
            estimator: tf.estimator,
            train_spec: tf.estimator.TrainSpec,
            eval_spec: tf.estimator.EvalSpec):
        event.broadcast_train_eval_start_timer(client, task)
        tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )
        event.broadcast_train_eval_stop_timer(client, task)

    return train_and_evaluate


def _execute_dispatched_function(
    client: skein.ApplicationClient,
    experiment: Union[Experiment, KerasExperiment]
) -> MonitoredThread:
    task_type, task_id = get_task_description()
    _logger.info(f"Starting execution {task_type}:{task_id}")
    if isinstance(experiment, Experiment):
        thread = MonitoredThread(
            name=f"{task_type}:{task_id}",
            target=_gen_monitored_train_and_evaluate(client),
            args=tuple(experiment),
            daemon=True)
    elif isinstance(experiment, KerasExperiment):
        raise ValueError("KerasExperiment using parameter strategy is unsupported")
    else:
        raise ValueError("experiment must be an Experiment or a KerasExperiment")
    thread.start()
    task = get_task()
    event.start_event(client, task)
    return thread


def _shutdown_container(
    client: skein.ApplicationClient,
    cluster_tasks: List[str],
    session_config: tf.compat.v1.ConfigProto,
    thread: Optional[MonitoredThread]
) -> None:
    # Wait for all tasks connected to this one. The set of tasks to
    # wait for contains all tasks in the cluster, or the ones
    # matching ``device_filters`` if set. The implementation assumes
    # that ``device_filers`` are symmetric.
    exception = thread.exception if thread is not None and isinstance(thread, MonitoredThread) \
        else None
    task = get_task()
    event.stop_event(client, task, exception)
    _wait_for_connected_tasks(
        client,
        cluster_tasks,
        getattr(session_config, "device_filters", []))

    event.broadcast_container_stop_time(client, task)

    if exception is not None:
        raise exception from None


def _wait_for_connected_tasks(client, all_tasks, device_filters, message='stop'):
    for task in all_tasks:
        if _matches_device_filters(task, device_filters):
            event.wait(client, f"{task}/{message}")


def _matches_device_filters(task: str, device_filters: List[str]):
    task_type, task_id = task.split(":", 1)
    for device_filter in device_filters:
        [(filter_type, filter_id)] = re.findall(
            r"^/job:([a-z]+)(?:/task:(\d+))?$",
            # Remove once https://github.com/tensorflow/tensorflow/pull/22566 is released
            device_filter.replace("master", "chief"))
        if (filter_type == task_type and
                (not filter_id or filter_id == task_id)):
            return True
    return not device_filters
