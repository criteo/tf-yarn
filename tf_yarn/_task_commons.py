import argparse
import json
import logging
import logging.config
import os
import re
import sys
from typing import List, Tuple, Dict, Optional

import cloudpickle
import skein
import tensorflow as tf

from tf_yarn import event, cluster, Experiment
from tf_yarn.__init__ import KV_CLUSTER_INSTANCES, KV_EXPERIMENT_FN
from tf_yarn._internal import MonitoredThread, iter_tasks
from tf_yarn.tensorboard import start_tf_board, get_termination_timeout


def _process_arguments() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--log-conf-file", type=str)
    args = parser.parse_args()
    log_conf_file = args.log_conf_file
    if log_conf_file is None:
        base_dir = os.path.dirname(sys.modules["tf_yarn"].__file__)
        log_conf_file = os.path.join(base_dir, "default.log.conf")
    logging.config.fileConfig(log_conf_file, disable_existing_loggers=True)
    tf.logging.info(f"using log file {log_conf_file}")


def _setup_container_logs(client):
    task = cluster.get_task()
    event.broadcast_container_start_time(client, task)
    container = next(c for c in client.get_containers()
                     if c.yarn_container_id == os.environ["CONTAINER_ID"])
    logs = container.yarn_container_logs
    if logs is not None and not logs.startswith("http://"):
        logs = "http://" + logs
    event.logs_event(client, task, logs)


def _prepare_container(
    host_port: Tuple[str, int]
) -> Tuple[skein.ApplicationClient, Dict[str, List[str]], List[str]]:
    """Keep socket open while preparing container """
    tf.logging.info("Python " + sys.version)
    tf.logging.info("Skein " + skein.__version__)
    tf.logging.info(f"TensorFlow {tf.GIT_VERSION} {tf.VERSION}")
    client = skein.ApplicationClient.from_current()
    _setup_container_logs(client)
    cluster_tasks = list(iter_tasks(json.loads(client.kv.wait(KV_CLUSTER_INSTANCES).decode())))
    cluster_spec = cluster.start_cluster(host_port, client, cluster_tasks)
    return client, cluster_spec, cluster_tasks


def _get_experiment(
    client: skein.ApplicationClient
) -> Experiment:
    try:
        experiment = cloudpickle.loads(client.kv.wait(KV_EXPERIMENT_FN))()
    except Exception as e:
        task = cluster.get_task()
        event.start_event(client, task)
        event.stop_event(client, task, e)
        raise
    return experiment


def _gen_monitored_train_and_evaluate(client: skein.ApplicationClient):
    task = cluster.get_task()

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
    experiment: Experiment
) -> MonitoredThread:
    task_type, task_id = cluster.get_task_description()
    tf.logging.info(f"Starting execution {task_type}:{task_id}")
    if task_type == 'tensorboard':
        thread = start_tf_board(client, experiment)
    else:
        thread = MonitoredThread(
            name=f"{task_type}:{task_id}",
            target=_gen_monitored_train_and_evaluate(client),
            args=tuple(experiment),
            daemon=True)
        thread.start()
    task = cluster.get_task()
    event.start_event(client, task)
    return thread


def matches_device_filters(task: str, device_filters: List[str]):
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


def wait_for_connected_tasks(client, all_tasks, device_filters, message='stop'):
    for task in all_tasks:
        if matches_device_filters(task, device_filters):
            event.wait(client, f"{task}/{message}")


def _shutdown_container(
    client: skein.ApplicationClient,
    cluster_tasks: List[str],
    run_config: tf.estimator.RunConfig,
    thread: Optional[MonitoredThread]
) -> None:
    # Wait for all tasks connected to this one. The set of tasks to
    # wait for contains all tasks in the cluster, or the ones
    # matching ``device_filters`` if set. The implementation assumes
    # that ``device_filers`` are symmetric.
    exception = thread.exception if thread is not None and isinstance(thread, MonitoredThread) \
        else None
    task = cluster.get_task()
    event.stop_event(client, task, exception)
    if cluster_tasks is None:
        tasks = None
    else:
        tasks = [c for c in cluster_tasks if not c.startswith('tensorboard')]
    wait_for_connected_tasks(
        client,
        tasks,
        getattr(run_config.session_config, "device_filters", []))

    if task.startswith('tensorboard'):
        timeout = get_termination_timeout()
        if thread is not None:
            thread.join(timeout)
        tf.logging.info(f"{task} finished")
    event.broadcast_container_stop_time(client, task)

    if exception is not None:
        raise exception from None
