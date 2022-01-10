import json
import logging
import os
import typing
from typing import List, NamedTuple

import cloudpickle
import skein

from tf_yarn import event, constants
from tf_yarn._internal import iter_tasks


def setup_logging():
    # tensorflow imports in tf_yarn.__init__ already have set up some loggers
    # erase them with a clean config
    log_conf_file = os.path.join(os.path.dirname(__file__), "default.log.conf")
    logging.config.fileConfig(log_conf_file)


def _setup_container_logs(client):
    task = get_task()
    event.broadcast_container_start_time(client, task)
    container = next(c for c in client.get_containers()
                     if c.yarn_container_id == os.environ["CONTAINER_ID"])
    logs = container.yarn_container_logs
    if logs is not None and not logs.startswith("http://"):
        logs = "http://" + logs
    event.logs_event(client, task, logs)


def _get_cluster_tasks(
    client: skein.ApplicationClient
) -> List[str]:
    return list(iter_tasks(json.loads(client.kv.wait(constants.KV_CLUSTER_INSTANCES).decode())))


def _get_experiment(
    client: skein.ApplicationClient
) -> NamedTuple:
    try:
        experiment = cloudpickle.loads(client.kv.wait(constants.KV_EXPERIMENT_FN))()
    except Exception as e:
        task = get_task()
        event.start_event(client, task)
        event.stop_event(client, task, e)
        raise
    return experiment


def n_try() -> int:
    return int(os.getenv("TF_YARN_N_TRY", "0"))


def get_task() -> str:
    return os.getenv("SKEIN_CONTAINER_ID", "").replace("_", ":", 1)


def get_task_type(task: str) -> str:
    parts = task.split(':')
    if len(parts) > 0:
        return parts[0]
    else:
        return ""


def is_worker(task_type: str = None) -> bool:
    if not task_type:
        task_type = get_task_type(get_task())

    return task_type == 'worker'


def is_evaluator(task_type: str = None) -> bool:
    if not task_type:
        task_type = get_task_type(get_task())

    return task_type == 'evaluator'


def is_chief(task_type: str = None) -> bool:
    if not task_type:
        task_type = get_task_type(get_task())

    return task_type == 'chief'


def get_task_description() -> typing.Tuple[str, int]:
    task = get_task()
    task_type, task_str = task.split(":", 1)
    task_id: int = int(task_str)
    return task_type, task_id
