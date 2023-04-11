from typing import Callable, NamedTuple, Tuple
import sys
import logging
import skein

from tf_yarn._task_commons import (
    _get_cluster_tasks,
    _get_experiment,
    choose_master,
    get_task_description,
    setup_logging,
)


setup_logging()
_logger = logging.getLogger(__name__)


def _log_sys_info() -> None:
    _logger.info(f"Python {sys.version}")
    _logger.info(f"Skein {skein.__version__}")


class TaskParameters(NamedTuple):
    task_type: str
    task_id: int
    world_size: int
    master_address: str
    master_port: int
    n_workers_per_executor: int = 1


def get_task():
    task_type, task_id = get_task_description()

    client = skein.ApplicationClient.from_current()
    addr, port = choose_master(client, task_id)
    _logger.info(f'master: {addr}:{port}')
    cluster_tasks = _get_cluster_tasks(client)
    n_workers_per_executor = 1

    world_size = len([t for t in cluster_tasks if "worker" in t]) * n_workers_per_executor
    _logger.info(
        f"Task type: {task_type}; Task id: {task_id};"
        f"World_size: {world_size}: Cluster tasks: {cluster_tasks}"
    )
    return TaskParameters(task_type, task_id, world_size, addr, port, n_workers_per_executor)


def main():
    _log_sys_info()
    client = skein.ApplicationClient.from_current()
    experiment = _get_experiment(client)
    assert isinstance(experiment, Callable)
    experiment()


if __name__ == "__main__":
    main()
