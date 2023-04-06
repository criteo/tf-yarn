from typing import NamedTuple, Tuple
import sys
import logging
import skein

from tf_yarn import _internal, event
from tf_yarn._task_commons import (
    _get_cluster_tasks,
    _get_experiment,
    get_task_description,
    setup_logging,
)

MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = "MASTER_PORT"

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


def setup_master(client: skein.ApplicationClient, rank: int) -> Tuple[str, int]:
    if rank == 0:
        # ideally launching the train function on the master node should happen inside this context
        # manager, but existing tf-yarn jobs run correctly with the port reservation as is
        with _internal.reserve_sock_addr() as host_port:
            master_addr = host_port[0]
            master_port = host_port[1]
            event.broadcast(client, MASTER_ADDR, master_addr)
            event.broadcast(client, MASTER_PORT, str(master_port))
    else:
        master_addr = event.wait(client, MASTER_ADDR)
        master_port = int(event.wait(client, MASTER_PORT))

    _logger.info(f'master: {master_addr}:{master_port}')
    return master_addr, master_port


def get_task():
    task_type, task_id = get_task_description()

    client = skein.ApplicationClient.from_current()
    addr, port = setup_master(client, task_id)
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
    experiment()


if __name__ == "__main__":
    main()
