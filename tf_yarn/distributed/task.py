from typing import NamedTuple
import os
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

_logger = logging.getLogger(__name__)

def _log_sys_info() -> None:
    _logger.info(f"Python {sys.version}")
    _logger.info(f"Skein {skein.__version__}")
    

class TaskParameters(NamedTuple):
    task_type: str 
    task_id: int
    world_size:int
    n_workers_per_executor: int = 1

    
def setup_master(client: skein.ApplicationClient, rank: int) -> None:
    if rank == 0:
        with _internal.reserve_sock_addr() as host_port:
            event.broadcast(client, MASTER_ADDR, host_port[0])
            event.broadcast(client, MASTER_PORT, str(host_port[1]))
            os.environ[MASTER_ADDR] = host_port[0]
            os.environ[MASTER_PORT] = str(host_port[1])
    else:
        master_addr = event.wait(client, MASTER_ADDR)
        master_port = event.wait(client, MASTER_PORT)
        os.environ[MASTER_ADDR] = master_addr
        os.environ[MASTER_PORT] = master_port
    
    

def get_task():
    task_type, task_id = get_task_description()

    client = skein.ApplicationClient.from_current()
    setup_master(client, task_id)
    cluster_tasks = _get_cluster_tasks(client)
    n_workers_per_executor = 1

    world_size = len([t for t in cluster_tasks if "worker" in t]) * n_workers_per_executor
    _logger.info(
        f"Task type: {task_type}; Task id: {task_id};"
        f"World_size: {world_size}: Cluster tasks: {cluster_tasks}"
    )
    return TaskParameters(task_type, task_id, world_size, n_workers_per_executor)


def main():
    _log_sys_info()
    client = skein.ApplicationClient.from_current()
    experiment = _get_experiment(client)
    experiment()
    

if __name__ == "__main__":
    main()
    