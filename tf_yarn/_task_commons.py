import json
import logging
import os
from typing import List, NamedTuple, Tuple
from time import perf_counter

import cloudpickle
import skein

from tf_yarn import _internal, event, constants
from tf_yarn._internal import iter_tasks
from tf_yarn.topologies import ContainerTask, ContainerKey


MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = "MASTER_PORT"


def setup_logging():
    # tensorflow imports in tf_yarn.__init__ already have set up some loggers
    # erase them with a clean config
    log_conf_file = os.path.join(os.path.dirname(__file__), "default.log.conf")
    logging.config.fileConfig(log_conf_file)


def _setup_container_logs(client):
    task = get_task_key()
    event.broadcast_container_start_time(client, task)
    container = next(c for c in client.get_containers()
                     if c.yarn_container_id == os.environ["CONTAINER_ID"])
    logs = container.yarn_container_logs
    if logs is not None and not logs.startswith("http://"):
        logs = "http://" + logs
    event.logs_event(client, task, logs)


def _get_cluster_tasks(
    client: skein.ApplicationClient
) -> List[ContainerTask]:
    return list(iter_tasks(json.loads(client.kv.wait(constants.KV_CLUSTER_INSTANCES).decode())))


def _compute_world_size(cluster_tasks: List[ContainerTask]):
    return sum([task.nb_proc for task in cluster_tasks])


def _get_nb_workers(task_id: int, cluster_tasks: List[ContainerTask]):
    return [task.nb_proc for task in cluster_tasks if task.id == task_id][0]


def get_pickled_experiment(client: skein.ApplicationClient) -> bytes:
    return client.kv.wait(constants.KV_EXPERIMENT_FN)


def _get_experiment(client: skein.ApplicationClient) -> NamedTuple:
    try:
        experiment = cloudpickle.loads(get_pickled_experiment(client))()
    except Exception as e:
        task = get_task_key()
        event.start_event(client, task)
        event.stop_event(client, task, e)
        raise
    return experiment


def n_try() -> int:
    return int(os.getenv("TF_YARN_N_TRY", "0"))


def get_task_key() -> ContainerKey:
    return ContainerKey.from_kv_str(os.getenv("SKEIN_CONTAINER_ID", "").replace("_", ":", 1))


def is_worker(task_type: str = None) -> bool:
    if not task_type:
        task_type = get_task_key().type

    return task_type == 'worker'


def is_evaluator(task_type: str = None) -> bool:
    if not task_type:
        task_type = get_task_key().type

    return task_type == 'evaluator'


def is_chief(task_type: str = None) -> bool:
    if not task_type:
        task_type = get_task_key().type

    return task_type == 'chief'


def choose_master(client: skein.ApplicationClient, rank: int) -> Tuple[str, int]:
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

    return master_addr, master_port


def compute_rank(task_id: int, local_rank: int, n_workers):
    # Todo: better computation of rank as sum[0:taskid](n_process(task))
    return (task_id * n_workers) + local_rank


# Context manager to print time elapsed in a block of code
class catchtime:
    def __init__(self, description: str):
        self.description = description

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time to {self.description}: {self.time:.3f} seconds'
        print(self.readout)
