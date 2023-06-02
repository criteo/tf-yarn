from typing import Callable, NamedTuple, Tuple, Any
import sys
import logging
import skein
import cloudpickle

from tf_yarn._task_commons import (
    _get_cluster_tasks,
    get_pickled_experiment,
    _get_nb_workers,
    compute_rank,
    choose_master,
    get_task_key,
    setup_logging,
    catchtime
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


def get_task(local_rank: int):
    task_key = get_task_key()

    client = skein.ApplicationClient.from_current()

    cluster_tasks = _get_cluster_tasks(client)
    n_workers = _get_nb_workers(task_key.id, cluster_tasks)
    rank = compute_rank(task_key.id, local_rank, n_workers)

    addr, port = choose_master(client, rank)
    _logger.info(f'master: {addr}:{port}')

    world_size = len([t for t in cluster_tasks if "worker" in t]) * n_workers
    _logger.info(
        f"Task type: {task_key.type}; Task id: {task_key.id}; Rank: {rank}; "
        f"Local rank: {local_rank}; World_size: {world_size}: Cluster tasks: {cluster_tasks}"
    )
    return TaskParameters(
        task_key.type, rank, world_size, addr, port, n_workers)


def unpickle_and_exec(experiment: bytes, local_rank: int):
    with catchtime('unpickle target function'):
        fcn = cloudpickle.loads(experiment)
    fcn(local_rank)


def parallel_run(n_workers: int, task_id: int,
                 pickled_fcn):
    from torch import multiprocessing as mp
    workers = list()
    mp.set_start_method("spawn", force=True)
    for local_rank in range(n_workers):
        worker = mp.Process(
            target=unpickle_and_exec,
            args=(pickled_fcn, local_rank)
        )
        print(f"starting process {local_rank} of task {task_id}")
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()


def main():
    _log_sys_info()
    client = skein.ApplicationClient.from_current()
    with catchtime("fetch Experiment function"):
        experiment = get_pickled_experiment(client)

    task_key = get_task_key()
    cluster_tasks = _get_cluster_tasks(client)
    n_workers_per_task = _get_nb_workers(task_key.id, cluster_tasks)
    if n_workers_per_task > 1:
        parallel_run(n_workers_per_task, task_key.id, experiment)
    else:
        unpickle_and_exec(experiment, 0)


if __name__ == "__main__":
    main()
