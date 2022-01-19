import logging
import os
import sys

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import skein

from tf_yarn._task_commons import (
    setup_logging, get_task_description, _get_experiment, _get_cluster_tasks
)
from tf_yarn import _internal, event
from tf_yarn.pytorch.experiment import PytorchExperiment, DataLoaderArgs
setup_logging()


_logger = logging.getLogger(__name__)

MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = "MASTER_PORT"
PYTORCH_DPP_RANK = "PYTORCH_DPP_RANK"


def _log_sys_info() -> None:
    _logger.info(f"Python {sys.version}")
    _logger.info(f"Skein {skein.__version__}")
    _logger.info(f"Pytorch {torch.__version__}")


def _create_dataloader(
    dataset: torch.utils.data.Dataset, dataloader_args: DataLoaderArgs
) -> torch.utils.data.DataLoader:
    sampler: DistributedSampler = DistributedSampler(
        dataset, shuffle=dataloader_args.shuffle
    )
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=dataloader_args.batch_size,
        num_workers=dataloader_args.num_workers, pin_memory=dataloader_args.pin_memory,
        drop_last=dataloader_args.drop_last, timeout=dataloader_args.timeout,
        prefetch_factor=dataloader_args.prefetch_factor, shuffle=False
    )


def _train(device: int, rank: int, world_size: int) -> None:
    _logger.info(f"[{os.getpid()}] device: {device}; rank: {rank}")
    os.environ[PYTORCH_DPP_RANK] = str(rank)

    client = skein.ApplicationClient.from_current()
    _setup_master(client, rank)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    experiment = _get_experiment(client)
    assert isinstance(experiment, PytorchExperiment)
    model = experiment.model.to(device)
    ddp_model = DDP(model, device_ids=[device])

    trainloader = _create_dataloader(
        experiment.train_dataset, experiment.dataloader_args
    )

    experiment.train_fn(ddp_model, trainloader, f"cuda:{device}")

    dist.destroy_process_group()
    _logger.info("Done training")


def _setup_master(client: skein.ApplicationClient, rank: int) -> None:
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


def _get_device(worker_id: int) -> int:
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_id = worker_id % n_gpus
        return gpu_id
    else:
        raise RuntimeError("Multi-CPU training is not supported yet")


def main() -> None:
    _log_sys_info()
    task_type, task_id = get_task_description()

    client = skein.ApplicationClient.from_current()
    experiment = _get_experiment(client)
    cluster_tasks = _get_cluster_tasks(client)
    n_workers_per_executor = experiment.n_workers_per_executor

    world_size = len([t for t in cluster_tasks if "worker" in t]) * n_workers_per_executor
    _logger.info(
        f"Task type: {task_type}; Task id: {task_id};"
        f"World_size: {world_size}: Cluster tasks: {cluster_tasks}"
    )

    if n_workers_per_executor > 1:
        workers = list()
        mp.set_start_method("spawn")
        for n in range(n_workers_per_executor):
            worker = mp.Process(
                target=_train,
                args=(
                    _get_device(n), (task_id * n_workers_per_executor) + n, world_size
                )
            )
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
    else:
        _train(0, task_id, world_size)


if __name__ == "__main__":
    main()
