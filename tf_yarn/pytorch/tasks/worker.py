import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Generator, Optional, Union

import skein
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import webdataset as wds
from cluster_pack import filesystem
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
try:
    import tensorflow_io as tfio
except Exception:
    pass

from tf_yarn import _internal, event, tensorboard
from tf_yarn._task_commons import (
    _get_cluster_tasks,
    _compute_world_size,
    _get_nb_workers,
    _get_experiment,
    compute_rank,
    choose_master,
    get_task_key,
    setup_logging
)
from tf_yarn.pytorch.experiment import DataLoaderArgs, PytorchExperiment

setup_logging()


_logger = logging.getLogger(__file__)

MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = "MASTER_PORT"


def _log_sys_info() -> None:
    _logger.info(f"Python {sys.version}")
    _logger.info(f"Skein {skein.__version__}")
    _logger.info(f"Pytorch {torch.__version__}")


def _create_dataloader(
    dataset: Union[torch.utils.data.Dataset, wds.WebDataset, wds.DataPipeline],
    dataloader_args: DataLoaderArgs,
) -> Union[torch.utils.data.DataLoader, wds.WebLoader]:
    if isinstance(dataset, (wds.WebDataset, wds.DataPipeline)):
        return wds.WebLoader(
            dataset,
            batch_size=dataloader_args.batch_size,
            num_workers=dataloader_args.num_workers,
            pin_memory=dataloader_args.pin_memory,
            drop_last=dataloader_args.drop_last,
            timeout=dataloader_args.timeout,
            prefetch_factor=dataloader_args.prefetch_factor,
            persistent_workers=dataloader_args.persistent_workers,
            shuffle=dataloader_args.shuffle,
        )
    sampler: Optional[DistributedSampler] = (
        DistributedSampler(dataset, shuffle=dataloader_args.shuffle)
        if not isinstance(dataset, torch.utils.data.IterableDataset)
        else None
    )
    if not dataloader_args.drop_last:
        _logger.error(
            "/!\\ Not dropping the last batch could result in a smaller "
            "batch size which could block your distributed training when aggregating "
            "tensors with allreduce/allgather. It will cause a memory corruption on "
            "the worker processing the smaller batch size and your training will fail "
            "or simply freeze. We strongly encourage setting DataLoaderArgs.drop_last "
            "to True"
        )
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=dataloader_args.batch_size,
        num_workers=dataloader_args.num_workers,
        pin_memory=dataloader_args.pin_memory,
        drop_last=dataloader_args.drop_last,
        timeout=dataloader_args.timeout,
        prefetch_factor=dataloader_args.prefetch_factor,
        shuffle=dataloader_args.shuffle,
        collate_fn=dataloader_args.collate_fn
    )


def _train(device: int, rank: int, world_size: int, collective_ops_backend: str) -> None:
    os.environ["NCCL_DEBUG"] = "INFO"
    _logger.info(f"[{os.getpid()}] device: {device}; rank: {rank}")

    client = skein.ApplicationClient.from_current()
    _setup_master(client, rank)

    dist.init_process_group(collective_ops_backend, rank=rank, world_size=world_size)

    experiment = _get_experiment(client)
    assert isinstance(experiment, PytorchExperiment)
    model = experiment.model.to(device)
    ddp_kwargs = experiment.ddp_args._asdict() if experiment.ddp_args else {}
    ddp_model = DDP(model, device_ids=[device], **ddp_kwargs)

    trainloader = _create_dataloader(experiment.train_dataset, experiment.dataloader_args)

    with tempfile.TemporaryDirectory() as tmp:
        tb_writer = SummaryWriter(tmp)
        experiment.main_fn(ddp_model, trainloader, f"cuda:{device}", rank, tb_writer)
        tb_writer.flush()
        tb_writer.close()

        if experiment.tensorboard_hdfs_dir:
            worker_tb_dir = os.path.join(experiment.tensorboard_hdfs_dir, f"worker{rank}")
            _upload_tensorboard_on_hdfs(tmp, worker_tb_dir)

    dist.destroy_process_group()
    _logger.info("Done training")


@contextmanager
def _tensorboard(
    tensorboard_dir: str, client: skein.ApplicationClient
) -> Generator[None, None, None]:
    task_key = get_task_key()

    thread = _internal.MonitoredThread(
        name=f"{task_key.to_kv_str()}",
        target=tensorboard.start_tf_board,
        args=(client, tensorboard_dir),
        daemon=True,
    )
    thread.start()

    yield

    timeout = tensorboard.get_termination_timeout()
    thread.join(timeout)


def _upload_tensorboard_on_hdfs(local_dir: str, hdfs_dir: str) -> None:
    resolved_fs, _ = filesystem.resolve_filesystem_and_path(hdfs_dir)
    if not resolved_fs.exists(hdfs_dir):
        resolved_fs.mkdir(hdfs_dir)
    for f in os.listdir(local_dir):
        hdfs_file_path = os.path.join(hdfs_dir, f)
        local_file_path = os.path.join(local_dir, f)
        resolved_fs.put(local_file_path, hdfs_file_path)


def _setup_master(client: skein.ApplicationClient, rank: int) -> None:
    addr, port = choose_master(client, rank)
    os.environ[MASTER_ADDR] = addr
    os.environ[MASTER_PORT] = str(port)
    _logger.info(f'master: {addr}:{port}')


def _get_device(worker_id: int) -> int:
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_id = worker_id % n_gpus
        return gpu_id
    else:
        raise RuntimeError("Multi-CPU training is not supported yet")


def _get_collective_ops_backend(n_workers_per_executor: int) -> str:
    # If a GPU is used by multiple workers, using NCCL can result in a deadlock
    # So in this case, we use gloo for collective ops
    return "nccl" if n_workers_per_executor <= torch.cuda.device_count() else "gloo"


def main() -> None:
    _log_sys_info()
    task_key = get_task_key()

    client = skein.ApplicationClient.from_current()
    experiment = _get_experiment(client)
    assert isinstance(experiment, PytorchExperiment)
    cluster_tasks = _get_cluster_tasks(client)
    n_workers_per_executor = _get_nb_workers(task_key.id, cluster_tasks)
    world_size = _compute_world_size(cluster_tasks)

    _logger.info(
        f"Task type: {task_key.type}; Task id: {task_key.id}; "
        f"World_size: {world_size}: Cluster tasks: {cluster_tasks}"
    )

    if n_workers_per_executor > 1:
        workers = list()
        mp.set_start_method("spawn", force=True)
        for n in range(n_workers_per_executor):
            rank = compute_rank(task_key.id, n, n_workers_per_executor)
            worker = mp.Process(
                target=_train,
                args=(
                    _get_device(n),
                    rank,
                    world_size,
                    _get_collective_ops_backend(n_workers_per_executor),
                ),
            )
            print(f"starting process {n} (rank {rank}) of task {task_key.to_kv_str()}")
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
    else:
        _train(0, task_key.id, world_size, "nccl")


if __name__ == "__main__":
    main()
