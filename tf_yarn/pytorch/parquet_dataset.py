import logging
from typing import Optional, List
import multiprocessing as mp

from torch.utils.data import IterableDataset
import torch.distributed as dist
from cluster_pack.filesystem import resolve_filesystem_and_path, EnhancedFileSystem
import pyarrow
import pyarrow.parquet as pq


logger = logging.getLogger()


class ParquetDataset(IterableDataset):
    def __init__(
        self, dataset_path: str, batch_size: int,
        num_samples: Optional[int] = None,
        columns: List[str] = None
    ) -> None:
        self.fs, _ = resolve_filesystem_and_path(dataset_path)
        self.columns = columns
        self.num_samples = num_samples if num_samples \
            else _read_num_samples(dataset_path, self.fs)
        self.dataset_file_paths = [
            f for f in self.fs.base_fs.ls(dataset_path) if f.endswith(".parquet")
        ]
        self.batch_size = batch_size
        self.worker_id = dist.get_rank() if dist.is_initialized() else 0
        self.num_workers = dist.get_world_size() if dist.is_initialized() else 1
        logger.info(f"worker_id: {self.worker_id}; num_workers: {self.num_workers}")

    def __iter__(self):
        for dataset_file_path in self.dataset_file_paths:
            with self.fs.base_fs.open(dataset_file_path) as f:
                parquet_file = pq.ParquetFile(f)
                # Drop last batch because all-reduce ops require batches to have the same
                # sizes
                batches = list(
                    parquet_file.iter_batches(
                        batch_size=self.batch_size, columns=self.columns
                    )
                )[:-1]
                n_batches = len(batches)
                n_batches_per_worker = n_batches // self.num_workers
                assert n_batches_per_worker > 0
                start = self.worker_id * n_batches_per_worker
                end = start + n_batches_per_worker
                logger.info(f"worker_id: {self.worker_id}; n_batches: {n_batches}; "
                            f"start: {start}; end: {end}")
                for b in batches[start:end]:
                    yield b

    def __len__(self) -> int:
        return self.num_samples // self.batch_size // self.num_workers


def _read_num_samples(dataset_path: str, fs: EnhancedFileSystem) -> int:
    mp.set_start_method('spawn', force=True)
    dataset_file_paths = [
        f for f in fs.base_fs.ls(dataset_path) if f.endswith(".parquet")
    ]
    with mp.Pool(5) as p:
        n_rows = p.map(_get_num_rows, dataset_file_paths)
        return sum(n_rows)


def _get_num_rows(file: str) -> int:
    fs, _ = resolve_filesystem_and_path(file)
    with fs.base_fs.open(file) as f:
        parquet_file = pq.ParquetFile(f)
        return parquet_file.metadata.num_rows
