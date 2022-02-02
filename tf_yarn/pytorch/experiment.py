from typing import NamedTuple, Callable, Optional

import torch


class DataLoaderArgs(NamedTuple):
    batch_size: Optional[int] = 1
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: int = 2
    shuffle: bool = False


class PytorchExperiment(NamedTuple):
    # Model to train
    model: torch.nn.Module

    # Main function run to train the model
    # inputs: model to train, dataloader, device (cpu:0, cpu:1, cuda:0, cuda:1 ...)
    # and rank (worker id)
    # outputs: None
    train_fn: Callable[
        [torch.nn.Module, torch.utils.data.dataloader.DataLoader, str, int],
        None
    ]

    # Training set
    train_dataset: torch.utils.data.Dataset

    # dataloader parameters (batch size, number of workers, collate function ...) used to
    # load and iterate over the training set
    dataloader_args: DataLoaderArgs

    # Number of workers per executors
    n_workers_per_executor: int = 1

    # HDFS directory where tensorboard results will be written at the end of the training
    tensorboard_hdfs_dir: Optional[str] = None
