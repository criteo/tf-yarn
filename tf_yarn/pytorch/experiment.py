from typing import NamedTuple, Callable, Any, Dict

import torch


class PytorchExperiment(NamedTuple):
    # Model to train
    model: torch.nn.Module

    # Main function run to train the model
    # inputs: model to train, dataloader and device (cpu:0, cpu:1, cuda:0, cuda:1 ...)
    # outputs: None
    train_fn: Callable[[torch.nn.Module, torch.utils.data.dataloader.DataLoader, str], None]

    # Training set
    train_dataset: torch.utils.data.Dataset

    # dataloader parameters (batch size, number of workers, collate function ...) used to
    # load and iterate over the training set
    dataloader_kwargs: Dict[Any, Any] = {}

    # Number of workers per executors
    n_workers_per_executor: int = 1
