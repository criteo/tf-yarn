import os
from typing import Optional, Union, Dict, Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from cluster_pack import filesystem

from tf_yarn.pytorch.tasks.worker import PYTORCH_DPP_RANK


def load_ckpt(
    model_ckpt_path: str, model: Union[DDP, torch.nn.Module],
    optimizer: torch.optim.Optimizer, device: Union[int, str]
) -> Dict[Any, Any]:
    resolved_fs, _ = filesystem.resolve_filesystem_and_path(model_ckpt_path)
    with resolved_fs.open(model_ckpt_path, "rb") as fd:
        checkpoint = torch.load(fd, map_location=torch.device(device))
    _unwrap_model(model).load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def save_ckpt(
    model_dir: str, model: Union[DDP, torch.nn.Module], optimizer: torch.optim.Optimizer,
    epoch: int, **kwargs: Dict[Any, Any]
) -> Optional[str]:
    if int(os.environ[PYTORCH_DPP_RANK]) != 0:
        return None

    state = {
        'model': _unwrap_model(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        **kwargs
    }
    resolved_fs, _ = filesystem.resolve_filesystem_and_path(model_dir)
    if not resolved_fs.exists(model_dir):
        resolved_fs.mkdir(model_dir)
    model_ckpt_path = os.path.join(model_dir, f"model_{epoch}.pt")
    with resolved_fs.open(model_ckpt_path, "wb") as fd:
        torch.save(state, fd)
    return model_ckpt_path


def _unwrap_model(model: Union[DDP, torch.nn.Module]) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model
