import os
import re
import logging
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from cluster_pack import filesystem


_logger = logging.getLogger(__name__)


def find_latest_ckpt(model_dir: str) -> Optional[str]:
    latest_ckpt = None
    latest_epoch = -1
    pattern = r".*model_(\d+).pt"
    resolved_fs, _ = filesystem.resolve_filesystem_and_path(model_dir)
    if resolved_fs.exists(model_dir):
        for p in resolved_fs.ls(model_dir):
            groups = re.match(pattern, p)
            if groups:
                epoch = int(groups.group(1))
                if epoch > latest_epoch:
                    latest_ckpt = groups.group(0)
                    latest_epoch = epoch
    return latest_ckpt


def load_latest_ckpt(
    model_dir: str, model: Union[DDP, torch.nn.Module],
    optimizer: torch.optim.Optimizer, device: Union[int, str]
) -> Optional[Dict[Any, Any]]:
    latest_ckpt = find_latest_ckpt(model_dir)
    if not latest_ckpt:
        _logger.info("No checkpoint to load")
        return None
    return load_ckpt(latest_ckpt, model, optimizer, device)


def load_ckpt(
    model_ckpt_path: str, model: Union[DDP, torch.nn.Module],
    optimizer: torch.optim.Optimizer, device: Union[int, str]
) -> Dict[Any, Any]:
    resolved_fs, _ = filesystem.resolve_filesystem_and_path(model_ckpt_path)
    _logger.info(f"Loading model checkpoint {model_ckpt_path}")
    with resolved_fs.open(model_ckpt_path, "rb") as fd:
        checkpoint = torch.load(fd, map_location=torch.device(device))
    _unwrap_model(model).load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def save_ckpt(
    model_dir: str, model: Union[DDP, torch.nn.Module], optimizer: torch.optim.Optimizer,
    epoch: int, **kwargs: Dict[Any, Any]
) -> str:
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
    with TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, f"model_{epoch}.pt")
        torch.save(state, tmp_file)
        resolved_fs.put(tmp_file, model_ckpt_path)
    return model_ckpt_path


def _unwrap_model(model: Union[DDP, torch.nn.Module]) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model
