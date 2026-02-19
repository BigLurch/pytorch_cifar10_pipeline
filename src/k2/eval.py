from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import evaluate, resolve_device


def load_checkpoint(
    model: nn.Module, ckpt_path: Path, device: torch.device
) -> nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def eval_checkpoint(
    model: nn.Module, loader: DataLoader, ckpt_path: Path, device: str = "auto"
) -> Dict[str, float]:
    dev = resolve_device(device)
    model = model.to(dev)
    model = load_checkpoint(model, ckpt_path, dev)
    return evaluate(model, loader, dev)
