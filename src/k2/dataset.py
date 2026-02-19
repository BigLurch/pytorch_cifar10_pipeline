from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


@dataclass
class DataConfig:
    data_dir: Path = Path("dl/cifar10")
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return train_tf, test_tf


def make_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = build_transforms()

    train_ds = CIFAR10(
        root=str(cfg.data_dir), train=True, download=False, transform=train_tf
    )
    test_ds = CIFAR10(
        root=str(cfg.data_dir), train=False, download=False, transform=test_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, test_loader
