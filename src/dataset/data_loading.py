"""
data_loading.py

Utilities to load the PneumoniaMNIST dataset and provide PyTorch DataLoaders.
We intentionally keep images in "raw" form (PIL / numpy) and defer any model-
specific preprocessing (e.g., CLIP preprocess) to the semantic inference module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from medmnist import PneumoniaMNIST


@dataclass(frozen=True)
class DataConfig:
    """
    Data loading configuration.

    Note: images are returned in raw form (no transforms here). This keeps the
    data module model-agnostic; model-specific preprocessing happens later.
    """
    data_root: str = "./data"
    batch_size: int = 32
    num_workers: int = 2


def load_split(split: str, cfg: DataConfig) -> PneumoniaMNIST:
    """
    Load one dataset split (train / val / test) from PneumoniaMNIST.

    Returns raw images plus labels.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    return PneumoniaMNIST(
        split=split,
        root=cfg.data_root,
        download=True,
        transform=None,  # Keep raw images; preprocess later
    )


def collate_raw_images(batch: List[Tuple[Any, torch.Tensor]]) -> Tuple[List[Any], torch.Tensor]:
    """
    Custom collate_fn for raw-image datasets.

    - Keeps images as a Python list (so PIL / numpy objects are preserved).
    - Stacks labels into a single tensor of shape [B, ...].
    """
    images, labels = zip(*batch)

    # MedMNIST labels may come as numpy arrays; convert each to a tensor then stack.
    labels_t = torch.stack([torch.as_tensor(y) for y in labels])
    return list(images), labels_t


def make_dataloader(dataset: PneumoniaMNIST, cfg: DataConfig, shuffle: bool) -> DataLoader:
    """
    Wrap a dataset split into a DataLoader.

    We use a custom collate_fn because images are raw objects, not tensors.
    """
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_raw_images,
    )


def get_dataloaders(cfg: DataConfig) -> Dict[str, Tuple[PneumoniaMNIST, DataLoader]]:
    """
    Convenience helper: returns a dict with keys 'train', 'val', 'test'.
    Each entry is (dataset, dataloader).

    We keep `shuffle=False` because we are labeling deterministically and want
    stable image_id generation across runs.
    """
    out: Dict[str, Tuple[PneumoniaMNIST, DataLoader]] = {}
    for split in ["train", "val", "test"]:
        ds = load_split(split, cfg)
        dl = make_dataloader(ds, cfg, shuffle=False)
        out[split] = (ds, dl)
    return out