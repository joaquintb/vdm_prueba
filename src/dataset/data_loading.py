from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import torch
from torch.utils.data import DataLoader

from medmnist import PneumoniaMNIST


@dataclass(frozen=True)
class DataConfig:
    """
    Configuration for data loading.
    Images are returned in raw form (no model-specific preprocessing).
    """
    data_root: str = "./data"
    batch_size: int = 8   # temp: default for CPU
    num_workers: int = 0  # safer default on Windows


def load_split(split: str, cfg: DataConfig) -> PneumoniaMNIST:
    """
    Loads one dataset split (train / val / test) from PneumoniaMNIST.
    Images are returned as raw objects (PIL or numpy, depending on MedMNIST).
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    return PneumoniaMNIST(
        split=split,
        root=cfg.data_root,
        download=True,
        transform=None,  # raw images; model-specific preprocessing is done later
    )


def collate_raw_images(batch: List[Tuple[Any, torch.Tensor]]):
    """
    Custom collate function:
    - keeps images as a list (PIL / numpy)
    - stacks labels into a tensor

    This allows applying the model-specific preprocess inside multimodal_recognition.
    """
    images, labels = zip(*batch)
    labels = [torch.as_tensor(y) for y in labels]  # numpy -> tensor
    labels = torch.stack(labels)
    return list(images), labels


def make_dataloader(dataset: PneumoniaMNIST, cfg: DataConfig, shuffle: bool) -> DataLoader:
    """
    Wraps the dataset into a DataLoader with a custom collate_fn so that
    raw images can be batched safely.
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
    Returns (dataset, dataloader) for each split.
    """
    out: Dict[str, Tuple[PneumoniaMNIST, DataLoader]] = {}
    for split in ["train", "val", "test"]:
        ds = load_split(split, cfg)
        dl = make_dataloader(ds, cfg, shuffle=(split == "train"))
        out[split] = (ds, dl)
    return out