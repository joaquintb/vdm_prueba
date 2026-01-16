from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist import PneumoniaMNIST

from PIL import Image

@dataclass(frozen=True)
class DataConfig:
    """
    Configuration for data loading and basic preprocessing.
    """
    data_root: str = "./data"
    image_size: int = 224
    batch_size: int = 8 # Temp (CPU)
    num_workers: int = 2

def pil_to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")

def build_base_transform(image_size: int) -> transforms.Compose:
    """
    Basic preprocessing:
    - resize to a fixed shape (consistent input for downstream models)
    - convert to RGB (many pretrained models expect 3 channels)
    - convert to float tensor in [0, 1]
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(pil_to_rgb),  # 1 channel -> 3 channels
            transforms.ToTensor(),  # float32 in [0, 1]
        ]
    )


def load_split(split: str, cfg: DataConfig) -> PneumoniaMNIST:
    """
    Loads one dataset split (train/val/test) from PneumoniaMNIST.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    return PneumoniaMNIST(
        split=split,
        root=cfg.data_root,
        download=True,
        transform=build_base_transform(cfg.image_size),
    )


def make_dataloader(dataset: PneumoniaMNIST, cfg: DataConfig, shuffle: bool) -> DataLoader:
    """
    Wraps the dataset into a DataLoader to enable efficient batch processing.
    Shuffling enabled only for training.
    """
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def get_dataloaders(cfg: DataConfig) -> Dict[str, Tuple[PneumoniaMNIST, DataLoader]]:
    """
    Convenience helper that returns (dataset, dataloader) for each split.
    """
    out: Dict[str, Tuple[PneumoniaMNIST, DataLoader]] = {}
    for split in ["train", "val", "test"]:
        ds = load_split(split, cfg)
        dl = make_dataloader(ds, cfg, shuffle=(split == "train"))
        out[split] = (ds, dl)
    return out
