from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist import PneumoniaMNIST


@dataclass(frozen=True)
class DataConfig:
    data_root: str = "./data"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2


def build_base_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),  # PIL → RGB (for CLIP)
            transforms.ToTensor(), # PIL → Tensor [0,1]
        ]
    )


def load_split(split: str, cfg: DataConfig) -> PneumoniaMNIST:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    return PneumoniaMNIST(
        split=split,
        root=cfg.data_root,
        download=True,
        transform=build_base_transform(cfg.image_size),
    )


def make_dataloader(dataset: PneumoniaMNIST, cfg: DataConfig, shuffle: bool) -> DataLoader:
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
    Returns dict with keys: 'train', 'val', 'test'
    Each value is (dataset, dataloader).
    """
    out: Dict[str, Tuple[PneumoniaMNIST, DataLoader]] = {}
    for split in ["train", "val", "test"]:
        ds = load_split(split, cfg)
        dl = make_dataloader(ds, cfg, shuffle=(split == "train"))
        out[split] = (ds, dl)
    return out


if __name__ == "__main__":

    cfg = DataConfig(data_root="./data", image_size=224, batch_size=8, num_workers=0)
    dls = get_dataloaders(cfg)

    images, labels = next(iter(dls["train"][1]))
    print("images:", images.shape, images.dtype, images.min().item(), images.max().item())
    print("labels:", labels.shape, labels[:8].view(-1).tolist())
