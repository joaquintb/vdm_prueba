from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF

from src.dataset.data_loading import DataConfig, get_dataloaders
from src.semantic.multimodal_recognition import RecognitionConfig, predict_labels


@dataclass(frozen=True)
class PrepareConfig:
    results_dir: str = "./results"
    split_order: tuple[str, ...] = ("train", "val", "test")

    # LoRA fine-tuning subset export
    export_lora_subset: bool = True
    lora_subset_dir: str = "./data/subset"
    lora_subset_size: int = 250  # total; will be balanced (size//2 per class)


def make_image_id(split: str, idx_in_split: int) -> str:
    return f"pneumonia_{split}_{idx_in_split:06d}"


def default_prompts() -> List[str]:
    return [
        "a chest X-ray with no evidence of pneumonia",
        "a chest X-ray with pulmonary opacity consistent with pneumonia",
    ]


def prompt_to_class(label: str) -> int:
    s = (label or "").lower()
    return 1 if "pneumonia" in s else 0


def prompt_to_short_label(prompt: str) -> str:
    s = prompt.lower()
    return "pneumonia" if "pneumonia" in s else "normal"


def gt_to_label(y: int) -> str:
    return "pneumonia" if int(y) == 1 else "normal"


def ensure_pil(img) -> Image.Image:
    """
    The MedMNIST dataloader may return PIL, numpy arrays or torch tensors.
    This makes saving consistent.
    """
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, torch.Tensor):
        return TF.to_pil_image(img)
    raise TypeError(f"Unsupported image type: {type(img)}")


def compute_binary_metrics(df: pd.DataFrame) -> Dict[str, object]:
    y_true = df["ground_truth"].to_numpy(dtype=int)
    y_pred = df["auto_label"].apply(prompt_to_class).to_numpy(dtype=int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return {
        "n": int(len(df)),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def save_metrics(df: pd.DataFrame, results_dir: Path) -> Path:
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {}
    for split in sorted(df["split"].unique()):
        summary[split] = compute_binary_metrics(df[df["split"] == split])

    summary["overall"] = compute_binary_metrics(df)

    metrics_path = metrics_dir / "auto_label_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return metrics_path


def init_lora_subset_export(cfg: PrepareConfig) -> Optional[Tuple[Path, Path]]:
    """
    Prepares output folders for a balanced LoRA fine-tuning subset:
      data/subset/images/
      data/subset/metadata.jsonl
    """
    if not cfg.export_lora_subset:
        return None

    subset_dir = Path(cfg.lora_subset_dir)
    images_dir = subset_dir / "images"
    subset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    meta_path = subset_dir / "metadata.jsonl"
    # Overwrite on each run to keep it deterministic / clean.
    if meta_path.exists():
        meta_path.unlink()

    return images_dir, meta_path


def maybe_export_to_lora_subset(
    img,
    y: int,
    prompts: List[str],
    images_dir: Path,
    meta_path: Path,
    counters: Dict[int, int],
    target_per_class: int,
    total_target: int,
) -> bool:
    """
    Saves one image + its prompt to the LoRA subset if we still need samples from its class.
    Returns True if it was saved.
    """
    if counters[0] + counters[1] >= total_target:
        return False
    if counters[int(y)] >= target_per_class:
        return False

    idx = counters[0] + counters[1]
    pil = ensure_pil(img).convert("RGB")

    fname = f"img_{idx:05d}.png"
    pil.save(images_dir / fname)

    record = {"file_name": f"images/{fname}", "text": prompts[int(y)]}
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    counters[int(y)] += 1
    return True


def main() -> None:
    data_cfg = DataConfig()
    prep_cfg = PrepareConfig()
    rec_cfg = RecognitionConfig()

    results_dir = Path(prep_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = default_prompts()
    dls = get_dataloaders(data_cfg)

    # Prepare optional LoRA subset export (balanced, from TRAIN split only)
    subset_paths = init_lora_subset_export(prep_cfg)
    subset_counters = {0: 0, 1: 0}
    subset_target_per_class = prep_cfg.lora_subset_size // 2

    rows: List[Dict[str, object]] = []

    for split in prep_cfg.split_order:
        _, loader = dls[split]
        idx_in_split = 0

        pbar = tqdm(loader, desc=f"Labelling {split} split", unit="batch", leave=True)

        for images, labels in pbar:
            # MedMNIST labels: shape [B, 1] -> [B]
            ground_truth = labels.view(-1).tolist()
            ground_truth = [int(x) for x in ground_truth]

            # Part 2: CLIP-like recognition (BiomedCLIP) to get auto labels + confidence
            auto_labels, conf_scores = predict_labels(images, prompts, rec_cfg)

            for i in range(len(images)):
                score = float(conf_scores[i])
                if not np.isfinite(score):
                    score = np.nan

                y = ground_truth[i]

                # Optional: export a balanced subset for LoRA fine-tuning (TRAIN only)
                subset_full = (subset_counters[0] + subset_counters[1]) >= prep_cfg.lora_subset_size
                if split == "train" and subset_paths is not None and not subset_full:
                    images_dir, meta_path = subset_paths
                    maybe_export_to_lora_subset(
                        img=images[i],
                        y=y,
                        prompts=prompts,
                        images_dir=images_dir,
                        meta_path=meta_path,
                        counters=subset_counters,
                        target_per_class=subset_target_per_class,
                        total_target=prep_cfg.lora_subset_size,
                    )

                rows.append(
                    {
                        "image_id": make_image_id(split, idx_in_split),
                        "auto_label": prompt_to_short_label(auto_labels[i]),
                        "ground_truth": y,
                        "ground_truth_label": gt_to_label(y),
                        "confidence_score": score,
                        "split": split,
                    }
                )
                idx_in_split += 1

            pbar.set_postfix(
                {
                    "rows": len(rows),
                    "subset": f"{subset_counters[0] + subset_counters[1]}/{prep_cfg.lora_subset_size}"
                    if split == "train" and subset_paths is not None
                    else "-",
                }
            )

    df = pd.DataFrame(
        rows,
        columns=["image_id", "auto_label", "ground_truth", "ground_truth_label", "confidence_score", "split"],
    )
    csv_path = results_dir / "labeled_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    metrics_path = save_metrics(df, results_dir)
    print(f"Saved: {metrics_path}")

    if subset_paths is not None:
        n_saved = subset_counters[0] + subset_counters[1]
        print(
            f"Saved LoRA subset to: {prep_cfg.lora_subset_dir} "
            f"(normal={subset_counters[0]}, pneumonia={subset_counters[1]}, total={n_saved})"
        )

if __name__ == "__main__":
    main()