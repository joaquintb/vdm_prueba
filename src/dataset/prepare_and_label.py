from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset.data_loading import DataConfig, get_dataloaders
from src.semantic.multimodal_recognition import RecognitionConfig, predict_labels


@dataclass(frozen=True)
class PrepareConfig:
    results_dir: str = "./results"
    split_order: tuple[str, ...] = ("train", "val", "test")


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
    """
    Saves minimal metrics (per split + overall) to results/metrics/metrics.json.
    """
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


def main() -> None:
    data_cfg = DataConfig()
    prep_cfg = PrepareConfig()
    rec_cfg = RecognitionConfig()

    results_dir = Path(prep_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = default_prompts()
    dls = get_dataloaders(data_cfg)

    rows: List[Dict[str, object]] = []

    for split in prep_cfg.split_order:
        _, loader = dls[split]
        idx_in_split = 0

        pbar = tqdm(loader, desc=f"Labelling {split} split", unit="batch", leave=True)

        for images, labels in pbar:
            ground_truth = labels.view(-1).tolist()
            ground_truth = [int(x) for x in ground_truth]

            auto_labels, conf_scores = predict_labels(images, prompts, rec_cfg)

            for i in range(len(images)):
                score = float(conf_scores[i])
                if not np.isfinite(score):
                    score = np.nan

                rows.append(
                    {
                        "image_id": make_image_id(split, idx_in_split),
                        "auto_label": prompt_to_short_label(auto_labels[i]),
                        "ground_truth": ground_truth[i],
                        "ground_truth_label": gt_to_label(ground_truth[i]),
                        "confidence_score": score,
                        "split": split,
                    }
                )
                idx_in_split += 1

            pbar.set_postfix({"rows": len(rows)})

    df = pd.DataFrame(rows, columns=["image_id", "auto_label", "ground_truth", "ground_truth_label", "confidence_score", "split"])
    csv_path = results_dir / "labeled_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    metrics_path = save_metrics(df, results_dir)
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
