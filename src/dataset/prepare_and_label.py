from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset.data_loading import DataConfig, get_dataloaders
from src.semantic.multimodal_recognition import RecognitionConfig, predict_labels


@dataclass(frozen=True)
class PrepareConfig:
    """
    High-level config for Part 1:
    where to write results and which splits to process.
    """
    results_dir: str = "./results"
    split_order: tuple[str, ...] = ("train", "val", "test")


def make_image_id(split: str, idx_in_split: int) -> str:
    """Stable, reproducible ID per sample (split + index)."""
    return f"pneumonia_{split}_{idx_in_split:06d}"


def default_prompts() -> List[str]:
    """
    Candidate textual descriptions (prompts) used by the CLIP-like model.
    """
    return [
        "a normal chest X-ray",
        "a chest X-ray showing pneumonia",
    ]

label_map = {
    "a normal chest X-ray": "normal",
    "a chest X-ray showing pneumonia": "pneumonia",
}

def main() -> None:
    # Part 1 orchestrates the pipeline: load data, call Part 2 (CLIP-like inference),
    # and store a single CSV artifact with the required schema.
    data_cfg = DataConfig()
    prep_cfg = PrepareConfig()
    rec_cfg = RecognitionConfig()

    # Ensure output directory exists.
    results_dir = Path(prep_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = default_prompts()
    dls = get_dataloaders(data_cfg)

    rows: List[Dict[str, object]] = []

    for split in ["test"]:
        _, loader = dls[split]
        idx_in_split = 0  # IDs are unique within each split

        # Progress bar is useful once CLIP inference is enabled (can be slow on CPU).
        pbar = tqdm(loader, desc=f"Labelling {split} split", unit="batch", leave=True)

        for images, labels in pbar:
            # MedMNIST labels arrive as shape [B, 1]. We flatten to [B] and convert to ints.
            ground_truth = labels.view(-1).tolist()
            ground_truth = [int(x) for x in ground_truth]

            # Part 2: given a batch of images + prompts, return predicted labels and confidences.
            auto_labels, conf_scores = predict_labels(images, prompts, rec_cfg)

            # Store one row per image 
            for i in range(len(images)):
                rows.append(
                    {
                        "image_id": make_image_id(split, idx_in_split),
                        "auto_label": label_map[auto_labels[i]],
                        "ground_truth": ground_truth[i],
                        "confidence_score": conf_scores[i],
                        "split": split,  # keeps split explicit for later filtering/metrics
                    }
                )
                idx_in_split += 1

            # Light feedback while running.
            pbar.set_postfix({"rows": len(rows)})

    # Save CSV with the exact required columns/order (plus 'split' for traceability).
    df = pd.DataFrame(rows, columns=["image_id", "auto_label", "ground_truth", "confidence_score", "split"])
    out_path = results_dir / "labeled_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()