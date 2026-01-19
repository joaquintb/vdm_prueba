"""
pipeline.py

Part 4: Integrated end-to-end pipeline.

This script orchestrates the full workflow:
1. Dataset preparation and automatic labeling.
2. Lightweight LoRA fine-tuning on a small labeled subset.
3. Image generation using Stable Diffusion + LCM (+ optional fine-tuned LoRA).

The pipeline is disk-based: each stage writes artifacts to disk and the next
stage reads from them. This makes the process robust, debuggable, and easy
to resume or partially skip.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# Parts 1&2: automatic labeling + subset export
from src.dataset.prepare_and_label import main as prepare_and_label_main

# Part 3b: LoRA fine-tuning
from src.generation.lora_fine_tuning import LoRAFineTuneConfig, run_lora_finetune

# Part 3a: LCM-based image generation
from src.generation.lcm_generate import (
    GenConfig,
    build_diff_pipeline,
    generate_images,
    default_prompts,
)


@dataclass(frozen=True)
class PipelineConfig:
    """
    Centralized paths for pipeline artifacts.
    """
    results_dir: str = "./results"
    labeled_csv: str = "./results/labeled_dataset.csv"
    metrics_json: str = "./results/metrics/auto_label_metrics.json"

    lora_subset_dir: str = "./data/subset"
    lora_out_dir: str = "./results/lora/pneumonia_lora"

    generated_dir: str = "./results/generated_images"


def _find_lora_weights(out_dir: Path) -> Optional[Path]:
    """
    Locate LoRA weights produced by training.

    Supports both standard Diffusers filenames and generic .safetensors outputs.
    """
    candidates = [
        out_dir / "pytorch_lora_weights.safetensors",
        out_dir / "pytorch_lora_weights.bin",
    ]
    for p in candidates:
        if p.exists():
            return p

    safes = sorted(out_dir.glob("*.safetensors"))
    return safes[0] if safes else None


def main() -> None:
    """
    Entry point for running the full pipeline.

    Each stage can be skipped independently, and `--force` allows recomputation
    even if artifacts already exist on disk.
    """
    parser = argparse.ArgumentParser(description="Run full VDM pipeline (disk-based).")
    parser.add_argument("--skip_labeling", action="store_true")
    parser.add_argument("--skip_lora", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig()

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    labeled_csv = Path(cfg.labeled_csv)
    metrics_json = Path(cfg.metrics_json)
    subset_dir = Path(cfg.lora_subset_dir)
    lora_out_dir = Path(cfg.lora_out_dir)

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    # -----------------------
    # Step 1: Dataset labeling
    # -----------------------
    if not args.skip_labeling:
        if args.force or not labeled_csv.exists():
            print("[1/3] Running prepare_and_label...")
            prepare_and_label_main()
        else:
            print("[1/3] Skipping prepare_and_label (artifacts already exist).")
    else:
        print("[1/3] Skipped (flag).")

    if not labeled_csv.exists():
        raise FileNotFoundError(f"Missing labeled CSV: {labeled_csv}")

    if metrics_json.exists():
        print(f"Found metrics: {metrics_json}")
    print()

    # -----------------------
    # Step 2: LoRA fine-tuning
    # -----------------------
    finetuned_weights = _find_lora_weights(lora_out_dir)

    if not args.skip_lora:
        if not subset_dir.exists():
            raise FileNotFoundError(
                f"LoRA subset not found at {subset_dir}. "
                "prepare_and_label must export it first."
            )

        if args.force or finetuned_weights is None:
            print("[2/3] Running LoRA fine-tuning...")
            run_lora_finetune(LoRAFineTuneConfig())
            finetuned_weights = _find_lora_weights(lora_out_dir)
        else:
            print("[2/3] Skipping LoRA fine-tuning (weights already exist).")
    else:
        print("[2/3] Skipped (flag).")

    if not args.skip_lora and finetuned_weights is None:
        raise FileNotFoundError(f"LoRA weights not found in {lora_out_dir}")

    if finetuned_weights:
        print(f"Using LoRA weights: {finetuned_weights}\n")

    # -----------------------
    # Step 3: Image generation
    # -----------------------
    if not args.skip_generation:
        print("[3/3] Generating images with LCM...")

        gen_cfg = GenConfig(
            out_dir=cfg.generated_dir,
            finetuned_lora_path=str(finetuned_weights) if finetuned_weights else None,
        )

        pipe = build_diff_pipeline(gen_cfg)

        # Simple, deterministic prompt set: balanced normal / pneumonia
        prompts = default_prompts(gen_cfg.num_images)

        generate_images(pipe, prompts, gen_cfg)
        print(f"Images saved to: {gen_cfg.out_dir}")
    else:
        print("[3/3] Skipped (flag).")

    # -----------------------
    # Pipeline summary
    # -----------------------
    summary = {
        "labeled_csv": str(labeled_csv),
        "metrics_json": str(metrics_json) if metrics_json.exists() else None,
        "lora_subset_dir": str(subset_dir),
        "lora_out_dir": str(lora_out_dir),
        "lora_weights": str(finetuned_weights) if finetuned_weights else None,
        "generated_dir": cfg.generated_dir,
        "num_generated": gen_cfg.num_images if not args.skip_generation else 0,
    }

    summary_path = results_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPipeline summary written to: {summary_path}")


if __name__ == "__main__":
    main()