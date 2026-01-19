"""
lora_fine_tuning.py

Part 3b: Lightweight LoRA fine-tuning wrapper for Stable Diffusion (text-to-image).

This script does NOT reimplement training logic. Instead, it wraps and calls
the official Diffusers LoRA training script via `accelerate launch`.

Assumes a small, already-prepared dataset on disk (images + metadata),
typically exported during the dataset preparation stage of the pipeline.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoRAFineTuneConfig:
    """
    Configuration for lightweight LoRA fine-tuning using Diffusers.

    This acts as a thin orchestration layer around the official
    `train_text_to_image_lora.py` script.
    """

    # Path to the vendored Diffusers training script (kept version-pinned)
    script_path: str = "third_party/diffusers/train_text_to_image_lora.py"

    # Dataset prepared beforehand:
    #   data/subset/images/*.png
    #   data/subset/metadata.jsonl
    train_data_dir: str = "data/subset"
    resolution: int = 256

    # Base diffusion model to adapt
    pretrained_model: str = "runwayml/stable-diffusion-v1-5"

    # Training setup: intentionally small and fast (suitable for Colab T4)
    batch_size: int = 1
    grad_accum_steps: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 200
    rank: int = 8
    seed: int = 42
    mixed_precision: str = "fp16"  # good default on T4 GPUs

    # Output directory where the LoRA adapter will be saved
    output_dir: str = "results/lora/pneumonia_lora"

    # Optional: save checkpoints every N steps (None = disabled)
    checkpointing_steps: int | None = None


def _validate_inputs(cfg: LoRAFineTuneConfig) -> None:
    """
    Sanity-checks all required inputs before launching training.

    Fails early if:
    - the Diffusers script is missing
    - the dataset structure is incomplete
    """
    script = Path(cfg.script_path)
    if not script.exists():
        raise FileNotFoundError(
            f"Could not find vendored Diffusers script at: {script}\n"
            "Make sure train_text_to_image_lora.py was copied into third_party/diffusers/."
        )

    train_dir = Path(cfg.train_data_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Train data dir not found: {train_dir}")

    images_dir = train_dir / "images"
    meta_path = train_dir / "metadata.jsonl"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    # Ensure output directory exists before training starts
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)


def run_lora_finetune(cfg: LoRAFineTuneConfig) -> Path:
    """
    Runs LoRA fine-tuning by calling the official Diffusers script
    through `accelerate launch`.

    Returns:
        Path to the directory containing the trained LoRA adapter.
    """
    _validate_inputs(cfg)

    # Build the command exactly as recommended by Diffusers
    cmd = [
        "accelerate",
        "launch",
        cfg.script_path,
        "--pretrained_model_name_or_path",
        cfg.pretrained_model,
        "--train_data_dir",
        cfg.train_data_dir,
        "--resolution",
        str(cfg.resolution),
        "--train_batch_size",
        str(cfg.batch_size),
        "--gradient_accumulation_steps",
        str(cfg.grad_accum_steps),
        "--learning_rate",
        str(cfg.learning_rate),
        "--max_train_steps",
        str(cfg.max_train_steps),
        "--rank",
        str(cfg.rank),
        "--seed",
        str(cfg.seed),
        "--output_dir",
        cfg.output_dir,
        "--mixed_precision",
        cfg.mixed_precision,
    ]

    # Optional checkpointing (supported by recent Diffusers versions)
    if cfg.checkpointing_steps is not None:
        cmd += ["--checkpointing_steps", str(cfg.checkpointing_steps)]

    # Print command for transparency and reproducibility
    print("\nRunning LoRA fine-tuning (official Diffusers script):\n")
    print(" ".join(cmd))
    print()

    # Execute training as a subprocess
    subprocess.run(cmd, check=True)

    out_dir = Path(cfg.output_dir)
    print(f"\nSaved LoRA adapter to: {out_dir}\n")
    return out_dir


def main() -> None:
    """
    Standalone entry point for LoRA fine-tuning.
    """
    cfg = LoRAFineTuneConfig()
    run_lora_finetune(cfg)


if __name__ == "__main__":
    main()
