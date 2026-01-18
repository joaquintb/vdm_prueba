from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoRAFineTuneConfig:
    """
    Thin wrapper to run the official Diffusers LoRA text-to-image fine-tuning script.

    Assumes you already have a small dataset on disk:
      data/subset/images/*.png
      data/subset/metadata.jsonl   ({"file_name": "...", "text": "..."} per line)

    The output is a LoRA adapter directory loadable with:
      pipe.load_lora_weights(<output_dir>)
    """

    # Vendored official script (keep it pinned to your diffusers version tag)
    script_path: str = "third_party/diffusers/train_text_to_image_lora.py"

    # Input dataset
    train_data_dir: str = "data/subset"
    resolution: int = 256

    # Base model
    pretrained_model: str = "runwayml/stable-diffusion-v1-5"

    # Training (light, for a T4)
    batch_size: int = 1
    grad_accum_steps: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 200
    rank: int = 8
    seed: int = 42
    mixed_precision: str = "fp16"  # good default on T4

    # Output
    output_dir: str = "results/lora/pneumonia_lora"

    # Optional: reduces disk usage (keeps only last checkpoint if enabled by script)
    checkpointing_steps: int | None = None


def _validate_inputs(cfg: LoRAFineTuneConfig) -> None:
    script = Path(cfg.script_path)
    if not script.exists():
        raise FileNotFoundError(
            f"Could not find vendored Diffusers script at: {script}\n"
            "Make sure you copied train_text_to_image_lora.py into third_party/diffusers/."
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

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def run_lora_finetune(cfg: LoRAFineTuneConfig) -> Path:
    """
    Runs LoRA fine-tuning via `accelerate launch`.
    Returns the output directory containing the LoRA adapter.
    """
    _validate_inputs(cfg)

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

    # Optional checkpointing (only if the script supports it; v0.36.0 does)
    if cfg.checkpointing_steps is not None:
        cmd += ["--checkpointing_steps", str(cfg.checkpointing_steps)]

    print("\nRunning LoRA fine-tuning (official Diffusers script):\n")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)

    out_dir = Path(cfg.output_dir)
    print(f"\nSaved LoRA adapter to: {out_dir}\n")
    return out_dir


def main() -> None:
    cfg = LoRAFineTuneConfig()
    run_lora_finetune(cfg)


if __name__ == "__main__":
    main()
