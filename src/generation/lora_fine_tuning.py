from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from medmnist import PneumoniaMNIST


@dataclass(frozen=True)
class FineTuneConfig:
    """
    LoRA fine-tuning wrapper around the official Diffusers training script.

    Steps:
    1) Export a small (image, prompt) dataset to disk.
    2) Call the official Diffusers LoRA script via `accelerate launch`.
    3) Save the LoRA adapter into `output_dir`, loadable with `pipe.load_lora_weights(output_dir)`.
    """

    # Model
    base_model: str = "runwayml/stable-diffusion-v1-5"

    # Data
    data_root: str = "./data"
    split: str = "train"
    max_samples: int = 200
    image_size: int = 256  # keep small for speed

    # Prompts derived from label (0 = normal, 1 = pneumonia)
    prompt_normal: str = "chest X-ray, normal lungs, no evidence of pneumonia, radiology image"
    prompt_pneumonia: str = "chest X-ray, pneumonia, pulmonary opacity, radiology image"

    # Training (light)
    batch_size: int = 1
    grad_accum_steps: int = 4
    learning_rate: float = 1e-4
    max_steps: int = 200
    lora_rank: int = 8
    seed: int = 42
    mixed_precision: str = "fp16"  # typical on Colab GPUs

    # Output
    train_data_dir: str = "./results/lora/train_data"
    output_dir: str = "./results/lora/pneumonia_lora"


def _label_to_prompt(y: int, cfg: FineTuneConfig) -> str:
    return cfg.prompt_normal if y == 0 else cfg.prompt_pneumonia


def export_train_data(cfg: FineTuneConfig) -> Path:
    """
    Export a tiny dataset to a directory that the official Diffusers script can read:
      - images/*.png
      - metadata.jsonl with {"file_name": "...", "text": "..."} per image
    """
    out_dir = Path(cfg.train_data_dir)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    ds = PneumoniaMNIST(split=cfg.split, root=cfg.data_root, download=True, transform=None)
    n = min(cfg.max_samples, len(ds))

    tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Lambda(lambda im: im.convert("RGB")),
        ]
    )

    meta_path = out_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for i in tqdm(range(n), desc="Exporting LoRA train data", unit="img"):
            img, label = ds[i]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            img = tf(img)

            y = int(label[0])  # MedMNIST labels are shape [1]
            prompt = _label_to_prompt(y, cfg)

            fname = f"img_{i:05d}.png"
            img.save(images_dir / fname)

            f.write(json.dumps({"file_name": f"images/{fname}", "text": prompt}) + "\n")

    return out_dir


def find_official_diffusers_lora_script() -> Path:
    """
    Locate the official Diffusers LoRA training script in the current environment.
    If it isn't available, we raise an error with a clear message.
    """
    import diffusers  # noqa

    pkg_dir = Path(sys.modules["diffusers"].__file__).resolve().parent

    candidates = [
        pkg_dir / "examples" / "text_to_image" / "train_text_to_image_lora.py",
        pkg_dir.parent / "diffusers" / "examples" / "text_to_image" / "train_text_to_image_lora.py",
        pkg_dir.parent / "examples" / "text_to_image" / "train_text_to_image_lora.py",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find the official Diffusers LoRA training script (train_text_to_image_lora.py) in this environment.\n"
        "Recommended fix:\n"
        "  1) Vendor the script into your repo (e.g., third_party/diffusers/train_text_to_image_lora.py)\n"
        "     and update this function to point to it.\n"
        "  2) Or install Diffusers from source so examples are available.\n"
    )


def run_official_lora_training(cfg: FineTuneConfig, train_dir: Path, script_path: Path) -> Path:
    """
    Call the official Diffusers LoRA training script via `accelerate launch`.
    Produces an adapter in `cfg.output_dir`, loadable with `pipe.load_lora_weights(cfg.output_dir)`.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate",
        "launch",
        str(script_path),
        "--pretrained_model_name_or_path",
        cfg.base_model,
        "--train_data_dir",
        str(train_dir),
        "--resolution",
        str(cfg.image_size),
        "--train_batch_size",
        str(cfg.batch_size),
        "--gradient_accumulation_steps",
        str(cfg.grad_accum_steps),
        "--learning_rate",
        str(cfg.learning_rate),
        "--max_train_steps",
        str(cfg.max_steps),
        "--rank",
        str(cfg.lora_rank),
        "--seed",
        str(cfg.seed),
        "--output_dir",
        str(out_dir),
        "--mixed_precision",
        cfg.mixed_precision,
    ]

    print("\nRunning official Diffusers LoRA training:")
    print(" ".join(cmd), "\n")
    subprocess.run(cmd, check=True)

    return out_dir


def main() -> None:
    cfg = FineTuneConfig()

    train_dir = export_train_data(cfg)
    script_path = find_official_diffusers_lora_script()
    out_dir = run_official_lora_training(cfg, train_dir, script_path)

    print(f"Saved LoRA adapter to: {out_dir}")


if __name__ == "__main__":
    main()
