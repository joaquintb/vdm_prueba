from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import DiffusionPipeline, LCMScheduler

from tqdm import tqdm


@dataclass(frozen=True)
class GenConfig:
    out_dir: str = "./results/generated_images"

    # Base + LCM adapter (your requested setup)
    base_model: str = "runwayml/stable-diffusion-v1-5"
    lcm_lora: str = "latent-consistency/lcm-lora-sdv1-5"

    # Fine-tuned LoRA path (Part 4). Keep None for Part 3 baseline.
    finetuned_lora_path: Optional[str] = None

    num_images: int = 10
    num_inference_steps: int = 4  # LCM: few steps
    guidance_scale: float = 1.0   # LCM typically works well around 1.0
    seed: int = 42

    batch_size: int = 1  # keep 1 for CPU; increase on GPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def label_to_prompt(label: str) -> str:
    label = label.lower().strip()
    if label == "pneumonia":
        return "chest X-ray, pneumonia, pulmonary opacity, radiology image"
    return "chest X-ray, normal lungs, no evidence of pneumonia, radiology image"


def default_prompts(n: int) -> List[str]:
    # Balanced-ish prompts: half normal, half pneumonia
    prompts: List[str] = []
    for i in range(n):
        label = "normal" if i < (n // 2) else "pneumonia"
        prompts.append(label_to_prompt(label))
    return prompts


def _load_lora(pipe: DiffusionPipeline, lora_ref: str, adapter_name: Optional[str] = None) -> None:
    """
    Loads a LoRA adapter. Supports both:
    - HF repo id (e.g. latent-consistency/...)
    - local folder path (e.g. results/lora/my_adapter/)
    """
    kwargs = {}
    if adapter_name is not None:
        kwargs["adapter_name"] = adapter_name

    try:
        pipe.load_lora_weights(lora_ref, **kwargs)
    except TypeError:
        # Older diffusers versions might not support adapter_name
        pipe.load_lora_weights(lora_ref)


def build_diff_pipeline(cfg: GenConfig) -> DiffusionPipeline:
    dtype = torch.float16 if cfg.device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(cfg.base_model, torch_dtype=dtype, safety_checker=None)

    # Switch to LCM scheduler (sampling recipe)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load LCM-LoRA adapter (speed/low-step generation)
    _load_lora(pipe, cfg.lcm_lora, adapter_name="lcm")

    # Optional: load your fine-tuned LoRA on top (Part 4)
    if cfg.finetuned_lora_path:
        _load_lora(pipe, cfg.finetuned_lora_path, adapter_name="finetuned")

    # CPU/GPU settings
    pipe = pipe.to(cfg.device)
    pipe.enable_attention_slicing()  # helps memory on CPU/GPU

    return pipe

@torch.inference_mode()
def generate_images(pipe: DiffusionPipeline, prompts: List[str], cfg: GenConfig) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs = max(1, int(cfg.batch_size))

    for start in tqdm(range(0, len(prompts), bs), desc="Generating images", unit="batch"):
        batch_prompts = prompts[start : start + bs]
        # Controlled randomness using a generator
        # One generator per image (so seeds are stable even with batching)
        generators = [
            torch.Generator(device=cfg.device).manual_seed(cfg.seed + (start + j))
            for j in range(len(batch_prompts))
        ]

        images = pipe(
            batch_prompts,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generators,
        ).images

        for j, img in enumerate(images):
            idx = start + j
            img.save(out_dir / f"gen_{idx:03d}.png")


def main() -> None:
    cfg = GenConfig()
    pipe = build_diff_pipeline(cfg)

    prompts = default_prompts(cfg.num_images)
    generate_images(pipe, prompts, cfg)

    print(f"Saved {cfg.num_images} images to: {cfg.out_dir}")

if __name__ == "__main__":
    main()
