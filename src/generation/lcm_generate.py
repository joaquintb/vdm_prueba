"""
lcm_generate.py

Part 3: LCM-based image generation for medical text-to-image synthesis.

This script generates synthetic chest X-ray images conditioned on text prompts
using Stable Diffusion + Latent Consistency Models (LCM), optionally combined
with a fine-tuned LoRA adapter trained on medical data.

Designed to be fast (few inference steps) and runnable on GPU (with CPU fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from diffusers import DiffusionPipeline, LCMScheduler
from tqdm import tqdm


@dataclass(frozen=True)
class GenConfig:
    """
    Configuration for image generation with LCM + LoRA.
    """
    out_dir: str = "./results/generated_images"

    # Base diffusion model + LCM adapter for fast sampling
    base_model: str = "runwayml/stable-diffusion-v1-5"
    lcm_lora: str = "latent-consistency/lcm-lora-sdv1-5"

    # Optional fine-tuned LoRA adapter (local path or folder)
    finetuned_lora_path: Optional[str] = None

    # Relative strength when combining multiple adapters
    lcm_weight: float = 1.0
    finetuned_weight: float = 1.0

    # Generation parameters
    num_images: int = 30
    num_inference_steps: int = 6
    guidance_scale: float = 1.0
    seed: int = 42

    # Batch size tuned for Colab T4; reduce if OOM
    batch_size: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def label_to_prompt(label: str) -> str:
    """
    Maps a binary class label to a descriptive medical text prompt.
    """
    label = label.lower().strip()
    if label == "pneumonia":
        return "chest X-ray, pneumonia, pulmonary opacity, radiology image"
    return "chest X-ray, normal lungs, no evidence of pneumonia, radiology image"


def default_prompts(n: int) -> List[str]:
    """
    Generates a simple, balanced list of prompts (half normal / half pneumonia).
    """
    prompts: List[str] = []
    for i in range(n):
        label = "normal" if i < (n // 2) else "pneumonia"
        prompts.append(label_to_prompt(label))
    return prompts


def _split_lora_ref(lora_ref: str) -> Tuple[str, Optional[str]]:
    """
    Handles different LoRA formats.

    If a .safetensors file is provided, Diffusers expects:
      - base directory
      - explicit weight file name

    Otherwise, treat it as a folder or Hugging Face repo ID.
    """
    p = Path(lora_ref)
    if p.suffix.lower() == ".safetensors" and p.exists():
        return str(p.parent), p.name
    return lora_ref, None


def _load_lora(
    pipe: DiffusionPipeline,
    lora_ref: str,
    adapter_name: Optional[str] = None,
) -> None:
    """
    Loads a LoRA adapter into the diffusion pipeline.

    Supports:
    - Hugging Face repo IDs
    - local directories
    - single .safetensors files
    """
    base_ref, weight_name = _split_lora_ref(lora_ref)

    kwargs = {}
    if adapter_name is not None:
        kwargs["adapter_name"] = adapter_name
    if weight_name is not None:
        kwargs["weight_name"] = weight_name

    pipe.load_lora_weights(base_ref, **kwargs)


def build_diff_pipeline(cfg: GenConfig) -> DiffusionPipeline:
    """
    Builds and configures the diffusion pipeline:
    - loads base Stable Diffusion
    - swaps scheduler to LCM
    - loads LoRA adapters
    - applies memory optimizations
    """
    # fp16 on GPU for speed/memory, fp32 on CPU for correctness
    dtype = torch.float16 if cfg.device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        safety_checker=None,  # not needed for medical images
    )

    # Replace scheduler with LCM scheduler (key for fast generation)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load LCM LoRA (required)
    _load_lora(pipe, cfg.lcm_lora, adapter_name="lcm")

    # Load fine-tuned LoRA if provided
    if cfg.finetuned_lora_path:
        _load_lora(pipe, cfg.finetuned_lora_path, adapter_name="finetuned")

    # If supported, blend adapters explicitly
    if cfg.finetuned_lora_path:
        try:
            pipe.set_adapters(
                ["lcm", "finetuned"],
                adapter_weights=[cfg.lcm_weight, cfg.finetuned_weight],
            )
        except Exception:
            # Older diffusers versions may not support adapter blending
            pass

    pipe = pipe.to(cfg.device)

    # Memory-saving options (important for Colab GPUs)
    pipe.enable_attention_slicing()
    if cfg.device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


@torch.inference_mode()
def generate_images(pipe: DiffusionPipeline, prompts: List[str], cfg: GenConfig) -> None:
    """
    Generates images from text prompts in batches and saves them to disk.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs = max(1, int(cfg.batch_size))

    for start in tqdm(range(0, len(prompts), bs), desc="Generating images", unit="batch"):
        batch_prompts = prompts[start : start + bs]

        # Use per-image generators for deterministic but varied outputs
        generators = [
            torch.Generator(device=cfg.device).manual_seed(cfg.seed + (start + j))
            for j in range(len(batch_prompts))
        ]

        result = pipe(
            batch_prompts,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generators,
        )

        for j, img in enumerate(result.images):
            idx = start + j
            img.save(out_dir / f"gen_{idx:03d}.png")


def main() -> None:
    """
    Minimal entry point for standalone generation runs.
    """
    cfg = GenConfig(
        finetuned_lora_path="./results/lora/pneumonia_lora/pytorch_lora_weights.safetensors",
        num_images=30,
    )

    if cfg.device != "cuda":
        print(
            "[Warning] CUDA not available. This will run on CPU and will be very slow.\n"
            "In Colab, choose Runtime -> Change runtime type -> GPU."
        )

    pipe = build_diff_pipeline(cfg)
    prompts = default_prompts(cfg.num_images)
    generate_images(pipe, prompts, cfg)

    print(f"Saved {cfg.num_images} images to: {cfg.out_dir}")


if __name__ == "__main__":
    main()