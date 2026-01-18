from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from diffusers import DiffusionPipeline, LCMScheduler
from tqdm import tqdm


@dataclass(frozen=True)
class GenConfig:
    out_dir: str = "./results/generated_images"

    # Base + LCM adapter (fast 4-step sampling)
    base_model: str = "runwayml/stable-diffusion-v1-5"
    lcm_lora: str = "latent-consistency/lcm-lora-sdv1-5"

    # Your fine-tuned LoRA (local). Can be:
    # - a folder containing LoRA weights
    # - OR a single file: .../pytorch_lora_weights.safetensors
    finetuned_lora_path: Optional[str] = None

    # How much to apply each adapter when multiple are supported
    lcm_weight: float = 1.0
    finetuned_weight: float = 1.0

    num_images: int = 10
    num_inference_steps: int = 6
    guidance_scale: float = 1.0
    seed: int = 42

    batch_size: int = 2  # T4 can usually handle 2 at 512x512; drop to 1 if OOM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def label_to_prompt(label: str) -> str:
    label = label.lower().strip()
    if label == "pneumonia":
        return "chest X-ray, pneumonia, pulmonary opacity, radiology image"
    return "chest X-ray, normal lungs, no evidence of pneumonia, radiology image"


def default_prompts(n: int) -> List[str]:
    prompts: List[str] = []
    for i in range(n):
        label = "normal" if i < (n // 2) else "pneumonia"
        prompts.append(label_to_prompt(label))
    return prompts


def _split_lora_ref(lora_ref: str) -> Tuple[str, Optional[str]]:
    """
    If lora_ref is a file (.../*.safetensors), return (parent_dir, weight_name).
    If it's a folder or HF repo id, return (lora_ref, None).
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
    Loads a LoRA adapter from either:
    - HF repo id (e.g. latent-consistency/...)
    - local folder
    - local .safetensors file
    """
    base_ref, weight_name = _split_lora_ref(lora_ref)

    kwargs = {}
    if adapter_name is not None:
        kwargs["adapter_name"] = adapter_name
    if weight_name is not None:
        kwargs["weight_name"] = weight_name

    pipe.load_lora_weights(base_ref, **kwargs)


def build_diff_pipeline(cfg: GenConfig) -> DiffusionPipeline:
    # On Colab GPU, fp16 is the right default. On CPU, use fp32.
    dtype = torch.float16 if cfg.device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        safety_checker=None,
    )

    # LCM sampling recipe
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load adapters
    _load_lora(pipe, cfg.lcm_lora, adapter_name="lcm")

    if cfg.finetuned_lora_path:
        _load_lora(pipe, cfg.finetuned_lora_path, adapter_name="finetuned")

    # Activate both adapters (if supported). If not, Diffusers will still apply the last loaded
    # adapter; but most recent versions support multi-adapter blending.
    if cfg.finetuned_lora_path:
        try:
            pipe.set_adapters(
                ["lcm", "finetuned"],
                adapter_weights=[cfg.lcm_weight, cfg.finetuned_weight],
            )
        except Exception:
            pass  # fallback: keep default behavior

    pipe = pipe.to(cfg.device)

    # Memory helpers
    pipe.enable_attention_slicing()
    if cfg.device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


@torch.inference_mode()
def generate_images(pipe: DiffusionPipeline, prompts: List[str], cfg: GenConfig) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs = max(1, int(cfg.batch_size))

    for start in tqdm(range(0, len(prompts), bs), desc="Generating images", unit="batch"):
        batch_prompts = prompts[start : start + bs]

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
        images = result.images

        for j, img in enumerate(images):
            idx = start + j
            img.save(out_dir / f"gen_{idx:03d}.png")


def main() -> None:
    cfg = GenConfig(
        finetuned_lora_path="./results/lora/pneumonia_lora/pytorch_lora_weights.safetensors",
        num_images=10,
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