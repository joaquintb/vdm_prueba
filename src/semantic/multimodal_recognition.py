from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

import open_clip


@dataclass(frozen=True)
class RecognitionConfig:
    """
    Configuration for BiomedCLIP inference.
    """
    model_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    context_length: int = 256  # from the model card
    cache_dir: Optional[str] = None


# Cached objects (loaded once per process)
_MODEL: Optional[torch.nn.Module] = None
_PREPROCESS = None
_TOKENIZER = None
_LOADED_CFG: Optional[RecognitionConfig] = None


def ensure_pil(img: Any) -> Image.Image:
    """
    Ensures the image is a PIL Image (MedMNIST may return PIL / numpy / tensor).
    """
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return F.to_pil_image(img)
    raise TypeError(f"Unsupported image type: {type(img)}")


def _load_model(cfg: RecognitionConfig):
    """
    Loads BiomedCLIP + preprocess + tokenizer once and reuses them across batches.
    If cfg changes (e.g., device), it reloads to avoid silent mismatches.
    """
    global _MODEL, _PREPROCESS, _TOKENIZER, _LOADED_CFG

    if _MODEL is not None and _LOADED_CFG == cfg:
        return _MODEL, _PREPROCESS, _TOKENIZER

    model, preprocess = open_clip.create_model_from_pretrained(
        cfg.model_id,
        device=cfg.device,
        cache_dir=cfg.cache_dir,
    )
    tokenizer = open_clip.get_tokenizer(cfg.model_id)

    model.eval()
    _MODEL, _PREPROCESS, _TOKENIZER = model, preprocess, tokenizer
    _LOADED_CFG = cfg

    return _MODEL, _PREPROCESS, _TOKENIZER


@torch.no_grad()
def predict_labels(
    images: List[Any],
    prompts: Sequence[str],
    cfg: RecognitionConfig,
) -> Tuple[List[str], np.ndarray]:
    """
    Assigns the most semantically similar prompt to each image using BiomedCLIP.

    Args:
        images: list of raw images (PIL / numpy / tensor)
        prompts: candidate textual prompts (one per class)
        cfg: inference configuration

    Returns:
        auto_labels: selected prompt per image
        confidence_scores: softmax confidence of the selected prompt
    """
    if len(images) == 0:
        return [], np.array([], dtype=np.float32)
    if len(prompts) == 0:
        raise ValueError("prompts must be a non-empty list of strings")

    model, preprocess, tokenizer = _load_model(cfg)

    # Preprocess images and stack into a batch tensor [B, 3, 224, 224]
    pil_images = [ensure_pil(img) for img in images]
    image_tensor = torch.stack([preprocess(img) for img in pil_images]).to(cfg.device)

    # Tokenize prompts (BiomedCLIP uses context_length=256)
    text_tokens = tokenizer(list(prompts), context_length=cfg.context_length).to(cfg.device)

    # Forward pass: features + logit_scale (as in the model card)
    image_features, text_features, logit_scale = model(image_tensor, text_tokens)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    probs = (logit_scale * (image_features @ text_features.T)).softmax(dim=-1)

    best_idx = torch.argmax(probs, dim=-1)
    best_conf = probs[torch.arange(probs.shape[0]), best_idx]

    auto_labels = [prompts[i] for i in best_idx.cpu().tolist()]
    confidence_scores = best_conf.cpu().numpy().astype(np.float32)

    return auto_labels, confidence_scores
