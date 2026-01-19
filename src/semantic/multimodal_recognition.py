"""
multimodal_recognition.py

Part 2: CLIP-like radiology semantic recognition (zero-shot).

This module wraps BiomedCLIP inference:
- Converts raw dataset images (PIL / numpy / tensor) into a consistent format.
- Uses the model-provided preprocess for correct normalization/resizing.
- Computes text-image similarities against a small set of prompts and returns
  the best-matching label + a confidence score.

Important: the model is cached in-process to avoid re-loading weights for every batch.
"""

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

    cache_dir can be used to control where HF/OpenCLIP stores weights
    (useful in Docker/CI environments).
    """
    model_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    context_length: int = 256  # from the model card
    cache_dir: Optional[str] = None


# Cached objects (loaded once per process).
# This is a big runtime win: we label the dataset in many batches, so reloading the
# model each call would dominate runtime.
_MODEL: Optional[torch.nn.Module] = None
_PREPROCESS = None
_TOKENIZER = None
_LOADED_CFG: Optional[RecognitionConfig] = None


def ensure_pil(img: Any) -> Image.Image:
    """
    Convert an input image to PIL.Image.

    MedMNIST can return different types depending on how it is accessed
    (PIL, numpy arrays, or torch tensors). BiomedCLIP's preprocess expects PIL.
    """
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return F.to_pil_image(img)
    raise TypeError(f"Unsupported image type: {type(img)}")


def _load_model(cfg: RecognitionConfig):
    """
    Load BiomedCLIP + preprocess + tokenizer once and reuse them across batches.

    If the config changes (e.g., device), we reload to avoid subtle mismatches
    such as a CPU model being used with CUDA tensors.
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
    Assign the most semantically similar prompt to each image (zero-shot).

    We:
    1) preprocess images using the model's preprocessing pipeline,
    2) tokenize prompts,
    3) compute normalized embeddings,
    4) convert similarities into probabilities via softmax,
    5) pick the argmax prompt as auto_label and its probability as confidence.

    Returns:
        auto_labels: selected prompt per image (one prompt per sample)
        confidence_scores: softmax confidence for the selected prompt
    """
    if len(images) == 0:
        return [], np.array([], dtype=np.float32)
    if len(prompts) == 0:
        raise ValueError("prompts must be a non-empty list of strings")

    model, preprocess, tokenizer = _load_model(cfg)

    # Preprocess images and stack into a batch tensor [B, 3, 224, 224].
    # We intentionally rely on BiomedCLIP's preprocess to match training-time normalization.
    pil_images = [ensure_pil(img) for img in images]
    image_tensor = torch.stack([preprocess(img) for img in pil_images]).to(cfg.device)

    # Tokenize prompts. BiomedCLIP expects context_length=256 (from the model card).
    text_tokens = tokenizer(list(prompts), context_length=cfg.context_length).to(cfg.device)

    # Forward pass: returns embeddings for both modalities and the model's learned logit scale.
    image_features, text_features, logit_scale = model(image_tensor, text_tokens)

    # L2-normalization makes the dot product equivalent to cosine similarity,
    # which is the standard CLIP retrieval setup.
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Similarity -> probabilities. logit_scale calibrates the sharpness of the softmax.
    probs = (logit_scale * (image_features @ text_features.T)).softmax(dim=-1)

    best_idx = torch.argmax(probs, dim=-1)
    best_conf = probs[torch.arange(probs.shape[0]), best_idx]

    auto_labels = [prompts[i] for i in best_idx.cpu().tolist()]
    confidence_scores = best_conf.cpu().numpy().astype(np.float32)

    return auto_labels, confidence_scores
