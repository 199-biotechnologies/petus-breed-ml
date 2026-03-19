"""
Production inference pipeline.

Single entry point that handles:
1. Load image → preprocess
2. Run through single model or ensemble
3. Optional TTA for low-confidence predictions
4. Return calibrated breed prediction with confidence
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .train import BreedClassifier, load_model, get_device, NUM_CLASSES
from .tta import predict_with_tta
from .dataset import get_breed_names


def load_image(image_path: str) -> Image.Image:
    """Load and validate an image."""
    img = Image.open(image_path).convert("RGB")
    return img


def preprocess_image(
    img: Image.Image,
    preprocess_config: dict,
) -> torch.Tensor:
    """Preprocess a PIL image for model input.

    Returns (1, C, H, W) tensor.
    """
    img_size = preprocess_config.get("input_size", 224)
    mean = preprocess_config.get("mean", [0.485, 0.456, 0.406])
    std = preprocess_config.get("std", [0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transform(img).unsqueeze(0)


@torch.no_grad()
def predict_single(
    model: BreedClassifier,
    image_path: str,
    breed_names: list[str],
    device: torch.device = None,
    use_tta: bool = True,
    tta_threshold: float = 0.7,
    top_k: int = 5,
) -> list[dict]:
    """Predict breed from a single image using one model.

    Returns list of top-K predictions with breed name and confidence.
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    img = load_image(image_path)
    preprocess_config = model.get_preprocess_config()
    tensor = preprocess_image(img, preprocess_config).to(device)

    if use_tta:
        probs = predict_with_tta(model, tensor, device, confidence_threshold=tta_threshold)
    else:
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    top_probs, top_indices = probs.topk(top_k, dim=1)

    results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        results.append({
            "breed": breed_names[idx],
            "confidence": round(prob * 100, 1),
            "class_idx": idx,
        })

    return results


@torch.no_grad()
def predict_ensemble(
    models: dict[str, BreedClassifier],
    image_path: str,
    breed_names: list[str],
    device: torch.device = None,
    use_tta: bool = True,
    top_k: int = 5,
) -> list[dict]:
    """Predict breed using multiple models (simple averaging).

    For full stacking ensemble, use the StackingEnsemble class.
    """
    if device is None:
        device = get_device()

    img = load_image(image_path)
    all_probs = []

    for name, model in models.items():
        model.eval()
        model.to(device)
        preprocess_config = model.get_preprocess_config()
        tensor = preprocess_image(img, preprocess_config).to(device)

        if use_tta:
            probs = predict_with_tta(model, tensor, device)
        else:
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)

        all_probs.append(probs)

    # Average probabilities across models
    avg_probs = torch.stack(all_probs).mean(dim=0)
    top_probs, top_indices = avg_probs.topk(top_k, dim=1)

    results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        results.append({
            "breed": breed_names[idx],
            "confidence": round(prob * 100, 1),
            "class_idx": idx,
        })

    return results


def print_predictions(results: list[dict], title: str = "Predictions"):
    """Pretty-print prediction results."""
    print(f"\n{title}")
    print("-" * 40)
    for i, r in enumerate(results):
        bar = "█" * int(r["confidence"] / 2)
        print(f"  {i+1}. {r['breed']:30s} {r['confidence']:5.1f}% {bar}")
