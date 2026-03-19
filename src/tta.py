"""
Test-Time Augmentation — adaptive, only triggers when confidence is low.

Augmentations:
- Horizontal flip
- Multi-scale (0.9×, 1.0×, 1.1×)
- Five-crop (center + 4 corners)
"""

import torch
import torch.nn.functional as F
from torchvision import transforms


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    confidence_threshold: float = 0.7,
) -> torch.Tensor:
    """Adaptive TTA — only apply augmentations if base confidence is low.

    Args:
        model: trained classifier
        image: (1, C, H, W) preprocessed tensor
        device: compute device
        confidence_threshold: skip TTA if base confidence exceeds this

    Returns:
        (1, num_classes) softmax probabilities
    """
    model.eval()
    image = image.to(device)

    # Base prediction
    logits = model(image)
    probs = F.softmax(logits, dim=1)
    base_conf = probs.max().item()

    if base_conf >= confidence_threshold:
        return probs

    # Low confidence → apply TTA
    all_probs = [probs]

    # 1. Horizontal flip
    flipped = torch.flip(image, dims=[3])
    all_probs.append(F.softmax(model(flipped), dim=1))

    # 2. Multi-scale
    _, _, h, w = image.shape
    for scale in [0.9, 1.1]:
        sh, sw = int(h * scale), int(w * scale)
        scaled = F.interpolate(image, size=(sh, sw), mode="bilinear", align_corners=False)
        # Resize back to original for model input
        scaled = F.interpolate(scaled, size=(h, w), mode="bilinear", align_corners=False)
        all_probs.append(F.softmax(model(scaled), dim=1))

    # 3. Center crop variants (tighter crop)
    crop_h, crop_w = int(h * 0.85), int(w * 0.85)
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    cropped = image[:, :, top:top+crop_h, left:left+crop_w]
    cropped = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
    all_probs.append(F.softmax(model(cropped), dim=1))

    return torch.stack(all_probs).mean(dim=0)


@torch.no_grad()
def predict_batch_with_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    confidence_threshold: float = 0.7,
) -> torch.Tensor:
    """Batch TTA — applies TTA only to low-confidence samples in batch.

    Args:
        images: (B, C, H, W) batch
    Returns:
        (B, num_classes) probabilities
    """
    model.eval()
    images = images.to(device)

    logits = model(images)
    probs = F.softmax(logits, dim=1)

    # Find low-confidence samples
    max_conf = probs.max(dim=1).values
    low_conf_mask = max_conf < confidence_threshold

    if not low_conf_mask.any():
        return probs

    # Apply TTA only to low-confidence samples
    low_conf_images = images[low_conf_mask]
    tta_probs = []
    for i in range(low_conf_images.size(0)):
        img = low_conf_images[i:i+1]
        tta_probs.append(predict_with_tta(model, img, device, confidence_threshold=0.0))

    # Replace low-confidence predictions
    result = probs.clone()
    result[low_conf_mask] = torch.cat(tta_probs, dim=0)
    return result
