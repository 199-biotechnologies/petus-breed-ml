"""
Advanced augmentations: MixUp, CutMix at batch level.

These are applied after the dataloader returns a batch,
not as part of the per-image transform pipeline.
"""

import torch
import numpy as np


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> tuple:
    """MixUp: convex combination of pairs of examples."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple:
    """CutMix: cut and paste patches between training images."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, h, w = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(h)
    cx = np.random.randint(w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to the actual area ratio
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)

    return x_cut, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for MixUp/CutMix: weighted sum of losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
