"""
Temperature scaling calibration + Expected Calibration Error (ECE).

Post-hoc calibration: learns a single temperature T on a held-out set
so that softmax(logits / T) produces well-calibrated probabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar
from tqdm import tqdm


class TemperatureScaler(nn.Module):
    """Wraps a model and applies temperature scaling to its logits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    @torch.no_grad()
    def calibrate(self, loader: DataLoader, device: torch.device) -> float:
        """Find optimal temperature on validation data using NLL."""
        self.model.eval()
        all_logits = []
        all_labels = []

        for images, labels in tqdm(loader, desc="Collecting logits for calibration", leave=False):
            images = images.to(device)
            logits = self.model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        # Optimize temperature via scipy
        def nll_at_temp(t):
            scaled = logits / t
            return F.cross_entropy(scaled, labels).item()

        result = minimize_scalar(nll_at_temp, bounds=(0.1, 10.0), method="bounded")
        optimal_t = result.x

        self.temperature.data = torch.tensor([optimal_t])
        print(f"Optimal temperature: {optimal_t:.3f}")
        return optimal_t


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Args:
        probs: (N, C) predicted probabilities
        labels: (N,) ground truth
        n_bins: number of confidence bins
    Returns:
        ECE value (lower is better, 0 = perfect calibration)
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return ece / len(labels)


def reliability_diagram_data(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict:
    """Compute data for a reliability diagram.

    Returns dict with bin_centers, bin_accuracies, bin_confidences, bin_counts.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
        bin_accuracies.append(accuracies[mask].mean())
        bin_confidences.append(confidences[mask].mean())
        bin_counts.append(count)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }
