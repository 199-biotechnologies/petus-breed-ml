"""
Stacking meta-learner ensemble.

Collects predictions (logits/probs) from all trained backbone models,
extracts meta-features, and fits sklearn LogisticRegression as meta-learner.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .train import BreedClassifier, get_device, NUM_CLASSES


def extract_meta_features(probs: np.ndarray) -> np.ndarray:
    """Extract meta-features from a single model's softmax probabilities.

    Per-sample features (8 total):
    - top1_prob: confidence of top prediction
    - margin: gap between top-1 and top-2
    - entropy: prediction uncertainty
    - top5_probs: probabilities of top-5 classes (5 values)

    Args:
        probs: (N, num_classes) softmax probabilities
    Returns:
        (N, 8) meta-features
    """
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]  # descending

    top1 = sorted_probs[:, 0:1]
    margin = sorted_probs[:, 0:1] - sorted_probs[:, 1:2]
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1, keepdims=True)
    top5 = sorted_probs[:, :5]

    return np.concatenate([top1, margin, entropy, top5], axis=1)  # (N, 8)


def extract_logit_features(probs: np.ndarray) -> np.ndarray:
    """Use full calibrated log-probabilities as features (Codex recommendation).

    With 3 models × 120 classes = 360 features. LogisticRegression with
    strong L2 regularization handles this well and preserves breed-specific
    information that summary statistics throw away.

    Args:
        probs: (N, num_classes) softmax probabilities
    Returns:
        (N, num_classes) log-probabilities
    """
    return np.log(probs + 1e-10)


@torch.no_grad()
def collect_predictions(
    model: BreedClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on dataloader, return (probs, labels)."""
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc=f"Collecting {model.backbone_name}", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


class StackingEnsemble:
    """Meta-learner that combines predictions from multiple backbone models.

    Supports two feature modes:
    - "meta": 8 summary features per model (top1, margin, entropy, top5)
    - "logits": Full log-probabilities per model (num_classes features each)
      → 3 models × 120 classes = 360 features. Much more expressive.
    """

    def __init__(self, feature_mode: str = "logits"):
        assert feature_mode in ("meta", "logits"), f"Unknown mode: {feature_mode}"
        self.feature_mode = feature_mode
        # Stronger L2 for logit mode (360 features needs more regularization)
        C = 0.1 if feature_mode == "logits" else 1.0
        self.meta_learner = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.model_names: list[str] = []
        self.fitted = False

    def fit(
        self,
        model_predictions: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> dict:
        """Fit the meta-learner on model predictions.

        Args:
            model_predictions: {backbone_name: (N, num_classes) probs}
            labels: (N,) ground truth
        Returns:
            dict with training metrics
        """
        self.model_names = sorted(model_predictions.keys())

        # Extract features from each model and concatenate
        extractor = extract_logit_features if self.feature_mode == "logits" else extract_meta_features
        all_features = []
        for name in self.model_names:
            probs = model_predictions[name]
            features = extractor(probs)
            all_features.append(features)

        X = np.concatenate(all_features, axis=1)  # (N, 8 * num_models)
        print(f"Meta-features shape: {X.shape} ({len(self.model_names)} models × 8 features)")

        X_scaled = self.scaler.fit_transform(X)
        self.meta_learner.fit(X_scaled, labels)
        self.fitted = True

        train_acc = 100.0 * np.mean(self.meta_learner.predict(X_scaled) == labels)
        print(f"Meta-learner train accuracy: {train_acc:.1f}%")

        return {"train_acc": train_acc, "n_features": X.shape[1], "n_samples": X.shape[0]}

    def predict(self, model_predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Predict using the ensemble.

        Args:
            model_predictions: {backbone_name: (N, num_classes) probs}
        Returns:
            (predictions, probabilities) — (N,) and (N, num_classes)
        """
        if not self.fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        extractor = extract_logit_features if self.feature_mode == "logits" else extract_meta_features
        all_features = []
        for name in self.model_names:
            if name not in model_predictions:
                raise ValueError(f"Missing predictions for model '{name}'")
            features = extractor(model_predictions[name])
            all_features.append(features)

        X = np.concatenate(all_features, axis=1)
        X_scaled = self.scaler.transform(X)

        predictions = self.meta_learner.predict(X_scaled)
        probabilities = self.meta_learner.predict_proba(X_scaled)

        return predictions, probabilities

    def evaluate(self, model_predictions: dict[str, np.ndarray], labels: np.ndarray) -> dict:
        """Evaluate ensemble on test data."""
        predictions, probabilities = self.predict(model_predictions)

        top1_acc = 100.0 * np.mean(predictions == labels)

        # Top-5 accuracy
        top5_preds = np.argsort(probabilities, axis=1)[:, -5:]
        top5_correct = np.any(top5_preds == labels[:, None], axis=1)
        top5_acc = 100.0 * np.mean(top5_correct)

        return {"top1_acc": top1_acc, "top5_acc": top5_acc}

    def save(self, path: str):
        """Save ensemble to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "meta_learner": self.meta_learner,
                "scaler": self.scaler,
                "model_names": self.model_names,
            }, f)
        print(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str) -> "StackingEnsemble":
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        ensemble = cls()
        ensemble.meta_learner = data["meta_learner"]
        ensemble.scaler = data["scaler"]
        ensemble.model_names = data["model_names"]
        ensemble.fitted = True
        return ensemble
