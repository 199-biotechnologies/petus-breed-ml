"""
Loss functions for fine-grained classification.

ArcFace: Angular margin loss — forces angular separation between breed embeddings.
Poly-1: Drop-in CE replacement with polynomial adjustment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace Additive Angular Margin Loss.

    Projects features onto a hypersphere and enforces angular margin
    between classes. Excellent for fine-grained classification where
    visually similar classes (e.g., Staffordshire vs AmStaff) need
    strong discriminative boundaries.

    Args:
        embed_dim: Feature embedding dimension
        num_classes: Number of classes
        scale: Feature scale (s). Default: 30.0
        margin: Angular margin (m) in radians. Default: 0.3
        label_smoothing: Smoothing factor. Default: 0.0
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.3,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

        # Learnable class weight vectors (on unit hypersphere)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, embed_dim) — raw features from backbone (NOT logits)
            labels: (B,) — ground truth class indices
        """
        # Normalize embeddings and weights to unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity (dot product on unit sphere)
        cosine = F.linear(embeddings, weight)  # (B, num_classes)
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))

        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical safety: when cos(θ) < cos(π - m), use linearized version
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        # Standard cross-entropy with optional label smoothing
        return F.cross_entropy(output, labels, label_smoothing=self.label_smoothing)


class ArcFaceHead(nn.Module):
    """Combined ArcFace projection head — replaces the standard MLP + CE pipeline.

    Takes raw backbone features, projects to embedding space, then applies ArcFace.
    During inference, use the projected embeddings for classification via cosine similarity.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        projection_dim: int = 512,
        scale: float = 30.0,
        margin: float = 0.3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.arcface = ArcFaceLoss(
            embed_dim=projection_dim,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
        )
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None):
        """
        During training (labels provided): returns ArcFace loss
        During inference (no labels): returns cosine similarity logits
        """
        projected = self.projector(features)

        if labels is not None:
            # Training mode: return loss
            return self.arcface(projected, labels)
        else:
            # Inference mode: return cosine similarity as logits
            projected = F.normalize(projected, p=2, dim=1)
            weight = F.normalize(self.arcface.weight, p=2, dim=1)
            return F.linear(projected, weight) * self.arcface.scale


class Poly1Loss(nn.Module):
    """Poly-1 Cross-Entropy Loss.

    Near drop-in replacement for CE. Adds a polynomial correction term
    that helps with hard examples. From "PolyLoss" paper (ICLR 2022).

    Args:
        num_classes: Number of classes
        epsilon: Polynomial coefficient. Default: 1.0
        label_smoothing: Smoothing factor. Default: 0.1
    """

    def __init__(self, num_classes: int = 120, epsilon: float = 1.0, label_smoothing: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        # Poly-1 adjustment
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(labels, self.num_classes).float()

        if self.label_smoothing > 0:
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        pt = (probs * one_hot).sum(dim=1)  # Probability of true class
        poly1 = ce_loss + self.epsilon * (1 - pt).mean()

        return poly1
