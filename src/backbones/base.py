"""
Backbone protocol — every backbone must expose these.

This is a structural (duck-typing) protocol, not an ABC.
Backbones are free to subclass nn.Module directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class BackboneProtocol(Protocol):
    """What every backbone must implement."""

    embed_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings (B, embed_dim)."""
        ...

    def freeze(self) -> None:
        """Freeze backbone weights for phase-1 training."""
        ...

    def unfreeze(self) -> None:
        """Unfreeze all weights for phase-2 training."""
        ...

    def get_param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        """Return optimizer param groups with differential LR."""
        ...

    def get_preprocess_config(self) -> dict:
        """Return normalization config: mean, std, input_size."""
        ...
