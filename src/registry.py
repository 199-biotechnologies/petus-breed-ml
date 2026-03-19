"""
Model registry — @register decorator pattern.

Adding a new model = 1 file in backbones/, decorated with @register("name").
Training, ensemble, and inference code never import specific models.
"""

from __future__ import annotations

import torch.nn as nn

_BACKBONE_REGISTRY: dict[str, type[nn.Module]] = {}


def register(name: str):
    """Decorator to register a backbone class by name."""
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in _BACKBONE_REGISTRY:
            raise ValueError(f"Backbone '{name}' already registered")
        _BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator


def get_backbone(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered backbone by name."""
    if name not in _BACKBONE_REGISTRY:
        available = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unknown backbone '{name}'. Available: {available}")
    return _BACKBONE_REGISTRY[name](**kwargs)


def list_backbones() -> list[str]:
    """Return names of all registered backbones."""
    return sorted(_BACKBONE_REGISTRY.keys())
