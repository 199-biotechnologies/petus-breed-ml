"""
DINOv2 ViT-B/14 backbone — frozen self-supervised features + trainable head.

Ported from unblurml DINOv2Classifier pattern.
DINOv2 produces excellent general-purpose features; we freeze the backbone
and only train the MLP head, optionally unfreezing last N blocks.
"""

import torch
import torch.nn as nn

from ..registry import register


@register("dinov2_vitb14")
class DINOv2Backbone(nn.Module):
    embed_dim = 768

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14",
            trust_repo=True,
        )
        if pretrained:
            self.backbone.eval()
        # Freeze by default — phase 1
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 returns CLS token embedding
        return self.backbone(x)

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self, last_n_blocks: int = 3) -> None:
        """Unfreeze last N transformer blocks + norm layer."""
        # First unfreeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False  # Start frozen

        # Unfreeze last N blocks
        total_blocks = len(self.backbone.blocks)
        for i in range(total_blocks - last_n_blocks, total_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # Always unfreeze final norm
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def get_param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        trainable = [p for p in self.backbone.parameters() if p.requires_grad]
        if trainable:
            return [{"params": trainable, "lr": lr * backbone_lr_mult}]
        return []

    def get_preprocess_config(self) -> dict:
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "input_size": 224,  # DINOv2 ViT-B/14: patch_size=14, 224/14=16 patches
        }
