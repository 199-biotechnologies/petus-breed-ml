"""
DINOv3 ViT-B/16 backbone via timm (Meta, August 2025).

7B-param SSL model trained on 1.7B images, distilled to ViT-B (86M params).
Uses Gram anchoring — first SSL model to outperform weakly-supervised models.
Exceptional fine-grained features from multi-crop pretraining (2 global + 8 local).
"""

import torch
import torch.nn as nn
import timm

from ..registry import register


@register("dinov3_vitb")
class DINOv3Backbone(nn.Module):
    embed_dim = 768

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_dinov3",
            pretrained=pretrained,
            num_classes=0,  # Feature extractor mode
        )
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self, last_n_blocks: int = 4) -> None:
        """Unfreeze last N transformer blocks + norm."""
        for param in self.model.parameters():
            param.requires_grad = False

        blocks = self.model.blocks
        total = len(blocks)
        for i in range(total - last_n_blocks, total):
            for param in blocks[i].parameters():
                param.requires_grad = True

        # Unfreeze final norm
        if hasattr(self.model, "norm"):
            for param in self.model.norm.parameters():
                param.requires_grad = True

    def get_param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if trainable:
            return [{"params": trainable, "lr": lr * backbone_lr_mult}]
        return []

    def get_preprocess_config(self) -> dict:
        cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        return {
            "mean": list(cfg["mean"]),
            "std": list(cfg["std"]),
            "input_size": cfg["input_size"][-1],
        }
