"""
timm-based backbones: EfficientNetV2-S and ConvNeXt V2 Tiny.

These are fully fine-tunable models with ImageNet pretraining.
"""

import torch
import torch.nn as nn
import timm

from ..registry import register


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TimmBackbone(nn.Module):
    """Generic wrapper around any timm model used as a feature extractor."""

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        # num_classes=0 → returns pooled features, no classifier head
        self.embed_dim = self.model.num_features
        self._model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        return [{"params": list(self.model.parameters()), "lr": lr * backbone_lr_mult}]

    def get_preprocess_config(self) -> dict:
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        return {
            "mean": list(data_cfg["mean"]),
            "std": list(data_cfg["std"]),
            "input_size": data_cfg["input_size"][-1],  # e.g. 224
        }


@register("efficientnetv2_s")
class EfficientNetV2S(TimmBackbone):
    def __init__(self, pretrained: bool = True):
        super().__init__("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=pretrained)


@register("convnextv2_tiny")
class ConvNeXtV2Tiny(TimmBackbone):
    def __init__(self, pretrained: bool = True):
        super().__init__("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=pretrained)
