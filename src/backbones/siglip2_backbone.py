"""
SigLIP 2 ViT-B/16 backbone via HuggingFace transformers.

SigLIP 2 is Google's improved vision-language model with strong
image classification features. We use only the vision encoder.
Uses 0.5/0.5 normalization (NOT ImageNet stats).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

from ..registry import register


@register("siglip2_vitb")
class SigLIP2Backbone(nn.Module):
    embed_dim = 768

    def __init__(self, pretrained: bool = True):
        super().__init__()
        model_name = "google/siglip2-base-patch16-224"
        if pretrained:
            full_model = AutoModel.from_pretrained(model_name)
            self.vision_model = full_model.vision_model
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            full_model = AutoModel.from_config(config)
            self.vision_model = full_model.vision_model
        # Freeze by default
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SigLIP vision model returns pooler_output (CLS-like)
        outputs = self.vision_model(pixel_values=x)
        return outputs.pooler_output

    def freeze(self) -> None:
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def unfreeze(self, last_n_layers: int = 3) -> None:
        """Unfreeze last N encoder layers + post-layernorm."""
        for param in self.vision_model.parameters():
            param.requires_grad = False

        encoder_layers = self.vision_model.encoder.layers
        total = len(encoder_layers)
        for i in range(total - last_n_layers, total):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

        # Unfreeze post-layernorm and pooler
        if hasattr(self.vision_model, "post_layernorm"):
            for param in self.vision_model.post_layernorm.parameters():
                param.requires_grad = True
        if hasattr(self.vision_model, "head"):
            for param in self.vision_model.head.parameters():
                param.requires_grad = True

    def get_param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        trainable = [p for p in self.vision_model.parameters() if p.requires_grad]
        if trainable:
            return [{"params": trainable, "lr": lr * backbone_lr_mult}]
        return []

    def get_preprocess_config(self) -> dict:
        return {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "input_size": 224,
        }
