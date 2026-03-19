"""
MLP classification head — shared across all backbones.

LayerNorm → Linear → GELU → Dropout → Linear → num_classes
"""

import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)
