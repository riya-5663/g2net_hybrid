"""CNN branch: 4-channel SFT image -> embedding (+ its own logit for CNN-only baselines)."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    def __init__(self, model_name: str = "convnext_tiny", pretrained: bool = False,
                 in_chans: int = 4, embed_dim: int = 256, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans,
                                          num_classes=0, drop_rate=drop_rate)
        self.proj = nn.Sequential(nn.Linear(self.backbone.num_features, embed_dim), nn.GELU(), nn.Dropout(drop_rate))
        self.head = nn.Linear(embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor):
        emb = self.proj(self.backbone(images))
        return emb, self.head(emb).squeeze(1)
