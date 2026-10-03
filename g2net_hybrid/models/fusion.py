"""Late-fusion hybrid: [CNN embedding ; physics embedding] -> MLP -> logit.

mode = "cnn"      -> experiment A (CNN only)
mode = "physics"  -> experiment B (physics features only)
mode = "hybrid"   -> experiment C (late fusion), with auxiliary per-branch heads so neither
                     branch is ignored during training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.cnn_branch import CNNBranch


class HybridModel(nn.Module):
    def __init__(self, n_phys: int, mode: str = "hybrid", cnn_name: str = "convnext_tiny",
                 pretrained: bool = False, embed_dim: int = 256, phys_dim: int = 64, drop: float = 0.1):
        super().__init__()
        assert mode in ("cnn", "physics", "hybrid")
        self.mode = mode
        self.cnn = CNNBranch(cnn_name, pretrained, embed_dim=embed_dim, drop_rate=drop) if mode != "physics" else None
        if mode != "cnn":
            self.phys = nn.Sequential(nn.Linear(n_phys, phys_dim), nn.GELU(), nn.Linear(phys_dim, phys_dim), nn.GELU())
            self.phys_head = nn.Linear(phys_dim, 1)
        self.register_buffer("phys_mean", torch.zeros(n_phys))
        self.register_buffer("phys_std", torch.ones(n_phys))
        if mode == "hybrid":
            self.fusion = nn.Sequential(nn.Linear(embed_dim + phys_dim, 128), nn.GELU(),
                                        nn.Dropout(drop), nn.Linear(128, 1))

    def set_phys_stats(self, feats: np.ndarray):
        self.phys_mean.copy_(torch.from_numpy(feats.mean(0)))
        self.phys_std.copy_(torch.from_numpy(feats.std(0) + 1e-6))

    def _norm(self, phys):
        phys = torch.nan_to_num(phys)
        return ((phys - self.phys_mean) / self.phys_std).clamp(-10, 10)

    def forward(self, image: torch.Tensor, phys: torch.Tensor) -> dict:
        out = {}
        if self.mode == "cnn":
            _, out["logit"] = self.cnn(image)
            return out
        pe = self.phys(self._norm(phys))
        out["logit_phys"] = self.phys_head(pe).squeeze(1)
        if self.mode == "physics":
            out["logit"] = out["logit_phys"]
            return out
        ce, out["logit_cnn"] = self.cnn(image)
        out["logit"] = self.fusion(torch.cat([ce, pe], dim=1)).squeeze(1)
        return out
