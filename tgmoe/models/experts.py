"""Expert CNN representation and FC regression heads (Secs. 3.2–3.3, 3.5)."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPatchCNN(nn.Module):
    """Strided CNN applied to each landmark patch with shared kernels (Fig. 4).

    Single-expert (baseline) channels: 20@5x5 → 40@5x5 → 80@3x3 → 160@3x3 → 30@1x1
    Pose-Tree expert channels:         7@5x5 → 14@5x5 → 28@3x3 → 56@3x3 → 30@1x1
    """

    def __init__(self, channels: Sequence[int] = (20, 40, 80, 160, 30)):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.out_channels = c4

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, P, 1, H, W) → (B, P * C) flattened features."""
        b, p, c, h, w = patches.shape
        x = patches.reshape(b * p, c, h, w)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B*P, C)
        return x.reshape(b, p * self.out_channels)


class ExpertRegressor(nn.Module):
    """Small FC expert: Pn × 128 → 128 × 2P with ReLU (Eq. 6)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        # Residual cascade: start near identity (zero displacement).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class BaselineRegressor(nn.Module):
    """Single large FC: Pn × 8192 → 8192 × 2P (Sec. 3.5 Baseline)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 8192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def make_expert_cnns(
    n_experts: int,
    channels: Sequence[int],
) -> nn.ModuleList:
    return nn.ModuleList([SharedPatchCNN(channels) for _ in range(n_experts)])


def make_expert_regressors(
    n_experts: int,
    in_dim: int,
    out_dim: int,
    hidden: int = 128,
) -> nn.ModuleList:
    return nn.ModuleList(
        [ExpertRegressor(in_dim, out_dim, hidden=hidden) for _ in range(n_experts)]
    )
