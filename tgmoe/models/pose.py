"""Head pose estimator (Sec. 3.1, Fig. 2).

Paper uses truncated ResNet-50 + three separate angle heads.
We ship a lightweight CNN backbone with the same three-head design so the
project trains without ImageNet weights; document as D2 in DEVIATIONS.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseEstimator(nn.Module):
    """Ω = Ω_γ(φ) || Ω_β(φ) || Ω_α(φ) (Eq. 1). Output yaw, pitch, roll in radians."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head_yaw = nn.Linear(embed_dim, 1)
        self.head_pitch = nn.Linear(embed_dim, 1)
        self.head_roll = nn.Linear(embed_dim, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, 1, H, W) in [-1, 1] → (B, 3) yaw/pitch/roll."""
        phi = self.backbone(image).flatten(1)
        yaw = self.head_yaw(phi)
        pitch = self.head_pitch(phi)
        roll = self.head_roll(phi)
        return torch.cat([yaw, pitch, roll], dim=-1)
