"""Alignment metrics (Sec. 4.1): inter-pupil NME for 2D face alignment (Eq. 12)."""

from __future__ import annotations

import torch


def nme_interpupil(
    pred: torch.Tensor,
    gt: torch.Tensor,
    interocular: torch.Tensor,
) -> torch.Tensor:
    """Normalized mean error per sample; pred/gt are (B, 2P)."""
    # mean L2 over landmarks
    b = pred.shape[0]
    p = pred.reshape(b, -1, 2)
    g = gt.reshape(b, -1, 2)
    dist = torch.linalg.norm(p - g, dim=-1).mean(dim=-1)
    return dist / interocular.clamp_min(1e-6)


def nme_bbox(
    pred: torch.Tensor,
    gt: torch.Tensor,
    bbox_norm: torch.Tensor,
) -> torch.Tensor:
    """Bounding-box normalized NME used for 3D protocol (Eq. 13)."""
    b = pred.shape[0]
    p = pred.reshape(b, -1, 2)
    g = gt.reshape(b, -1, 2)
    dist = torch.linalg.norm(p - g, dim=-1).mean(dim=-1)
    return dist / bbox_norm.clamp_min(1e-6)
