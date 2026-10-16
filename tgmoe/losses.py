"""Training losses (Sec. 3.5): L2 landmark displacement + optional pose L2."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def landmark_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean L2 over landmark coordinates (paper: optimize L2 loss)."""
    return F.mse_loss(pred, gt)


def pose_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, gt)


def total_loss(
    pred_shape: torch.Tensor,
    gt_shape: torch.Tensor,
    pred_pose: Optional[torch.Tensor] = None,
    gt_pose: Optional[torch.Tensor] = None,
    pose_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    lm = landmark_l2(pred_shape, gt_shape)
    out: Dict[str, torch.Tensor] = {"landmark_l2": lm, "loss": lm}
    if pred_pose is not None and gt_pose is not None and pose_weight > 0:
        pl = pose_l2(pred_pose, gt_pose)
        out["pose_l2"] = pl
        out["loss"] = lm + float(pose_weight) * pl
    return out
