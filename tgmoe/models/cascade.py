"""Pose-Tree-MoE cascade stage and multi-stage aligner (Secs. 3.2–3.5, Fig. 1)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import (
    BaselineRegressor,
    SharedPatchCNN,
    make_expert_cnns,
    make_expert_regressors,
)
from .gates import SoftmaxGate, TreeGate
from .pose import PoseEstimator


def extract_patches(
    images: torch.Tensor,
    shapes: torch.Tensor,
    patch_size: int = 32,
) -> torch.Tensor:
    """Extract shape-indexed grayscale patches (Sec. 3.2).

    images: (B, 1, H, W) in [-1, 1]
    shapes: (B, 2P) absolute pixel coords (x0,y0,x1,y1,...)
    returns: (B, P, 1, patch, patch)
    """
    b, _, h, w = images.shape
    p = shapes.shape[1] // 2
    half = patch_size / 2.0
    # grid_sample expects align_corners coords in [-1, 1]
    ys = torch.linspace(-half + 0.5, half - 0.5, patch_size, device=images.device)
    xs = torch.linspace(-half + 0.5, half - 0.5, patch_size, device=images.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    # (patch, patch)
    patches = []
    for i in range(p):
        cx = shapes[:, 2 * i]
        cy = shapes[:, 2 * i + 1]
        # absolute sampling coords → normalized
        abs_x = cx[:, None, None] + gx
        abs_y = cy[:, None, None] + gy
        nx = 2.0 * abs_x / max(w - 1, 1) - 1.0
        ny = 2.0 * abs_y / max(h - 1, 1) - 1.0
        grid = torch.stack([nx, ny], dim=-1)  # (B, ph, pw, 2)
        patch = F.grid_sample(
            images, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        patches.append(patch)
    return torch.stack(patches, dim=1)  # (B, P, 1, ph, pw)


class CascadeStage(nn.Module):
    """One cascade stage: pose-gated expert CNNs + gated expert regressors."""

    def __init__(
        self,
        n_landmarks: int = 68,
        feat_per_lm: int = 30,
        n_rep_experts: int = 8,
        n_reg_experts: int = 64,
        rep_gate: str = "tree",  # tree | softmax | none
        reg_gate: str = "tree",
        rep_channels: Tuple[int, ...] = (7, 14, 28, 56, 30),
        patch_size: int = 32,
        use_pose_gate: bool = True,
    ):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.feat_per_lm = feat_per_lm
        self.patch_size = patch_size
        self.use_pose_gate = use_pose_gate
        self.feat_dim = n_landmarks * feat_per_lm
        self.out_dim = n_landmarks * 2
        self.n_rep_experts = n_rep_experts
        self.n_reg_experts = n_reg_experts
        self.rep_gate_kind = rep_gate
        self.reg_gate_kind = reg_gate

        if n_rep_experts == 1:
            self.rep_experts = nn.ModuleList(
                [SharedPatchCNN(channels=(20, 40, 80, 160, 30))]
            )
        else:
            self.rep_experts = make_expert_cnns(n_rep_experts, rep_channels)

        if rep_gate == "tree" and n_rep_experts > 1:
            depth = int(n_rep_experts).bit_length() - 1
            assert 1 << depth == n_rep_experts, "n_rep_experts must be power of 2"
            self.rep_gate = TreeGate(3 if use_pose_gate else self.feat_dim, depth)
        elif rep_gate == "softmax" and n_rep_experts > 1:
            self.rep_gate = SoftmaxGate(3 if use_pose_gate else self.feat_dim, n_rep_experts)
        else:
            self.rep_gate = None

        if n_reg_experts == 1:
            self.reg_experts = nn.ModuleList(
                [BaselineRegressor(self.feat_dim, self.out_dim)]
            )
            self.reg_gate = None
        else:
            self.reg_experts = make_expert_regressors(
                n_reg_experts, self.feat_dim, self.out_dim, hidden=128
            )
            if reg_gate == "tree":
                depth = int(n_reg_experts).bit_length() - 1
                assert 1 << depth == n_reg_experts
                self.reg_gate = TreeGate(self.feat_dim, depth)
            elif reg_gate == "softmax":
                self.reg_gate = SoftmaxGate(self.feat_dim, n_reg_experts)
            else:
                self.reg_gate = None  # uniform MoE average

    def forward(
        self,
        images: torch.Tensor,
        shapes: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        patches = extract_patches(images, shapes, self.patch_size)
        # Expert CNN features → H columns (Eq. 3)
        feats = [cnn(patches) for cnn in self.rep_experts]  # list of (B, Pn)
        H = torch.stack(feats, dim=-1)  # (B, Pn, L)

        extras: Dict[str, torch.Tensor] = {}
        if self.rep_gate is None or self.n_rep_experts == 1:
            g = H.new_ones(H.shape[0], self.n_rep_experts) / self.n_rep_experts
        else:
            gate_in = pose if (self.use_pose_gate and pose is not None) else H.mean(dim=-1)
            g = self.rep_gate(gate_in)
        extras["rep_gate"] = g
        h = torch.einsum("bpl,bl->bp", H, g)  # Eq. 4

        # Expert regressors (Eqs. 5–7)
        deltas = torch.stack([r(h) for r in self.reg_experts], dim=-1)  # (B, 2P, L')
        if self.reg_gate is None:
            gp = deltas.new_ones(deltas.shape[0], self.n_reg_experts) / self.n_reg_experts
        else:
            gp = self.reg_gate(h)
        extras["reg_gate"] = gp
        delta = torch.einsum("bpl,bl->bp", deltas, gp)
        return delta, extras


class PoseTreeMoE(nn.Module):
    """Full cascaded Pose-Tree-MoE (Fig. 1, Sec. 3.5)."""

    def __init__(
        self,
        n_stages: int = 4,
        n_landmarks: int = 68,
        variant: str = "pose_tree_moe",
        patch_size: int = 32,
        n_rep_experts: Optional[int] = None,
        n_reg_experts: Optional[int] = None,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.n_landmarks = n_landmarks
        self.variant = variant
        self.pose_net = PoseEstimator()

        # Variant presets matching Sec. 3.5 / Table 1
        presets = {
            "baseline": dict(
                n_rep=1, n_reg=1, rep_gate="none", reg_gate="none", pose=False,
                ch=(20, 40, 80, 160, 30),
            ),
            "moe": dict(
                n_rep=1, n_reg=64, rep_gate="none", reg_gate="none", pose=False,
                ch=(20, 40, 80, 160, 30),
            ),
            "softmax_moe": dict(
                n_rep=1, n_reg=64, rep_gate="none", reg_gate="softmax", pose=False,
                ch=(20, 40, 80, 160, 30),
            ),
            "tree_moe": dict(
                n_rep=1, n_reg=64, rep_gate="none", reg_gate="tree", pose=False,
                ch=(20, 40, 80, 160, 30),
            ),
            "pose_softmax_moe": dict(
                n_rep=8, n_reg=64, rep_gate="softmax", reg_gate="tree", pose=True,
                ch=(7, 14, 28, 56, 30),
            ),
            "pose_tree_moe": dict(
                n_rep=8, n_reg=64, rep_gate="tree", reg_gate="tree", pose=True,
                ch=(7, 14, 28, 56, 30),
            ),
        }
        if variant not in presets:
            raise ValueError(f"unknown variant {variant}; choose from {list(presets)}")
        cfg = presets[variant]
        if n_rep_experts is not None:
            cfg["n_rep"] = n_rep_experts
        if n_reg_experts is not None:
            cfg["n_reg"] = n_reg_experts

        self.stages = nn.ModuleList(
            [
                CascadeStage(
                    n_landmarks=n_landmarks,
                    n_rep_experts=cfg["n_rep"],
                    n_reg_experts=cfg["n_reg"],
                    rep_gate=cfg["rep_gate"],
                    reg_gate=cfg["reg_gate"],
                    rep_channels=cfg["ch"],
                    patch_size=patch_size,
                    use_pose_gate=cfg["pose"],
                )
                for _ in range(n_stages)
            ]
        )

    def forward(
        self,
        images: torch.Tensor,
        init_shape: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Returns final shape, per-stage shapes, and pose estimate.
        """
        if pose is None:
            pose = self.pose_net(images)
        shape = init_shape
        stage_shapes = []
        for stage in self.stages:
            delta, _ = stage(images, shape, pose=pose)
            shape = shape + delta
            stage_shapes.append(shape)
        return shape, stage_shapes, pose


# Alias used in docs / configs
CascadeAligner = PoseTreeMoE
