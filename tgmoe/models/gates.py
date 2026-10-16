"""Gating networks (Sec. 3.4): softmax gate and neural tree gate."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxGate(nn.Module):
    """Softmax gate g(x) = softmax(Wx + b) (Eq. 8)."""

    def __init__(self, in_dim: int, n_experts: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.fc(x), dim=-1)


class TreeGate(nn.Module):
    """Neural tree gate of depth D with L=2^D leaves (Eqs. 9–11).

    Each internal node n has a soft routing probability d_n(x)=σ(w_n·x+b_n).
    Leaf probability μ_l is the product of left/right decisions along the path.
    """

    def __init__(self, in_dim: int, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError("tree depth must be >= 1")
        self.depth = depth
        self.n_leaves = 1 << depth
        self.n_internal = self.n_leaves - 1
        self.routers = nn.Linear(in_dim, self.n_internal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # d: (B, n_internal) in (0, 1) — P(go left)
        d = torch.sigmoid(self.routers(x))
        batch = x.shape[0]
        # Build leaf probabilities bottom-up via path products
        # node indexing: heap order, root=0; left=2i+1, right=2i+2
        mu = x.new_ones(batch, 1)  # start at root with prob 1
        # Expand level by level
        for level in range(self.depth):
            n_nodes = 1 << level
            # routing decisions for nodes at this level
            start = n_nodes - 1
            end = start + n_nodes
            d_level = d[:, start:end]  # (B, n_nodes)
            left = mu * d_level
            right = mu * (1.0 - d_level)
            mu = torch.stack((left, right), dim=-1).reshape(batch, -1)
        return mu
