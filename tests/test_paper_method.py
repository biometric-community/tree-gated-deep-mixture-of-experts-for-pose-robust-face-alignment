"""Gates, cascade presets and metrics, checked against Sec. 3 and Eqs. (8)-(13)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tgmoe.data.dataset_300w import _mirror_68
from tgmoe.metrics import nme_bbox, nme_interpupil
from tgmoe.models.cascade import CascadeStage, PoseTreeMoE, extract_patches
from tgmoe.models.gates import SoftmaxGate, TreeGate


def eq_10(d: torch.Tensor, depth: int) -> torch.Tensor:
    """Leaf probabilities as products of left (d) and right (1 - d) decisions."""
    leaves = []
    for leaf in range(1 << depth):
        p, node = torch.ones(d.shape[0]), 0
        for level in reversed(range(depth)):
            go_right = (leaf >> level) & 1
            p = p * ((1 - d[:, node]) if go_right else d[:, node])
            node = 2 * node + 1 + go_right
        leaves.append(p)
    return torch.stack(leaves, dim=1)


def test_tree_gate_follows_eq_10():
    torch.manual_seed(0)
    gate = TreeGate(5, depth=3)
    x = torch.randn(4, 5)
    assert torch.allclose(gate(x), eq_10(torch.sigmoid(gate.routers(x)), 3), atol=1e-6)


@pytest.mark.parametrize("gate", [TreeGate(5, depth=6), SoftmaxGate(5, 64)])
def test_gates_are_distributions_over_64_experts(gate):
    g = gate(torch.randn(4, 5))
    assert g.shape == (4, 64)
    assert torch.allclose(g.sum(dim=1), torch.ones(4), atol=1e-5)


def test_softmax_gate_follows_eq_8():
    gate = SoftmaxGate(5, 4)
    x = torch.randn(3, 5)
    assert torch.allclose(gate(x), F.softmax(x @ gate.fc.weight.t() + gate.fc.bias, dim=1), atol=1e-6)


@pytest.fixture(scope="module")
def pose_tree_moe():
    return PoseTreeMoE(n_stages=1, variant="pose_tree_moe")


def test_pose_tree_moe_has_the_expert_counts_of_sec_3_5(pose_tree_moe, targets):
    stage = pose_tree_moe.stages[0]
    assert len(stage.rep_experts) == targets["architecture"]["cnn_experts"]
    assert len(stage.reg_experts) == targets["architecture"]["regression_experts"]
    assert stage.reg_gate.depth == targets["architecture"]["tree_depth"]
    assert stage.feat_dim == targets["architecture"]["features"]


def test_representation_gate_takes_the_head_pose(pose_tree_moe):
    assert pose_tree_moe.stages[0].rep_gate.routers.in_features == 3


def test_expert_regressors_have_the_paper_hidden_size(pose_tree_moe, targets):
    assert pose_tree_moe.stages[0].reg_experts[0].net[0].out_features == targets["architecture"]["expert_hidden"]


def test_baseline_is_one_large_regressor(targets):
    stage = CascadeStage(n_rep_experts=1, n_reg_experts=1, rep_gate="none", reg_gate="none", use_pose_gate=False)
    assert stage.reg_experts[0].net[0].out_features == targets["architecture"]["baseline_hidden"]


def test_unweighted_moe_averages_its_experts():
    stage = CascadeStage(n_rep_experts=1, n_reg_experts=4, rep_gate="none", reg_gate="none", use_pose_gate=False)
    images = torch.zeros(1, 1, 150, 150)
    _, extras = stage(images, torch.full((1, 136), 75.0))
    assert torch.allclose(extras["reg_gate"], torch.full((1, 4), 0.25))


@pytest.mark.parametrize(
    "key,value",
    [("n_stages", "stages"), ("n_landmarks", "landmarks")],
)
def test_model_config_matches_the_paper(default_cfg, targets, key, value):
    assert default_cfg["model"][key] == targets["architecture"][value]


@pytest.mark.parametrize("key,value", [("image_size", "image_size"), ("patch_size", "patch_size")])
def test_data_config_matches_the_paper(default_cfg, targets, key, value):
    assert default_cfg["data"][key] == targets["architecture"][value]


def test_patches_are_centred_on_the_landmark():
    image = torch.linspace(-1, 1, 150).repeat(150, 1)[None, None]  # value depends on x
    patches = extract_patches(image, torch.tensor([[60.0, 75.0]]), patch_size=32)
    centre = patches[0, 0, 0, 16, 15:17].mean()
    assert centre.item() == pytest.approx(image[0, 0, 75, 60].item(), abs=0.02)


def test_nme_matches_eq_12():
    pred = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
    gt = torch.zeros(1, 4)
    assert nme_interpupil(pred, gt, torch.tensor([10.0])).item() == pytest.approx(0.25)


def test_bbox_nme_matches_eq_13():
    pred = torch.tensor([[3.0, 4.0]])
    assert nme_bbox(pred, torch.zeros(1, 2), torch.tensor([20.0])).item() == pytest.approx(0.25)


def test_mirroring_is_its_own_inverse_and_fixes_the_nose_bridge():
    pts = np.random.RandomState(0).normal(size=(68, 2))
    assert np.array_equal(_mirror_68(_mirror_68(pts)), pts)
    assert np.array_equal(_mirror_68(pts)[27:31], pts[27:31])
    assert np.array_equal(_mirror_68(pts)[36], pts[45])
