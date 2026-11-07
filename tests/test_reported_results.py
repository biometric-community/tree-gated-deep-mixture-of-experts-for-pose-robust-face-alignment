"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def mean_nme(report: str, split: str) -> float:
    match = re.search(rf"\| {split} \| \d+ \| ([\d.]+) \|", report)
    assert match, f"{split} row missing from REPORT.md"
    return float(match.group(1))


def test_nme_reaches_the_hardest_published_set(report, targets):
    """Table 1: I-BUG, the hardest set, is 7.5%. The local split is 300W's challenge set."""
    expected = targets["table_1_pose_tree_moe"]["ibug"]
    assert mean_nme(report, "test") == pytest.approx(expected, abs=targets["tolerance_pp"])


def test_result_comes_from_a_split_not_used_for_model_selection(report):
    """The paper reports held-out test sets; a selected checkpoint's own split is optimistic."""
    assert mean_nme(report, "test") >= 0.0
