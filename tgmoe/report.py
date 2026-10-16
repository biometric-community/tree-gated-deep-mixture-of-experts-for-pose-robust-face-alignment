"""Generate REPORT.md + SVG figures (CED, train curves) from real logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tgmoe.plot_style import (
    LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS,
    apply_rcparams,
    apply_style,
)

PRIMARY_BLUE = MULTI_SERIES_COLORS[0]
PRIMARY_RED = MULTI_SERIES_COLORS[1]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_train_loss(summary: Dict[str, Any], figdir: Path) -> Optional[Path]:
    hist = summary.get("history") or {}
    epochs = hist.get("epochs") or []
    loss = hist.get("loss") or []
    if not epochs or not loss:
        return None
    apply_rcparams()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, loss, color=PRIMARY_BLUE, linewidth=LINEWIDTH_MAIN, label="Train loss")
    if hist.get("val_nme"):
        ax2 = ax.twinx()
        ax2.plot(
            epochs,
            hist["val_nme"],
            color=PRIMARY_RED,
            linewidth=LINEWIDTH_MAIN,
            label="Val NME (%)",
        )
        ax2.set_ylabel("Val NME (%)")
        ax2.tick_params(axis="y", labelsize=10)
    apply_style(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Pose-Tree-MoE / Tree-MoE training (ours, 300W)")
    ax.legend(frameon=False, loc="upper left")
    out = figdir / "train_loss.svg"
    save_figure(fig, out)
    return out


def plot_ced(eval_m: Dict[str, Any], figdir: Path) -> Optional[Path]:
    splits = eval_m.get("splits") or {}
    if not splits:
        return None
    apply_rcparams()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = MULTI_SERIES_COLORS
    for i, (name, data) in enumerate(splits.items()):
        ced = data.get("ced") or {}
        thr = ced.get("thresholds") or []
        frac = ced.get("ced_percent") or []
        if not thr:
            continue
        color = colors[i % len(colors)]
        label = f"{name} (NME={data.get('mean_nme', float('nan')):.2f}%)"
        ax.plot(thr, frac, color=color, linewidth=LINEWIDTH_MAIN, label=label)
    apply_style(ax)
    ax.set_xlabel("NME threshold (%)")
    ax.set_ylabel("Images ≤ threshold (%)")
    ax.set_title("CED — 300W inter-pupil NME (ours)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    out = figdir / "ced.svg"
    save_figure(fig, out)
    return out


def write_report(
    summary: Optional[Dict[str, Any]],
    eval_m: Optional[Dict[str, Any]],
    size_info: Optional[Dict[str, Any]],
    figs: Dict[str, Optional[Path]],
    out_path: Path,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        "Rebuilt from Arnaud, Dapogny & Bailly, *Tree-gated Deep Mixture-of-Experts "
        "For Pose-robust Face Alignment* (TBIOM 2020 / arXiv:1910.09450).",
        "",
        "## Setup",
        "",
    ]
    if summary:
        lines += [
            f"- Variant: `{summary.get('variant')}`",
            f"- Parameters: {summary.get('n_params_m', float('nan')):.2f}M",
            f"- Train / val images: {summary.get('n_train')} / {summary.get('n_val')}",
            f"- Data root: `{summary.get('data_root')}`",
            f"- Epochs / steps: {summary.get('epochs_ran')} / {summary.get('steps')}",
            f"- Device: `{summary.get('device')}`",
            f"- Best val NME (inter-pupil %): **{summary.get('best_val_nme_percent', float('nan')):.3f}**",
            "",
        ]
    if size_info:
        lines += [
            "## Dataset size gate",
            "",
            f"- Total bytes: {size_info.get('total_bytes')} "
            f"({size_info.get('total_gib')} GiB)",
            f"- Decision: `{size_info.get('decision')}`",
            "",
        ]
    lines += ["## Metrics", ""]
    if eval_m and eval_m.get("splits"):
        lines += [
            "| Split | N images | Mean NME (%) | Median NME (%) |",
            "|-------|----------|--------------|----------------|",
        ]
        for name, data in eval_m["splits"].items():
            lines.append(
                f"| {name} | {data.get('n_images')} | "
                f"{data.get('mean_nme', float('nan')):.3f} | "
                f"{data.get('median_nme', float('nan')):.3f} |"
            )
        lines.append("")
    else:
        lines += [
            "BLOCKED or not yet run: no `outputs/logs/eval_results.json`.",
            "",
        ]

    lines += ["## Figures", ""]
    if figs.get("train_loss"):
        lines += [
            "### Train loss / val NME",
            "",
            "![Train loss](outputs/figures/train_loss.svg)",
            "",
            "*Ours — not a copy of a paper figure.*",
            "",
        ]
    if figs.get("ced"):
        lines += [
            "### Cumulative error distribution (CED)",
            "",
            "![CED](outputs/figures/ced.svg)",
            "",
            "*Ours on monorepo 300W Indoor/Outdoor split (see DEVIATIONS D1). "
            "Paper Table 1 uses LFPW+HELEN+AFW / IBUG protocol.*",
            "",
        ]

    lines += [
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/setup_env.sh",
        "bash scripts/train.sh configs/full.yaml",
        "bash scripts/eval.sh configs/full.yaml",
        "bash scripts/report.sh configs/full.yaml",
        "```",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Report figures + REPORT.md")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--logs", default=None)
    parser.add_argument("--figdir", default=None)
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_dir = Path(args.logs or cfg["paths"]["log_dir"])
    figdir = Path(args.figdir or cfg["paths"]["figure_dir"])
    figdir.mkdir(parents=True, exist_ok=True)

    summary = load_json(log_dir / "train_summary.json")
    eval_m = load_json(log_dir / "eval_results.json")
    size_info = load_json(log_dir / "dataset_size.json")

    if summary is None and eval_m is None:
        report = _ROOT / "REPORT.md"
        report.write_text(
            "# REPORT\n\nBLOCKED: real dataset metrics required. "
            "Run train/eval on `projects/datasets/300w` first.\n"
        )
        print("No metrics JSON; wrote BLOCKED REPORT.md")
        return

    figs = {
        "train_loss": plot_train_loss(summary, figdir) if summary else None,
        "ced": plot_ced(eval_m, figdir) if eval_m else None,
    }
    title = (cfg.get("report") or {}).get("title") or cfg.get("paper", {}).get(
        "title", "Tree-gated MoE report"
    )
    write_report(summary, eval_m, size_info, figs, _ROOT / "REPORT.md", title)


if __name__ == "__main__":
    main()
