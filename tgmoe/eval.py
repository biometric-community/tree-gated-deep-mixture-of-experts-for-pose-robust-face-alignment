"""Evaluate Tree-gated MoE on 300W (inter-pupil NME, Sec. 4.1 / Eq. 12)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tgmoe.data.dataset_300w import build_dataloaders
from tgmoe.metrics import nme_interpupil
from tgmoe.models.cascade import PoseTreeMoE


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def ced_curve(nme_percent: np.ndarray, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 10.0, 51)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    frac = [(nme_percent <= t).mean() * 100.0 for t in thresholds]
    return {
        "thresholds": thresholds.tolist(),
        "ced_percent": [float(x) for x in frac],
    }


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_nme = []
    paths = []
    for batch in loader:
        images = batch["image"].to(device)
        init = batch["init_shape"].to(device)
        gt = batch["landmarks"].to(device)
        iod = batch["interocular"].to(device)
        pred, _, _ = model(images, init)
        nme = (nme_interpupil(pred, gt, iod) * 100.0).cpu().numpy()
        all_nme.extend(nme.tolist())
        paths.extend(batch["path"])
    arr = np.asarray(all_nme, dtype=np.float64)
    return {
        "n_images": int(len(arr)),
        "mean_nme": float(arr.mean()) if len(arr) else float("nan"),
        "std_nme": float(arr.std()) if len(arr) else float("nan"),
        "median_nme": float(np.median(arr)) if len(arr) else float("nan"),
        "ced": ced_curve(arr),
        "per_image_nme": all_nme,
        "paths": paths,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Eval Tree-gated MoE face alignment")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint or cfg["paths"].get("checkpoint", "outputs/checkpoints/best.pt"))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    train_loader, val_loader, meta = build_dataloaders(cfg, _ROOT)
    mcfg = cfg["model"]
    model = PoseTreeMoE(
        n_stages=int(mcfg.get("n_stages", 4)),
        n_landmarks=int(mcfg.get("n_landmarks", 68)),
        variant=str(mcfg.get("variant", "pose_tree_moe")),
        patch_size=int(cfg["data"].get("patch_size", 32)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    val_stats = run_eval(model, val_loader, device)
    train_stats = run_eval(model, train_loader, device)

    results = {
        "checkpoint": str(ckpt_path),
        "variant": mcfg.get("variant"),
        "data_root": meta["root"],
        "metric": "nme_interpupil_percent",
        "splits": {
            "val": {k: v for k, v in val_stats.items() if k not in ("per_image_nme", "paths")},
            "train": {k: v for k, v in train_stats.items() if k not in ("per_image_nme", "paths")},
        },
        "val_per_image_nme": val_stats["per_image_nme"],
        "config": args.config,
    }
    # Keep CED in splits
    results["splits"]["val"]["ced"] = val_stats["ced"]
    results["splits"]["train"]["ced"] = train_stats["ced"]

    out = log_dir / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(
        f"Wrote {out} val_nme={val_stats['mean_nme']:.3f}% "
        f"train_nme={train_stats['mean_nme']:.3f}% n_val={val_stats['n_images']}"
    )


if __name__ == "__main__":
    main()
