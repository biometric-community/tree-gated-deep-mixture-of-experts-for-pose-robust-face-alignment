"""Run landmark prediction on a folder of images (or 300W .pts pairs)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tgmoe.data.dataset_300w import Face300WDataset, discover_samples
from tgmoe.models.cascade import PoseTreeMoE


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict landmarks with Tree-gated MoE")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input", default=None, help="Dataset root with image/.pts pairs")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    pred_dir = Path(cfg["paths"]["pred_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint or cfg["paths"].get("checkpoint", "outputs/checkpoints/best.pt"))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mean_shape = np.asarray(ckpt.get("mean_shape"), dtype=np.float32)

    root = Path(args.input) if args.input else Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = (_ROOT / root).resolve()
    pairs = discover_samples(root)[: int(args.limit)]
    if not pairs:
        raise FileNotFoundError(f"No image/.pts pairs under {root}")

    mcfg = cfg["model"]
    model = PoseTreeMoE(
        n_stages=int(mcfg.get("n_stages", 4)),
        n_landmarks=int(mcfg.get("n_landmarks", 68)),
        variant=str(mcfg.get("variant", "pose_tree_moe")),
        patch_size=int(cfg["data"].get("patch_size", 32)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = Face300WDataset(
        pairs,
        image_size=int(cfg["data"].get("image_size", 150)),
        augment=False,
        mean_shape=mean_shape,
    )
    records = []
    for i in range(len(ds)):
        item = ds[i]
        images = item["image"].unsqueeze(0).to(device)
        init = item["init_shape"].unsqueeze(0).to(device)
        pred, stage_shapes, pose = model(images, init)
        rec = {
            "path": item["path"],
            "landmarks": pred.squeeze(0).cpu().numpy().reshape(-1, 2).tolist(),
            "pose_yaw_pitch_roll": pose.squeeze(0).cpu().numpy().tolist(),
            "n_stages": len(stage_shapes),
        }
        records.append(rec)
        stem = Path(item["path"]).stem
        np.savetxt(pred_dir / f"{stem}.pts.txt", pred.squeeze(0).cpu().numpy().reshape(-1, 2))

    out = pred_dir / "predictions.json"
    with open(out, "w") as f:
        json.dump({"n": len(records), "items": records}, f, indent=2)
        f.write("\n")
    print(f"Wrote {out} ({len(records)} samples)")


if __name__ == "__main__":
    main()
