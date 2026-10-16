"""Train Pose-Tree-MoE / Tree-MoE cascade on real 300W landmarks (Sec. 3.5)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tgmoe.data.dataset_300w import build_dataloaders
from tgmoe.losses import landmark_l2, total_loss
from tgmoe.metrics import nme_interpupil
from tgmoe.models.cascade import PoseTreeMoE


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@torch.no_grad()
def eval_nme(model: PoseTreeMoE, loader, device: torch.device) -> float:
    model.eval()
    vals = []
    for batch in loader:
        images = batch["image"].to(device)
        init = batch["init_shape"].to(device)
        gt = batch["landmarks"].to(device)
        iod = batch["interocular"].to(device)
        pred, _, _ = model(images, init)
        nme = nme_interpupil(pred, gt, iod)
        vals.extend((nme * 100.0).detach().cpu().tolist())
    model.train()
    return float(sum(vals) / max(len(vals), 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Tree-gated MoE face alignment")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    seed = int(cfg.get("train", {}).get("seed", 42))
    torch.manual_seed(seed)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, meta = build_dataloaders(cfg, _ROOT)

    mcfg = cfg["model"]
    model = PoseTreeMoE(
        n_stages=int(mcfg.get("n_stages", 4)),
        n_landmarks=int(mcfg.get("n_landmarks", 68)),
        variant=str(mcfg.get("variant", "pose_tree_moe")),
        patch_size=int(cfg["data"].get("patch_size", 32)),
    ).to(device)

    tcfg = cfg["train"]
    lr = float(tcfg.get("lr", 0.001))
    epochs = int(tcfg["epochs"])
    pose_w = float(tcfg.get("pose_loss_weight", 0.1))
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")
    log_every = int(tcfg.get("log_every", 10))

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"variant={mcfg.get('variant')} params={n_params:.2f}M device={device} "
        f"n_train={meta['n_train']} n_val={meta['n_val']}"
    )

    history = {
        "epochs": [],
        "loss": [],
        "landmark_l2": [],
        "pose_l2": [],
        "val_nme": [],
        "lr": [],
        "steps": [],
    }
    global_step = 0
    t0 = time.time()
    best_nme = float("inf")
    mean_shape = meta["mean_shape"]

    for epoch in range(epochs):
        model.train()
        running = {"loss": 0.0, "lm": 0.0, "pose": 0.0, "n": 0}
        stop = False
        for batch in train_loader:
            images = batch["image"].to(device)
            init = batch["init_shape"].to(device)
            gt = batch["landmarks"].to(device)
            gt_pose = batch["pose"].to(device)

            pred, stage_shapes, pred_pose = model(images, init)
            # Supervise every cascade stage (coarse-to-fine residual learning).
            lm = sum(landmark_l2(s, gt) for s in stage_shapes) / max(len(stage_shapes), 1)
            losses = total_loss(pred, gt, pred_pose, gt_pose, pose_weight=pose_w)
            pose_term = losses.get("pose_l2")
            if pose_term is not None and pose_w > 0:
                loss = lm + float(pose_w) * pose_term
            else:
                loss = lm
            losses["landmark_l2"] = lm
            losses["loss"] = loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running["loss"] += float(loss.item())
            running["lm"] += float(lm.item())
            running["pose"] += float(pose_term.item()) if pose_term is not None else 0.0
            running["n"] += 1
            global_step += 1

            if global_step % log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} "
                    f"lm={lm.item():.4f}"
                )

            if max_steps is not None and global_step >= int(max_steps):
                stop = True
                break

        n = max(running["n"], 1)
        val_nme = eval_nme(model, val_loader, device)
        history["epochs"].append(epoch)
        history["loss"].append(running["loss"] / n)
        history["landmark_l2"].append(running["lm"] / n)
        history["pose_l2"].append(running["pose"] / n)
        history["val_nme"].append(val_nme)
        history["lr"].append(lr)
        history["steps"].append(global_step)
        print(f"epoch={epoch} val_nme={val_nme:.3f}%")

        ckpt = {
            "model": model.state_dict(),
            "cfg": cfg,
            "mean_shape": mean_shape.tolist(),
            "epoch": epoch,
            "val_nme": val_nme,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_nme < best_nme:
            best_nme = val_nme
            torch.save(ckpt, Path(cfg["paths"].get("checkpoint", ckpt_dir / "best.pt")))
            torch.save(ckpt, ckpt_dir / "best.pt")

        if stop:
            break

    elapsed = time.time() - t0
    summary = {
        "variant": mcfg.get("variant"),
        "n_params_m": n_params,
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "data_root": meta["root"],
        "best_val_nme_percent": best_nme,
        "epochs_ran": history["epochs"][-1] + 1 if history["epochs"] else 0,
        "steps": global_step,
        "elapsed_sec": elapsed,
        "device": str(device),
        "history": history,
        "config": args.config,
    }
    out = log_dir / "train_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {out} best_val_nme={best_nme:.3f}% elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
