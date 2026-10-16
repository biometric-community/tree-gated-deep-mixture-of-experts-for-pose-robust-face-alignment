"""300W challenge loader (68-point .pts). Real images only — no synthetic corpora.

Paper Sec. 4.1.1 trains on LFPW+HELEN+AFW (3148). This monorepo packaging of
`projects/datasets/300w` provides the 600 Indoor/Outdoor challenge images with
68 landmarks; we split 80/20 for train/eval (see DEVIATIONS D1).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _read_pts(path: Path) -> np.ndarray:
    coords: List[Tuple[float, float]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("version") or line.startswith("n_points") or line in ("{", "}"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
    arr = np.asarray(coords, dtype=np.float32)
    if arr.shape != (68, 2):
        raise ValueError(f"expected 68 points in {path}, got {arr.shape}")
    return arr


def _estimate_pose_from_landmarks(pts: np.ndarray) -> np.ndarray:
    """Rough yaw/pitch/roll (radians) from 68 landmarks for pose supervision.

    Uses eye–mouth geometry; not the paper ResNet-50, but provides real
    landmark-derived labels as the paper notes pose can be inferred from landmarks.
    """
    left_eye = pts[36:42].mean(axis=0)
    right_eye = pts[42:48].mean(axis=0)
    nose = pts[30]
    mouth = pts[48:68].mean(axis=0)
    eye_mid = 0.5 * (left_eye + right_eye)
    interocular = np.linalg.norm(right_eye - left_eye) + 1e-6
    # yaw ~ horizontal nose offset; pitch ~ vertical nose vs eyes/mouth; roll ~ eye angle
    yaw = float(np.clip((nose[0] - eye_mid[0]) / interocular, -1.5, 1.5))
    pitch = float(
        np.clip(((mouth[1] - nose[1]) - (nose[1] - eye_mid[1])) / interocular, -1.5, 1.5)
    )
    roll = float(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    return np.asarray([yaw, pitch, roll], dtype=np.float32)


def discover_samples(root: Path) -> List[Tuple[Path, Path]]:
    root = Path(root)
    pairs: List[Tuple[Path, Path]] = []
    for pts in sorted(root.rglob("*.pts")):
        stem = pts.with_suffix("")
        img = None
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            cand = stem.with_suffix(ext)
            if cand.exists():
                img = cand
                break
        if img is not None:
            pairs.append((img, pts))
    return pairs


class Face300WDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path]],
        image_size: int = 150,
        augment: bool = False,
        mean_shape: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.mean_shape = mean_shape  # (136,) in pixel coords of resized image
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, img_path: Path, pts_path: Path):
        img = Image.open(img_path).convert("L")
        pts = _read_pts(pts_path)
        w0, h0 = img.size
        # Crop to landmark bbox + margin, then resize to image_size
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        bw, bh = xmax - xmin, ymax - ymin
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        side = 1.4 * max(bw, bh)
        x0 = int(max(0, cx - side / 2))
        y0 = int(max(0, cy - side / 2))
        x1 = int(min(w0, cx + side / 2))
        y1 = int(min(h0, cy + side / 2))
        img = img.crop((x0, y0, x1, y1))
        pts = pts - np.array([x0, y0], dtype=np.float32)
        cw, ch = img.size
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        sx = self.image_size / max(cw, 1)
        sy = self.image_size / max(ch, 1)
        pts = pts * np.array([sx, sy], dtype=np.float32)
        return img, pts

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, pts_path = self.samples[idx]
        img, pts = self._load(img_path, pts_path)

        # Augmentation (Sec. 3.5): translate, scale, horizontal flip
        scale = 1.0
        tx = ty = 0.0
        flip = False
        if self.augment:
            scale = float(self.rng.normal(1.0, 0.1))
            tx = float(self.rng.normal(0.0, 10.0))
            ty = float(self.rng.normal(0.0, 10.0))
            flip = bool(self.rng.rand() < 0.5)

        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        if flip:
            arr = arr[:, ::-1].copy()
            pts = pts.copy()
            pts[:, 0] = self.image_size - 1 - pts[:, 0]
            # Swap left/right landmark indices (standard 68-point mirroring)
            pts = _mirror_68(pts)

        pose = _estimate_pose_from_landmarks(pts)
        gt = pts.reshape(-1)

        if self.mean_shape is None:
            init = gt.copy()
        else:
            init = self.mean_shape.copy()
            # Apply same geometric jitter to init (paper: augment initial mean shape)
            cx = self.image_size / 2.0
            cy = self.image_size / 2.0
            init = init.reshape(-1, 2)
            init = (init - np.array([cx, cy])) * scale + np.array([cx + tx, cy + ty])
            if flip:
                init[:, 0] = self.image_size - 1 - init[:, 0]
                init = _mirror_68(init)
            init = init.reshape(-1)

        return {
            "image": torch.from_numpy(arr[None, ...]),  # (1,H,W)
            "landmarks": torch.from_numpy(gt.astype(np.float32)),
            "init_shape": torch.from_numpy(init.astype(np.float32)),
            "pose": torch.from_numpy(pose),
            "interocular": torch.tensor(
                float(np.linalg.norm(pts[42:48].mean(0) - pts[36:42].mean(0)) + 1e-6),
                dtype=torch.float32,
            ),
            "bbox_norm": torch.tensor(
                float(math.sqrt(max((pts[:, 0].max() - pts[:, 0].min()) *
                                    (pts[:, 1].max() - pts[:, 1].min()), 1.0))),
                dtype=torch.float32,
            ),
            "path": str(img_path),
        }


def _mirror_68(pts: np.ndarray) -> np.ndarray:
    """Mirror 68-landmark topology (iBUG indexing)."""
    # Pair swaps for facial symmetry
    pairs = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (31, 35), (32, 34),
        (48, 54), (49, 53), (50, 52), (60, 64), (61, 63),
        (67, 65), (59, 55), (58, 56),
    ]
    out = pts.copy()
    for a, b in pairs:
        out[[a, b]] = out[[b, a]]
    return out


def compute_mean_shape(samples: List[Tuple[Path, Path]], image_size: int = 150) -> np.ndarray:
    ds = Face300WDataset(samples, image_size=image_size, augment=False, mean_shape=None)
    acc = np.zeros(136, dtype=np.float64)
    for i in range(len(ds)):
        acc += ds[i]["landmarks"].numpy()
    return (acc / max(len(ds), 1)).astype(np.float32)


def build_dataloaders(cfg: dict, project_root: Path):
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    if not root.is_absolute():
        root = (project_root / root).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root missing: {root}. Register 300W under projects/datasets/300w."
        )
    pairs = discover_samples(root)
    if not pairs:
        raise FileNotFoundError(f"No image/.pts pairs under {root}")

    rng = np.random.RandomState(int(cfg.get("train", {}).get("seed", 42)))
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = max(1, int(len(pairs) * float(data_cfg.get("val_fraction", 0.2))))
    val_idx = set(idx[:n_val].tolist())
    train_pairs = [pairs[i] for i in idx if i not in val_idx]
    val_pairs = [pairs[i] for i in idx if i in val_idx]

    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")
    if max_train is not None:
        train_pairs = train_pairs[: int(max_train)]
    if max_val is not None:
        val_pairs = val_pairs[: int(max_val)]

    image_size = int(data_cfg.get("image_size", 150))
    mean_shape = compute_mean_shape(train_pairs, image_size=image_size)

    train_ds = Face300WDataset(
        train_pairs,
        image_size=image_size,
        augment=True,
        mean_shape=mean_shape,
        seed=int(cfg.get("train", {}).get("seed", 42)),
    )
    val_ds = Face300WDataset(
        val_pairs,
        image_size=image_size,
        augment=False,
        mean_shape=mean_shape,
        seed=0,
    )

    bs = int(cfg.get("train", {}).get("batch_size", 8))
    nw = int(cfg.get("train", {}).get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False
    )
    meta = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_total": len(pairs),
        "root": str(root),
        "mean_shape": mean_shape,
    }
    return train_loader, val_loader, meta
