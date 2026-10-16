# Implementation plan — Tree-gated Deep MoE face alignment

## Paper

- **Title:** Tree-gated Deep Mixture-of-Experts For Pose-robust Face Alignment
- **Authors:** Arnaud, Dapogny, Bailly (TBIOM 2020; arXiv:1910.09450)
- **Slug:** `tree-gated-deep-mixture-of-experts-for-pose-robust-face-alignment`
- **Package:** `tgmoe`

## Method summary

1. Head-pose estimator → yaw/pitch/roll (Sec. 3.1).
2. Cascade of stages; each stage extracts 32×32 patches around landmarks.
3. Representation: single CNN or Pose-Tree-MoE (8 expert CNNs gated by pose tree).
4. Regression: Baseline / MoE / Softmax-MoE / Tree-MoE (64 FC experts).
5. Train with L2 on landmark coords; ADAM lr=0.001; 150×150 grayscale ∈ [−1,1]; 4 stages.

## Datasets (monorepo)

| Dataset | On disk | Role in paper | Our use |
|---------|---------|---------------|---------|
| `projects/datasets/300w` | ~4.0 GiB, 600 Indoor/Outdoor + .pts | Classic 2D train/test (paper: 3148 train) | **Primary** train/eval |
| `projects/datasets/aflw` | ~27 GiB | Related large-pose corpus | Not used (size / protocol) |
| `projects/datasets/aflw2000` | ~0.85 GiB images, **no landmark .mat** | Related | Images only — cannot align |
| `projects/datasets/aflw2000-3d` | empty stub | Paper 3D test set | **Missing** |
| 300W-LP (W-LP) | not present | Paper 3D train | **Missing** |

**Size gate (required roots for default protocol):** `300w` only → ~4.0 GiB **< 5 GiB** → full train required after smoke.

## Figure inventory (Results-style)

| Paper fig | Type | Our artifact |
|-----------|------|--------------|
| Table 1 architecture ablations (NME) | table | `REPORT.md` metrics table |
| CED / error curves (implied by NME) | CED line | `outputs/figures/ced.svg` |
| Train dynamics (ours) | loss + val NME | `outputs/figures/train_loss.svg` |
| Fig. 5–7 qualitative / gates | method viz | skipped (method cartoons) |

## Upstream

No official public training repository (author profile lists paper only). Reimplementation from PDF/arXiv.

## Work order

1. Models: gates, experts, pose, cascade — done.
2. 300W loader + augment — done.
3. train / eval / predict / report + scripts.
4. Smoke → size gate → full train → report.
5. Five fidelity passes + publish.
