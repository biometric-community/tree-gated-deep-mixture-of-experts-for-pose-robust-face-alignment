# Tree-gated Deep Mixture-of-Experts for Pose-robust Face Alignment

Rebuilt from Arnaud, Dapogny & Bailly, *Tree-gated Deep Mixture-of-Experts For Pose-robust Face Alignment* (TBIOM 2020 / arXiv:1910.09450).

## Setup

- Variant: `pose_tree_moe`
- Parameters: 72.98M
- Train / val images: 480 / 120
- Data root: `/data1/public/tbiom/projects/datasets/300w/extracted/300W`
- Epochs / steps: 25 / 1500
- Device: `cuda:0`
- Best val NME (inter-pupil %): **21.486**

## Dataset size gate

- Total bytes: 4285214098 (3.9909 GiB)
- Decision: `full_train`

## Metrics

| Split | N images | Mean NME (%) | Median NME (%) |
|-------|----------|--------------|----------------|
| val | 120 | 21.486 | 19.979 |
| train | 480 | 26.697 | 22.893 |

## Figures

### Train loss / val NME

![Train loss](outputs/figures/train_loss.svg)

*Ours — not a copy of a paper figure.*

### Cumulative error distribution (CED)

![CED](outputs/figures/ced.svg)

*Ours on monorepo 300W Indoor/Outdoor split (see DEVIATIONS D1). Paper Table 1 uses LFPW+HELEN+AFW / IBUG protocol.*

## Reproduce

```bash
bash scripts/setup_env.sh
bash scripts/train.sh configs/full.yaml
bash scripts/eval.sh configs/full.yaml
bash scripts/report.sh configs/full.yaml
```
