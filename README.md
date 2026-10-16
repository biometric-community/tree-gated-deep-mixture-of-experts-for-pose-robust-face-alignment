# Tree-gated Deep Mixture-of-Experts for Pose-robust Face Alignment

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of **Pose-Tree-MoE / Tree-MoE** cascaded face alignment (Arnaud, Dapogny & Bailly, TBIOM 2020).

**Repository:** https://github.com/biometric-community/tree-gated-deep-mixture-of-experts-for-pose-robust-face-alignment

Monorepo path: `projects/papers/tree-gated-deep-mixture-of-experts-for-pose-robust-face-alignment/`

## Features

- Tree-structured and softmax gating over expert CNNs / FC regressors
- Cascaded Pose-Tree-MoE variants matching Sec. 3.5 presets
- Train / eval / predict / report on real **300W** images under `projects/datasets/300w`
- Experimental CED and train curves via vendored **dev-plot** style

## Setup

```bash
bash scripts/setup_env.sh
# uses shared monorepo venv at <repo>/.venv
```

## Data

Point configs at the monorepo dataset (default):

```text
../../datasets/300w/extracted/300W
```

See `DEVIATIONS.md` (D1, D4) for protocol differences vs the paper’s LFPW/HELEN/AFW and AFLW2000-3D setups.

## Train / eval / report / predict

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/smoke.sh
bash scripts/check_dataset_size.sh ../../datasets/300w
bash scripts/train_full.sh          # when size gate < 5 GiB
# or stepwise:
bash scripts/train.sh configs/full.yaml
bash scripts/eval.sh configs/full.yaml
bash scripts/predict.sh configs/full.yaml
bash scripts/report.sh configs/full.yaml
```

Results: `REPORT.md`, `outputs/logs/*.json`, `outputs/figures/*.svg`.

## Layout

```text
tgmoe/           # package (models, data, train/eval/predict/report)
configs/         # smoke.yaml, default.yaml, full.yaml
scripts/         # setup_env, train, eval, predict, report, smoke, train_full, check_dataset_size
```

## Fidelity

See `FIDELITY_AUDIT.md`, `IMPLEMENTATION_PLAN.md`, `SOURCE_CODE.md`, `DEVIATIONS.md`.

## Citation

```bibtex
@article{arnaud2020treegated,
  title={Tree-gated Deep Mixture-of-Experts For Pose-robust Face Alignment},
  author={Arnaud, Est{\`e}phe and Dapogny, Arnaud and Bailly, K{\'e}vin},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  volume={2},
  number={2},
  pages={122--132},
  year={2020}
}
```
