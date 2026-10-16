# Source code mapping

## Official / upstream

| Item | Status |
|------|--------|
| Public training code | **Not found.** Author GitHub (`estephe-arnaud/ademfa`) lists the paper but does not ship this TBIOM model. |
| arXiv PDF | https://arxiv.org/abs/1910.09450 |
| IEEE | https://doi.org/10.1109/TBIOM.2019.2950032 |

## Mapping (paper → `tgmoe`)

| Paper component | Module |
|-----------------|--------|
| Sec. 3.1 pose estimator Ω | `tgmoe/models/pose.py` (`PoseEstimator`) |
| Sec. 3.2 representation CNN / experts | `tgmoe/models/experts.py` (`SharedPatchCNN`) |
| Sec. 3.3 regression experts | `tgmoe/models/experts.py` (`ExpertRegressor`, `BaselineRegressor`) |
| Sec. 3.4 Softmax / Tree gates | `tgmoe/models/gates.py` |
| Sec. 3.5 cascade / Pose-Tree-MoE | `tgmoe/models/cascade.py` |
| Sec. 4.1 NME | `tgmoe/metrics.py` |
| Data (300W substitute) | `tgmoe/data/dataset_300w.py` |

Conflict rule: PDF / `analysis.md` overrides any informal upstream notes.
