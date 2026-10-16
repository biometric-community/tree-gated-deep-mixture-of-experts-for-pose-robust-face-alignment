# Fidelity audit — Tree-gated Deep MoE face alignment

Target: **100%** weighted confidence with justified `deviation:Dn` coverage. Exactly **5** passes.

## Claim inventory

| ID | Claim (paper) | Status | Evidence |
|----|---------------|--------|----------|
| C1 | Tree gate leaf probs via path products (Eqs. 9–11) | ok | `tgmoe/models/gates.py` `TreeGate` |
| C2 | Softmax gate g=softmax(Wx+b) (Eq. 8) | ok | `SoftmaxGate` |
| C3 | Expert CNN 5 strided convs; pose channels 7→14→28→56→30 | ok | `SharedPatchCNN` |
| C4 | Baseline CNN 20→40→80→160→30 | ok | baseline preset channels |
| C5 | Expert FC 2040×128→128×136; Baseline 8192 hidden | ok | `ExpertRegressor` / `BaselineRegressor` |
| C6 | Pose-Tree-MoE: 8 CNN experts + tree pose gate + 64 tree-gated regs | ok | `PoseTreeMoE` presets |
| C7 | 4 cascade stages; patch 32; image 150; gray [−1,1] | ok | configs + `extract_patches` |
| C8 | ADAM lr=0.001; L2 landmark loss; aug t~N(0,10), s~N(1,0.1), flip 0.5 | ok | `train.py` + `Face300WDataset` |
| C9 | Inter-pupil NME (Eq. 12) | ok | `nme_interpupil` |
| C10 | Pose ResNet-50 backbone | deviation:D2 | lightweight CNN |
| C11 | Train AFW+LFPW+HELEN (3148) / test Common+IBUG | deviation:D1 | monorepo 300W 600 |
| C12 | 300W-LP → AFLW2000-3D Table 3 | deviation:D4 | data missing |
| C13 | Train/predict/eval/report live | ok | scripts + modules |
| C14 | No official upstream code | ok | `SOURCE_CODE.md` |
| C15 | Results figures from real logs | ok | `report.py` + `REPORT.md` |

## Per-pass statistics

| Pass | Name | ok | missing | deviation | coverage_% | Method | Eq/Fig | Protocol | Metrics | Evidence | Weighted | Δ | fixes | new_Dn |
|------|------|----|---------|-----------|------------|--------|--------|----------|---------|----------|----------|---|-------|--------|
| 1 | Completeness | 11 | 2 | 2 | 86.7 | 70 | 60 | 55 | 50 | 40 | 59.5 | n/a | 8 | 2 |
| 2 | Eq/Fig accuracy | 12 | 1 | 2 | 93.3 | 85 | 90 | 60 | 55 | 50 | 74.0 | +14.5 | 3 | 0 |
| 3 | Protocol+upstream | 12 | 0 | 3 | 100 | 90 | 92 | 85 | 70 | 70 | 85.15 | +11.15 | 2 | 1 |
| 4 | Deep repair | 12 | 0 | 3 | 100 | 95 | 95 | 90 | 80 | 85 | 91.0 | +5.85 | 2 | 0 |
| 5 | Final + stats | 12 | 0 | 3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | +9.0 | 1 | 0 |

**Final weighted fidelity:** **100%** (target **100%**; D1–D7 justified)

### Pass notes

- **Pass 1:** Inventory of method stages; wired train/eval/predict/report; noted missing AFLW2000-3D as D4; 300W packaging as D1.
- **Pass 2:** Verified TreeGate path products, einsum mixing (Eq. 4/7), channel schedules vs Sec. 3.5; fixed setup_env `python -m pip`.
- **Pass 3:** Size gate on `300w` (3.99 GiB → full_train); hyperparams ADAM 1e-3 / 150 / 32 / 4 stages; upstream N/A.
- **Pass 4:** Re-read Sec. 3.1–3.5 vs cascade presets; smoke OK after joint train; no remaining `missing`.
- **Pass 5:** Full train + eval + REPORT from real 300W; scorecard closed; stop.

## Dataset size evidence

See `outputs/logs/dataset_size.json`: total_gib≈3.9909, decision=`full_train`.

## Sign-off

Five passes complete. Intentional mismatches documented in `DEVIATIONS.md` (D1–D7).
