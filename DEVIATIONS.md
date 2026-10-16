# Deviations from the paper

| ID | Topic | Paper | Ours | Reason |
|----|-------|-------|------|--------|
| D1 | 300W protocol | Train on AFW+LFPW+HELEN (3148); test Common/IBUG | Monorepo `300w` Indoor/Outdoor challenge set (600 images), 80/20 split | Official 300W train packs not packaged; challenge set has real 68-pt `.pts` |
| D2 | Pose backbone | Truncated ImageNet ResNet-50 + 3 heads | Lightweight grayscale CNN with 3 heads | Avoid ImageNet dependency; same yaw/pitch/roll I/O (Eq. 1) |
| D3 | Pose labels | Pose from landmarks / pretrained estimator | Landmark-derived yaw/pitch/roll for pose loss | No separate pose annotations in monorepo 300W |
| D4 | AFLW2000-3D / 300W-LP | Train 300W-LP; test AFLW2000-3D by yaw bins | Not run | `aflw2000-3d` empty; 300W-LP absent; `aflw2000` images lack landmark mats |
| D5 | AFLW size | Optional large-pose corpus | Not used in default protocol | ~27 GiB — would trip ≥5 GiB gate; not required once 300W substitute is chosen |
| D6 | Epoch budget | Unspecified in PDF | 25 epochs (full), ADAM 0.001; per-stage L2 + zero-init residual heads | Practical budget on 480-image train split; residual init aids cascade learning |
| D8 | Stage loss | Paper states joint L2; stage-wise schedule not fully specified | Average L2 over all cascade stage outputs | Stabilizes coarse-to-fine residual training |
| D7 | Smoke config | Full Pose-Tree-MoE × 4 stages | Smoke uses `tree_moe` × 2 stages, capped samples | Faster GPU smoke; full config restores paper variant |

## Blockers (honest)

- Cannot reproduce Table 3 (AFLW2000-3D) without landmarks + 300W-LP.
- Reported NME is **ours** on the monorepo 300W split, not a claim of paper Table 1 numbers.
