#!/usr/bin/env bash
# Full protocol on required datasets when size gate says full_train.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=/dev/null
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p outputs/logs outputs/checkpoints outputs/figures outputs/predictions

echo "=== dataset size gate ==="
set +e
bash scripts/check_dataset_size.sh ../../datasets/300w
gate=$?
set -e
if [[ "$gate" -eq 3 ]]; then
  echo "SKIP full train: required datasets >= 5 GiB (see outputs/logs/dataset_size.json)"
  exit 3
elif [[ "$gate" -ne 0 ]]; then
  echo "ERROR: check_dataset_size failed (exit $gate)" >&2
  exit "$gate"
fi

echo "=== full train ==="
"$PY" -u -m tgmoe.train --config configs/full.yaml 2>&1 | tee outputs/logs/train_full.log
echo "=== full eval ==="
"$PY" -u -m tgmoe.eval --config configs/full.yaml --checkpoint outputs/checkpoints/best.pt
echo "=== full predict ==="
"$PY" -u -m tgmoe.predict --config configs/full.yaml --checkpoint outputs/checkpoints/best.pt --limit 32
echo "=== full report ==="
"$PY" -u -m tgmoe.report --config configs/full.yaml
echo "FULL TRAIN+EVAL+REPORT OK"
