#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=/dev/null
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p outputs/logs outputs/checkpoints outputs/figures outputs/predictions
echo "=== import / --help ==="
"$PY" -c "import tgmoe; print('tgmoe OK')"
"$PY" -m tgmoe.train --help >/dev/null
"$PY" -m tgmoe.report --help >/dev/null
echo "=== smoke train ==="
"$PY" -u -m tgmoe.train --config configs/smoke.yaml
echo "=== smoke eval ==="
"$PY" -u -m tgmoe.eval --config configs/smoke.yaml --checkpoint outputs/checkpoints/smoke_best.pt
echo "=== smoke predict ==="
"$PY" -u -m tgmoe.predict --config configs/smoke.yaml --checkpoint outputs/checkpoints/smoke_best.pt --limit 4
echo "=== smoke report ==="
"$PY" -u -m tgmoe.report --config configs/smoke.yaml
echo "SMOKE OK"
