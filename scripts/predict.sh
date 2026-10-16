#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=/dev/null
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CFG="${1:-configs/default.yaml}"
shift || true
exec "$PY" -u -m tgmoe.predict --config "$CFG" "$@"
