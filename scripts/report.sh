#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=/dev/null
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
CFG="${1:-configs/default.yaml}"
shift || true
exec "$PY" -u -m tgmoe.report --config "$CFG" "$@"
