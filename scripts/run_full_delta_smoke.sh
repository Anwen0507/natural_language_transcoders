#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_DIR/configs/full_delta_h100.yaml}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -m compileall -q "$REPO_DIR/delta_nla"
"$PYTHON" -m delta_nla.preflight --config "$CONFIG"
