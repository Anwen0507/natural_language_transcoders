#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_DIR/.venv"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV/bin/pip" install -r "$REPO_DIR/requirements-full-delta.lock.txt"
"$VENV/bin/pip" install --no-deps -e "$REPO_DIR"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" - <<'PY'
import torch, transformers, datasets, pyarrow, delta_nla
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "pyarrow": pyarrow.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "bf16": torch.cuda.is_bf16_supported(),
    "gpu": torch.cuda.get_device_name(0),
})
PY
