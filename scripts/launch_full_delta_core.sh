#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_DIR/configs/full_delta_h100.yaml}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
RUN_DIR="$($PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["run_dir"])' "$CONFIG")"
STATUS_DIR="$RUN_DIR/status"
BUDGET_STATE="$STATUS_DIR/budget.json"
mkdir -p "$STATUS_DIR"

# Preserve one deadline across resumptions. This prevents a restarted launcher
# from silently granting itself another complete H100 budget.
if [[ ! -f "$BUDGET_STATE" ]]; then
  "$PYTHON" - "$CONFIG" "$BUDGET_STATE" <<'PY'
import json
import os
import sys
import time

import yaml

config_path, destination = sys.argv[1:]
with open(config_path) as handle:
    budget_hours = float(yaml.safe_load(handle)["runtime"]["budget_hours"])
started = int(time.time())
payload = {
    "budget_hours": budget_hours,
    "started_epoch": started,
    "deadline_epoch": started + round(budget_hours * 3600),
}
temporary = destination + ".tmp"
with open(temporary, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, destination)
PY
fi

deadline="$($PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["deadline_epoch"]))' "$BUDGET_STATE")"
remaining="$((deadline - $(date +%s)))"
if (( remaining <= 0 )); then
  touch "$STATUS_DIR/BUDGET_EXHAUSTED"
  echo "Experiment budget deadline has already passed." >&2
  exit 124
fi

echo "[$(date --iso-8601=seconds)] hard budget guard: ${remaining}s remain"
exec timeout --signal=TERM --kill-after=600s "${remaining}s" \
  "$REPO_DIR/scripts/run_full_delta_experiment.sh" "$CONFIG"
