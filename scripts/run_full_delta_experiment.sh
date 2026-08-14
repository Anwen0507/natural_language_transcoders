#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_DIR/configs/full_delta_h100.yaml}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
RUN_DIR="$($PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["run_dir"])' "$CONFIG")"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/paperspace/.cache/huggingface}"

if [[ ! -f "$RUN_DIR/preflight.json" ]]; then
  echo "preflight.json is absent; run scripts/run_full_delta_smoke.sh first" >&2
  exit 2
fi

start_epoch="$(date +%s)"
printf '{"started_at":"%s","start_epoch":%s,"pid":%s}\n' \
  "$(date --iso-8601=seconds)" "$start_epoch" "$$" > "$STATUS_DIR/launch.json"

on_exit() {
  code=$?
  end_epoch="$(date +%s)"
  printf '{"finished_at":"%s","exit_code":%s,"elapsed_seconds":%s}\n' \
    "$(date --iso-8601=seconds)" "$code" "$((end_epoch-start_epoch))" > "$STATUS_DIR/exit.json"
  if [[ "$code" -eq 0 ]]; then
    touch "$STATUS_DIR/SUCCESS"
  else
    touch "$STATUS_DIR/FAILED"
  fi
  if [[ "${DELTA_NLA_NO_SHUTDOWN:-0}" != "1" ]]; then
    # The experiment config explicitly authorizes power-off on either terminal
    # state. A 10-minute grace period permits SSH inspection/cancellation.
    sudo shutdown -h +10 "delta NLA experiment terminal state: exit $code" || true
  fi
  exit "$code"
}
trap on_exit EXIT

run_stage() {
  name="$1"; shift
  if [[ -f "$STATUS_DIR/${name}.done" ]]; then
    echo "[$(date --iso-8601=seconds)] SKIP $name (already complete)"
    return
  fi
  echo "[$(date --iso-8601=seconds)] START $name"
  "$PYTHON" "$@" 2>&1 | tee -a "$LOG_DIR/${name}.log"
  touch "$STATUS_DIR/${name}.done"
  echo "[$(date --iso-8601=seconds)] DONE $name"
}

run_stage extract -m delta_nla.extract --config "$CONFIG"
run_stage stats -m delta_nla.stats --config "$CONFIG"
run_stage teacher -m delta_nla.teacher --config "$CONFIG"
run_stage sft_ar -m delta_nla.sft --config "$CONFIG" --role ar
run_stage sft_av -m delta_nla.sft --config "$CONFIG" --role av
run_stage rl -m delta_nla.rl --config "$CONFIG"
run_stage evaluate -m delta_nla.evaluate --config "$CONFIG"
