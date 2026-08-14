#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_DIR/configs/full_delta_h100.yaml}"
PORT="${DEEPSEEK_PORT:-8000}"
CONTAINER="${DEEPSEEK_CONTAINER_NAME:-delta-nla-deepseek-teacher}"

curl --fail --silent --show-error "http://127.0.0.1:${PORT}/health" >/dev/null

export DELTA_NLA_TEACHER_BACKEND=openai_compat
export DELTA_NLA_TEACHER_BASE_URL="http://127.0.0.1:${PORT}/v1"
export DELTA_NLA_TEACHER_MODEL=deepseek-r1-distill-qwen-32b-awq
export DELTA_NLA_TEACHER_REVISION=1de6a3f7b151f6ea0f6d42acb3566e094eb8a264
export DELTA_NLA_TEACHER_GUIDED_REGEX=1
# Single-token forms of the forbidden plain-language terms under this exact
# Qwen tokenizer. Multi-token or unusual-case forms remain covered by retries.
export DELTA_NLA_TEACHER_LOGIT_BIAS_JSON='{"60788":-100,"87440":-100,"18927":-100,"88464":-100,"86639":-100,"48216":-100,"49150":-100,"47502":-100,"97582":-100,"3215":-100,"4621":-100,"3781":-100,"4196":-100,"43687":-100,"71459":-100,"22879":-100,"84744":-100,"52329":-100,"21730":-100,"81426":-100,"77278":-100,"62510":-100,"15089":-100,"79388":-100,"49988":-100,"49418":-100}'
export DELTA_NLA_TEACHER_BATCH_SIZE="${DELTA_NLA_TEACHER_BATCH_SIZE:-32}"
export DELTA_NLA_TEACHER_SERVER_CONTAINER="$CONTAINER"

exec "$REPO_DIR/scripts/launch_full_delta_core.sh" "$CONFIG"
