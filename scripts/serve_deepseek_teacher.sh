#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${DEEPSEEK_VLLM_IMAGE:-vllm/vllm-openai:v0.8.5}"
MODEL_DIR="${DEEPSEEK_MODEL_DIR:-/home/paperspace/models/DeepSeek-R1-Distill-Qwen-32B-AWQ-1de6a3f}"
CONTAINER="${DEEPSEEK_CONTAINER_NAME:-delta-nla-deepseek-teacher}"
PORT="${DEEPSEEK_PORT:-8000}"

if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "DeepSeek checkpoint is incomplete under $MODEL_DIR" >&2
  exit 2
fi
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "Container $CONTAINER already exists; stop or remove it first." >&2
  exit 2
fi

exec docker run --rm --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" \
  "$IMAGE" \
  --model /model \
  --served-model-name deepseek-r1-distill-qwen-32b-awq \
  --quantization awq_marlin \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --generation-config vllm \
  --host 0.0.0.0 \
  --port 8000
