#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
TUNED_MODEL="${TUNED_MODEL:?Set TUNED_MODEL to the merged multi-stage model directory}"
BASE_PORT="${BASE_PORT:-8000}"
TUNED_PORT="${TUNED_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve "$BASE_MODEL" \
  --port "$BASE_PORT" \
  --tensor-parallel-size 2 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --served-model-name Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 \
  >"$LOG_DIR/base_vllm.log" 2>&1 &
echo $! > "$LOG_DIR/base_vllm.pid"

CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve "$TUNED_MODEL" \
  --port "$TUNED_PORT" \
  --tensor-parallel-size 2 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --served-model-name financial-qwen3.5-9b \
  --reasoning-parser qwen3 \
  >"$LOG_DIR/tuned_vllm.log" 2>&1 &
echo $! > "$LOG_DIR/tuned_vllm.pid"

echo "Base:  http://127.0.0.1:${BASE_PORT}/v1"
echo "Tuned: http://127.0.0.1:${TUNED_PORT}/v1"
echo "PIDs saved under $LOG_DIR"
