#!/usr/bin/env bash
set -euo pipefail
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.8-27B}"
TEACHER_PORT="${TEACHER_PORT:-9000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve "$TEACHER_MODEL" \
  --port "$TEACHER_PORT" \
  --tensor-parallel-size 4 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.92 \
  --served-model-name qwen3.8-27b-teacher \
  --reasoning-parser qwen3 \
  >"$LOG_DIR/teacher_vllm.log" 2>&1 &
echo $! > "$LOG_DIR/teacher_vllm.pid"
echo "Teacher endpoint: http://127.0.0.1:${TEACHER_PORT}/v1"
