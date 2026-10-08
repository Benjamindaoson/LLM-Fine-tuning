#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${LOG_DIR:-./logs}"
PIDFILE="$LOG_DIR/teacher_vllm.pid"
if [[ -f "$PIDFILE" ]]; then
  PID="$(cat "$PIDFILE")"
  kill "$PID" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "teacher stopped"
fi
