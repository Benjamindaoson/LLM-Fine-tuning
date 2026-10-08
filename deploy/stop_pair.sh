#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${LOG_DIR:-./logs}"
for name in base_vllm tuned_vllm; do
  pidfile="$LOG_DIR/${name}.pid"
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "stopped $name pid=$pid"
    fi
    rm -f "$pidfile"
  fi
done
