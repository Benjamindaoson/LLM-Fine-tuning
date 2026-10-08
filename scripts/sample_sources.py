#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path


def priority(seed: int, source_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{source_id}".encode()).digest()[:8], "big")


def main():
    ap = argparse.ArgumentParser(description="Deterministically select exactly N sources without loading the full corpus.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--require_exact", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    heap = []
    total = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["source_id"]
            p = priority(args.seed, sid)
            item = (-p, sid, row)
            if len(heap) < args.count:
                heapq.heappush(heap, item)
            elif p < -heap[0][0]:
                heapq.heapreplace(heap, item)
            total += 1

    if args.require_exact and total < args.count:
        raise SystemExit(f"requested {args.count} sources but only {total} are available")
    selected = sorted(((-p, sid, row) for p, sid, row in heap), key=lambda x: (x[0], x[1]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for _, _, row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input_rows": total, "selected": len(selected), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
