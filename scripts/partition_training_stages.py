#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SPECIALIZED = {"numeric_reasoning", "multi_turn"}


def replay(source_id: str, ratio: float, seed: int) -> bool:
    h = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    x = int.from_bytes(h[:8], "big") / 2**64
    return x < ratio


def main():
    ap = argparse.ArgumentParser(description="Build domain-SFT stage 1 and reasoning/dialogue stage 2 with replay.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--replay_ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    Path(args.stage1).parent.mkdir(parents=True, exist_ok=True)
    c1, c2 = Counter(), Counter()
    with open(args.input, "r", encoding="utf-8") as f, open(args.stage1, "w", encoding="utf-8") as s1, open(args.stage2, "w", encoding="utf-8") as s2:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row.get("task_type", "unknown"))
            sid = str(row.get("source_id", ""))
            s1.write(json.dumps(row, ensure_ascii=False) + "\n")
            c1[task] += 1
            if task in SPECIALIZED or replay(sid, args.replay_ratio, args.seed):
                s2.write(json.dumps(row, ensure_ascii=False) + "\n")
                c2[task] += 1

    report = {
        "stage1": {"samples": sum(c1.values()), "task_distribution": dict(c1)},
        "stage2": {"samples": sum(c2.values()), "task_distribution": dict(c2), "replay_ratio": args.replay_ratio},
        "policy": "stage1=all clean domain SFT; stage2=numeric_reasoning+multi_turn+deterministic replay",
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
