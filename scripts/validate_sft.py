#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finllm_prod.data import normalize_text
from finllm_prod.formatting import normalize_messages


def main():
    ap = argparse.ArgumentParser(description="Validate, deduplicate and normalize SFT JSONL.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--require_source_id", action="store_true")
    ap.add_argument("--min_assistant_chars", type=int, default=20)
    ap.add_argument("--expected_valid", type=int, default=0, help="Fail unless exactly this many valid rows remain; 0 disables.")
    args = ap.parse_args()

    seen = set()
    counts = Counter()
    tasks = Counter()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for path in args.inputs:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    counts["raw"] += 1
                    try:
                        row = json.loads(line)
                        if args.require_source_id and not row.get("source_id"):
                            raise ValueError("missing source_id")
                        messages = normalize_messages(row)
                        assistants = [m["content"] for m in messages if m["role"] == "assistant"]
                        if not assistants or len(normalize_text(assistants[-1])) < args.min_assistant_chars:
                            raise ValueError("assistant answer too short")
                        digest = hashlib.sha256(
                            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
                        ).hexdigest()
                        if digest in seen:
                            counts["duplicate"] += 1
                            continue
                        seen.add(digest)
                        clean = dict(row)
                        clean["messages"] = messages
                        clean.pop("instruction", None)
                        clean.pop("input", None)
                        clean.pop("output", None)
                        out.write(json.dumps(clean, ensure_ascii=False) + "\n")
                        counts["valid"] += 1
                        tasks[str(row.get("task_type", "unknown"))] += 1
                    except Exception:
                        counts["invalid"] += 1

    report = {"counts": dict(counts), "task_distribution": dict(tasks), "output": args.output}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.expected_valid and counts["valid"] != args.expected_valid:
        raise SystemExit(f"expected {args.expected_valid} valid rows, got {counts['valid']}")


if __name__ == "__main__":
    main()
