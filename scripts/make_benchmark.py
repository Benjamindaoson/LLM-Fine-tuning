#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finllm_prod.formatting import normalize_messages, prompt_messages_and_reference


def main():
    ap = argparse.ArgumentParser(description="Normalize held-out SFT records into the generation benchmark schema.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_samples", type=int, default=8000)
    ap.add_argument("--require_source_id", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if args.require_source_id and not item.get("source_id"):
                raise ValueError("benchmark record missing source_id")
            messages = normalize_messages(item)
            prompt, reference = prompt_messages_and_reference({"messages": messages})
            rows.append({
                "id": item.get("id", item.get("source_id", f"row-{len(rows)}")),
                "source_id": item.get("source_id"),
                "task_type": item.get("task_type", "unknown"),
                "prompt_messages": prompt,
                "reference": reference,
            })
            if args.max_samples and len(rows) >= args.max_samples:
                break
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(rows), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
