#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: str):
    ids, content = set(), set()
    rows = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            sid = row.get("source_id")
            if sid:
                ids.add(str(sid))
            messages = row.get("messages")
            if messages:
                payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            else:
                payload = json.dumps({k: row.get(k) for k in ("instruction", "input", "output")}, ensure_ascii=False, sort_keys=True)
            content.add(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return {"rows": rows, "ids": ids, "content": content}


def main():
    ap = argparse.ArgumentParser(description="Fail if train/val/test source IDs or exact conversations overlap.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    data = {k: load(getattr(args, k)) for k in ("train", "val", "test")}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlaps = {}
    bad = False
    for a, b in pairs:
        sid = len(data[a]["ids"] & data[b]["ids"])
        exact = len(data[a]["content"] & data[b]["content"])
        overlaps[f"{a}_{b}"] = {"source_id_overlap": sid, "exact_conversation_overlap": exact}
        bad = bad or sid > 0 or exact > 0
    report = {
        "rows": {k: data[k]["rows"] for k in data},
        "unique_source_ids": {k: len(data[k]["ids"]) for k in data},
        "overlaps": overlaps,
        "pass": not bad,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
