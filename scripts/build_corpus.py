#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finllm_prod.data import (
    SqliteDeduper,
    basic_quality_filter,
    canonical_source_text,
    deterministic_split,
    discover_files,
    exact_text_hash,
    iter_records,
    normalize_text,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Million-scale streaming financial corpus cleaner/deduper/splitter.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Files or directories: JSONL/JSON/CSV/Parquet")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=20000)
    ap.add_argument("--train_ratio", type=float, default=0.94)
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--max_unique", type=int, default=0, help="0 = no limit")
    ap.add_argument("--min_unique", type=int, default=0, help="Fail if fewer unique high-quality documents remain.")
    ap.add_argument("--commit_every", type=int, default=5000)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = discover_files(args.inputs)
    if not files:
        raise SystemExit("no supported input files")

    handles = {s: open(out / f"sources_{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    deduper = SqliteDeduper(out / "dedup.sqlite")
    counts = Counter()
    reject_reasons = Counter()

    try:
        for path in files:
            counts["files"] += 1
            for row in iter_records(path):
                counts["raw"] += 1
                try:
                    title, text, summary = canonical_source_text(row)
                except Exception:
                    reject_reasons["missing_text"] += 1
                    continue
                ok, reason = basic_quality_filter(text, args.min_chars, args.max_chars)
                if not ok:
                    reject_reasons[reason] += 1
                    continue
                h = exact_text_hash(text)
                if not deduper.add(h):
                    reject_reasons["exact_duplicate"] += 1
                    continue

                sid = h[:24]
                split = deterministic_split(sid, args.seed, args.train_ratio, args.val_ratio)
                source_name = normalize_text(row.get("source", row.get("Source", path.stem)))
                rec = {
                    "source_id": sid,
                    "split": split,
                    "title": title,
                    "text": text,
                    "summary": summary,
                    "source": source_name,
                    "text_sha256": h,
                }
                handles[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[f"kept_{split}"] += 1
                counts["unique"] += 1
                if counts["unique"] % args.commit_every == 0:
                    deduper.flush()
                if args.max_unique and counts["unique"] >= args.max_unique:
                    break
            if args.max_unique and counts["unique"] >= args.max_unique:
                break
    finally:
        deduper.close()
        for f in handles.values():
            f.close()

    manifest = {
        "inputs": [str(p) for p in files],
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1 - args.train_ratio - args.val_ratio,
        "counts": dict(counts),
        "rejected": dict(reject_reasons),
        "invariants": ["exact text deduplicated globally", "source_id sets are pairwise disjoint"],
    }
    (out / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.min_unique and counts["unique"] < args.min_unique:
        raise SystemExit(f"need at least {args.min_unique} unique high-quality documents, got {counts['unique']}")


if __name__ == "__main__":
    main()
