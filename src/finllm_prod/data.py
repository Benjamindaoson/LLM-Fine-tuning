from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
WS_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = ZERO_WIDTH_RE.sub("", text)
    return WS_RE.sub(" ", text).strip()


def first_nonempty(record: Dict[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = normalize_text(record.get(field, ""))
        if value:
            return value
    return ""


def canonical_source_text(
    record: Dict[str, Any],
    text_fields: Sequence[str] = ("Article", "article", "content", "text", "body"),
    title_fields: Sequence[str] = ("title", "headline", "Title"),
    summary_fields: Sequence[str] = ("Summary", "summary", "abstract"),
) -> tuple[str, str, str]:
    text = first_nonempty(record, text_fields)
    title = first_nonempty(record, title_fields)
    summary = first_nonempty(record, summary_fields)
    if not text:
        raise ValueError("record contains no source text")
    return title, text, summary


def source_id(record: Dict[str, Any]) -> str:
    title, text, summary = canonical_source_text(record)
    payload = "\x1f".join((title, text, summary)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def exact_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def deterministic_split(source_key: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("ratios must satisfy train>0, val>=0, train+val<1")
    h = hashlib.sha256(f"{seed}:{source_key}".encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big") / 2**64
    if x < train_ratio:
        return "train"
    if x < train_ratio + val_ratio:
        return "val"
    return "test"


def basic_quality_filter(text: str, min_chars: int = 200, max_chars: int = 20000) -> tuple[bool, str]:
    text = normalize_text(text)
    n = len(text)
    if n < min_chars:
        return False, "too_short"
    if max_chars and n > max_chars:
        return False, "too_long"
    printable = sum(ch.isprintable() for ch in text)
    if n and printable / n < 0.98:
        return False, "non_printable"
    useful = sum(ch.isalnum() or ("\u4e00" <= ch <= "\u9fff") for ch in text)
    if n and useful / n < 0.35:
        return False, "low_signal"
    return True, "ok"


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{lineno}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def iter_records(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        yield from iter_jsonl(path)
        return
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    yield row
        elif isinstance(obj, dict):
            rows = obj.get("data") if isinstance(obj.get("data"), list) else [obj]
            for row in rows:
                if isinstance(row, dict):
                    yield row
        return
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("reading parquet requires pyarrow") from exc
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                yield row
        return
    raise ValueError(f"unsupported source format: {path}")


def discover_files(inputs: Sequence[str]) -> List[Path]:
    supported = {".jsonl", ".ndjson", ".json", ".csv", ".parquet", ".pq"}
    files: List[Path] = []
    for value in inputs:
        p = Path(value)
        if p.is_dir():
            files.extend(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in supported)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(value)
    return sorted(set(files))


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


class SqliteDeduper:
    """Disk-backed exact deduper suitable for million-scale corpora."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY)")
        self.conn.commit()

    def add(self, key: str) -> bool:
        try:
            self.conn.execute("INSERT INTO seen(hash) VALUES (?)", (key,))
            return True
        except sqlite3.IntegrityError:
            return False

    def flush(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
