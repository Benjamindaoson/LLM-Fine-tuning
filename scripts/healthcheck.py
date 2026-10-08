#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import httpx


def probe(url: str, timeout: float):
    t0 = time.perf_counter()
    r = httpx.get(url.rstrip("/") + "/models", timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return {
        "ok": True,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "models": data.get("data", []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--timeout", type=float, default=5)
    ap.add_argument("--wait_seconds", type=float, default=0, help="Retry until ready for up to N seconds.")
    ap.add_argument("--interval", type=float, default=5)
    args = ap.parse_args()

    deadline = time.monotonic() + max(args.wait_seconds, 0)
    last = None
    while True:
        try:
            result = probe(args.url, args.timeout)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        except Exception as exc:
            last = exc
            if time.monotonic() >= deadline:
                print(json.dumps({"ok": False, "error": repr(last)}, ensure_ascii=False, indent=2))
                raise SystemExit(2)
            time.sleep(max(args.interval, 0.2))


if __name__ == "__main__":
    main()
