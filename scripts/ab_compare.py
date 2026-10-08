#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


def percentile(values, q: float):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


async def call(client, url, model, messages, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = await client.post(
        url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    latency = time.perf_counter() - t0
    usage = data.get("usage", {}) or {}
    completion = int(usage.get("completion_tokens", 0) or 0)
    return {
        "latency_s": latency,
        "text": data["choices"][0]["message"].get("content", ""),
        "usage": usage,
        "completion_tokens_per_s": completion / latency if completion and latency > 0 else None,
    }


def summarize(rows, key):
    lat = [float(r[key]["latency_s"]) for r in rows if r.get(key)]
    tps = [float(r[key]["completion_tokens_per_s"]) for r in rows if r.get(key) and r[key].get("completion_tokens_per_s") is not None]
    return {
        "requests": len(lat),
        "latency_mean_s": statistics.fmean(lat) if lat else None,
        "latency_p50_s": percentile(lat, 0.50),
        "latency_p95_s": percentile(lat, 0.95),
        "completion_tokens_per_s_mean": statistics.fmean(tps) if tps else None,
    }


async def run(args):
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if args.max_samples and len(rows) >= args.max_samples:
                    break
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
    async with httpx.AsyncClient(limits=limits) as client, open(out, "w", encoding="utf-8") as fout:
        for row in rows:
            messages = row.get("prompt_messages") or row.get("messages")
            if not messages:
                continue
            base_task = call(client, args.base_url, args.base_model, messages, args.max_tokens)
            tuned_task = call(client, args.tuned_url, args.tuned_model, messages, args.max_tokens)
            base, tuned = await asyncio.gather(base_task, tuned_task)
            rec = {
                "id": row.get("id", row.get("source_id")),
                "reference": row.get("reference"),
                "base": base,
                "tuned": tuned,
            }
            results.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    summary = {
        "base": summarize(results, "base"),
        "tuned": summarize(results, "tuned"),
        "samples": len(results),
        "base_url": args.base_url,
        "tuned_url": args.tuned_url,
    }
    summary_path = out.with_name(out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Concurrent online A/B comparison against two vLLM OpenAI-compatible endpoints.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tuned_url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--base_model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tuned_model", default="financial-qwen3.5-9b")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=100)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
