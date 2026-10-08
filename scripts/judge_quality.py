#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

JSON_RE = re.compile(r"\{.*\}", re.S)
RUBRIC = """你是严格的金融大模型评测器。根据问题、参考答案和候选答案输出 JSON：
{
  "task_correct": 0或1,
  "numerical_factual_correct": 0或1,
  "has_unsupported_number": 0或1,
  "reason": "不超过80字"
}
规则：
1. task_correct 关注是否完成问题且结论正确；
2. numerical_factual_correct 关注数字、计算、时间、公司与事实是否准确；
3. 候选答案出现题目与参考答案均无法支持的具体数字时，has_unsupported_number=1；
4. 不因篇幅、语气或格式给高分；
5. 只输出 JSON。"""


def parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip().strip("`")
    try:
        obj = json.loads(text)
    except Exception:
        match = JSON_RE.search(text)
        if not match:
            raise ValueError("judge returned no JSON")
        obj = json.loads(match.group(0))
    for key in ("task_correct", "numerical_factual_correct", "has_unsupported_number"):
        value = int(obj[key])
        if value not in (0, 1):
            raise ValueError(f"{key} must be 0 or 1")
        obj[key] = value
    obj["reason"] = str(obj.get("reason", ""))[:200]
    return obj


def render_case(row: dict[str, Any]) -> str:
    prompt = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" for m in row.get("prompt_messages", [])
    )
    return f"问题与上下文：\n{prompt}\n\n参考答案：\n{row.get('reference','')}\n\n候选答案：\n{row.get('prediction','')}"


async def judge_one(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, args, row: dict[str, Any]):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": render_case(row)},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    async with semaphore:
        response = await client.post(
            args.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=args.timeout,
        )
    response.raise_for_status()
    score = parse_json(response.json()["choices"][0]["message"].get("content", ""))
    return {**row, "judge": score}


async def run(args):
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if args.max_samples and len(rows) >= args.max_samples:
                    break
    if not rows:
        raise SystemExit("empty generation file")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and out.exists():
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    completed[str(row.get("id"))] = row

    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency + 4, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client, out.open("a", encoding="utf-8") as sink:
        pending = [row for row in rows if str(row.get("id")) not in completed]
        for start in range(0, len(pending), args.concurrency * 4):
            batch = pending[start : start + args.concurrency * 4]
            results = await asyncio.gather(
                *(judge_one(client, semaphore, args, row) for row in batch),
                return_exceptions=True,
            )
            for source, result in zip(batch, results):
                if isinstance(result, Exception):
                    if not args.allow_partial:
                        raise result
                    result = {**source, "judge_error": repr(result)}
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()
                completed[str(source.get("id"))] = result

    judged = [row for row in completed.values() if isinstance(row.get("judge"), dict)]
    if not judged:
        raise SystemExit("no successfully judged rows")
    metrics = {
        "samples": len(judged),
        "task_accuracy": sum(r["judge"]["task_correct"] for r in judged) / len(judged),
        "numerical_factual_accuracy": sum(r["judge"]["numerical_factual_correct"] for r in judged) / len(judged),
        "unsupported_number_rate": sum(r["judge"]["has_unsupported_number"] for r in judged) / len(judged),
        "judge_model": args.model,
    }
    metrics_path = out.with_name(out.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Judge financial generations with a local OpenAI-compatible model.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_url", default=os.getenv("JUDGE_BASE_URL", "http://127.0.0.1:9000/v1"))
    ap.add_argument("--api_key", default=os.getenv("JUDGE_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "Qwen/Qwen3.8-27B"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
