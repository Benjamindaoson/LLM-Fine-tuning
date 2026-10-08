#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict

JSON_BLOCK = re.compile(r"\{.*\}", re.S)
STOP = object()

SYSTEM = """你是金融领域训练数据构建器。给定一篇金融原文，生成一个高质量中文监督样本。
要求：
1. 问题必须可由原文与基础金融知识回答，禁止虚构原文不存在的公司、日期和数字；
2. 覆盖金融知识问答、财务分析、行业研究、数值推理、多轮追问等任务；
3. 回答结构清晰，关键数字可追溯；如需推导，写清计算依据；必要时明确不确定性；
4. 不给出无依据的明确买入/卖出指令；
5. 多轮样本必须保持 user/assistant 交替，并以 assistant 结束；
6. 仅输出 JSON，不要 Markdown。
JSON 结构：
{"task_type":"financial_qa|financial_analysis|numeric_reasoning|industry_research|multi_turn","messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
多轮样本可继续追加 user/assistant。"""


def parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.I)
    try:
        obj = json.loads(text)
    except Exception:
        m = JSON_BLOCK.search(text)
        if not m:
            raise ValueError("no JSON object in model response")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        raise ValueError("invalid generated schema")
    return obj


def validate_message_structure(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("message must be an object")
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            raise ValueError("invalid role/content")
        normalized.append(role)
    start = 0
    if normalized[0] == "system":
        start = 1
    if start >= len(normalized) or normalized[start] != "user":
        raise ValueError("dialogue must start with user after optional system")
    expected = "user"
    for role in normalized[start:]:
        if role != expected:
            raise ValueError(f"dialogue roles must alternate, expected {expected}, got {role}")
        expected = "assistant" if expected == "user" else "user"
    if normalized[-1] != "assistant":
        raise ValueError("dialogue must end with assistant")


def choose_task(source_id: str) -> str:
    task_types = ["financial_qa", "financial_analysis", "numeric_reasoning", "industry_research", "multi_turn"]
    weights = [25, 25, 20, 15, 15]
    bucket = int(source_id[:8], 16) % sum(weights)
    acc = 0
    for task, weight in zip(task_types, weights):
        acc += weight
        if bucket < acc:
            return task
    return task_types[-1]


async def request_one(client, base_url: str, api_key: str, model: str, row: Dict[str, Any], timeout: float):
    requested = choose_task(row["source_id"])
    user = (
        f"指定任务类型：{requested}\n"
        f"标题：{row.get('title','')}\n来源：{row.get('source','')}\n\n"
        f"金融原文：\n{row.get('text','')}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": 1800,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = await client.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"].get("content", "")
    obj = parse_json(content)
    validate_message_structure(obj.get("messages"))
    obj["task_type"] = requested
    obj["source_id"] = row["source_id"]
    obj["split"] = row.get("split")
    obj["source"] = row.get("source")
    obj["source_text_sha256"] = row.get("text_sha256")
    obj["generator_model"] = model
    return obj


def conversation_digest(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj.get("messages", []), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_existing(path: Path) -> tuple[set[str], set[str]]:
    done: set[str] = set()
    digests: set[str] = set()
    if not path.exists():
        return done, digests
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("source_id")
                if sid:
                    done.add(str(sid))
                if obj.get("messages"):
                    digests.add(conversation_digest(obj))
            except Exception:
                continue
    return done, digests


async def main_async(args):
    import httpx

    output = Path(args.output)
    error_log = Path(args.error_log)
    done, seen_digests = load_existing(output) if args.resume else (set(), set())
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(args.concurrency * 4, 64))
    write_lock = asyncio.Lock()
    stats = {"input": 0, "queued": 0, "ok": 0, "failed": 0, "skipped_existing": 0}

    limits = httpx.Limits(max_connections=max(args.concurrency + 4, 16), max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client, output.open("a", encoding="utf-8") as out_f, error_log.open("a", encoding="utf-8") as err_f:

        async def producer():
            emitted = 0
            with open(args.input, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    stats["input"] += 1
                    row = json.loads(line)
                    sid = str(row.get("source_id", ""))
                    if not sid:
                        async with write_lock:
                            err_f.write(json.dumps({"source_id": None, "error": "missing source_id"}, ensure_ascii=False) + "\n")
                            err_f.flush()
                            stats["failed"] += 1
                        continue
                    if sid in done:
                        stats["skipped_existing"] += 1
                        continue
                    if args.limit and emitted >= args.limit:
                        break
                    await queue.put(row)
                    emitted += 1
                    stats["queued"] += 1
            for _ in range(args.concurrency):
                await queue.put(STOP)

        async def worker(worker_id: int):
            while True:
                row = await queue.get()
                try:
                    if row is STOP:
                        return
                    last = None
                    for attempt in range(args.retries):
                        try:
                            obj = await request_one(client, args.base_url, args.api_key, args.model, row, args.timeout)
                            final_assistants = [m.get("content", "") for m in obj.get("messages", []) if m.get("role") == "assistant"]
                            if not final_assistants or len(final_assistants[-1].strip()) < 20:
                                raise ValueError("assistant answer too short")
                            digest = conversation_digest(obj)
                            async with write_lock:
                                if digest in seen_digests:
                                    raise ValueError("duplicate generated conversation")
                                seen_digests.add(digest)
                                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                out_f.flush()
                                stats["ok"] += 1
                            break
                        except Exception as exc:
                            last = exc
                            if attempt + 1 < args.retries:
                                await asyncio.sleep(min(20.0, 2 ** attempt + random.random()))
                    else:
                        async with write_lock:
                            err_f.write(json.dumps({
                                "source_id": row.get("source_id"),
                                "worker": worker_id,
                                "error": repr(last),
                            }, ensure_ascii=False) + "\n")
                            err_f.flush()
                            stats["failed"] += 1
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
        await producer()
        await queue.join()
        await asyncio.gather(*workers)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["failed"] and not args.allow_partial:
        raise SystemExit(f"generation incomplete: {stats['failed']} rows failed after retries")


def main():
    ap = argparse.ArgumentParser(description="Streaming, resumable financial SFT generation with an OpenAI-compatible local teacher.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--error_log", default="generation_errors.jsonl")
    ap.add_argument("--base_url", default=os.getenv("TEACHER_BASE_URL", "http://127.0.0.1:9000/v1"))
    ap.add_argument("--api_key", default=os.getenv("TEACHER_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "Qwen/Qwen3.8-27B"))
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow_partial", action="store_true", help="Exit zero even if some rows fail after all retries.")
    args = ap.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.error_log).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
