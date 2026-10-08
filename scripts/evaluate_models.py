#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finllm_prod import IGNORE_INDEX, MODEL_REVISION
from finllm_prod.formatting import ResponseOnlyCollator, encode_response_only
from finllm_prod.metrics import aggregate_generation_metrics, number_metrics, rouge_l_f1, rouge_n_f1
from finllm_prod.modeling import load_text_model, load_tokenizer


def parse_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError("--model must be NAME=PATH_OR_HF_ID")
    return spec.split("=", 1)


def load_benchmark(path: str, max_samples: int) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples and len(rows) >= max_samples:
                    break
    if not rows:
        raise ValueError("empty benchmark")
    return rows


def render_prompt(tok, messages):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def eval_answer_ppl(model, tok, rows: List[Dict], max_length: int, batch_size: int, device) -> float:
    collator = ResponseOnlyCollator(tok.pad_token_id, pad_to_multiple_of=8)
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(rows), batch_size):
        feats = []
        for row in rows[start : start + batch_size]:
            messages = list(row["prompt_messages"]) + [{"role": "assistant", "content": row["reference"]}]
            feats.append(encode_response_only(tok, messages, max_length=max_length, truncation="keep_end", enable_thinking=False))
        batch = {k: v.to(device) for k, v in collator(feats).items()}
        active = int((batch["labels"] != IGNORE_INDEX).sum().item())
        with torch.no_grad():
            loss = model(**batch).loss
        total_nll += float(loss.item()) * active
        total_tokens += active
    return math.exp(total_nll / max(total_tokens, 1))


def evaluate_one(name: str, path: str, rows: List[Dict], args, out_dir: Path):
    tok = load_tokenizer(path, args.revision, padding_side="left")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32
    model = load_text_model(path, args.revision, dtype=dtype, device_map="auto", use_cache=True)
    model.eval()
    device = next(model.parameters()).device

    ppl = eval_answer_ppl(model, tok, rows, args.max_length, args.ppl_batch_size, device)
    details = []
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        rendered = [render_prompt(tok, row["prompt_messages"]) for row in chunk]
        batch = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length, add_special_tokens=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        input_width = batch["input_ids"].shape[1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        outputs = tok.batch_decode(generated[:, input_width:], skip_special_tokens=True)
        per_latency = elapsed / len(chunk)
        for row, pred in zip(chunk, outputs):
            ref = row["reference"]
            prompt_text = "\n".join(str(m.get("content", "")) for m in row["prompt_messages"])
            nm = number_metrics(pred, ref, prompt_text)
            details.append({
                "id": row.get("id"),
                "source_id": row.get("source_id"),
                "task_type": row.get("task_type"),
                "prompt_messages": row.get("prompt_messages", []),
                "prediction": pred.strip(),
                "reference": ref,
                "rouge1": rouge_n_f1(pred, ref, 1),
                "rouge2": rouge_n_f1(pred, ref, 2),
                "rougeL": rouge_l_f1(pred, ref),
                **nm,
                "latency_s": per_latency,
            })

    metrics = aggregate_generation_metrics(details)
    metrics.update({"model": name, "path": path, "answer_only_ppl": ppl, "samples": len(details)})
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / f"{name}_generations.jsonl", "w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Controlled Base vs SFT vs Multi-stage financial benchmark.")
    ap.add_argument("--model", action="append", required=True, help="Repeat NAME=PATH_OR_HF_ID")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--ppl_batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_benchmark(args.benchmark, args.max_samples)
    metrics = [evaluate_one(*parse_spec(spec), rows, args, out) for spec in args.model]

    fields = ["model", "answer_only_ppl", "rouge1", "rouge2", "rougeL", "number_precision", "number_recall", "number_f1", "unsupported_number_rate", "latency_s", "samples", "path"]
    with open(out / "comparison.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: m.get(k) for k in fields} for m in metrics])
    md = ["# Base / SFT / Multi-stage 统一评测", "", "| 模型 | 回答困惑度↓ | 字符ROUGE-L↑ | 数值F1↑ | 无依据数字率↓ | 平均延迟(s)↓ |", "|---|---:|---:|---:|---:|---:|"]
    for m in metrics:
        md.append(f"| {m['model']} | {m['answer_only_ppl']:.3f} | {m['rougeL']:.4f} | {m['number_f1']:.4f} | {m['unsupported_number_rate']:.4f} | {m['latency_s']:.3f} |")
    (out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
