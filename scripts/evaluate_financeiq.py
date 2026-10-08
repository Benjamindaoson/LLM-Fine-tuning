#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finllm_prod import MODEL_REVISION
from finllm_prod.modeling import load_text_model, load_tokenizer

CHOICE_RE = re.compile(r'(?i)(?:"answer"\s*:\s*")?\b([ABCD])\b')


def parse_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError("--model must be NAME=PATH_OR_HF_ID")
    return spec.split("=", 1)


def parse_choice(text: str):
    try:
        obj = json.loads((text or "").strip())
        ans = str(obj.get("answer", "")).strip().upper()
        if ans in "ABCD" and len(ans) == 1:
            return ans
    except Exception:
        pass
    m = CHOICE_RE.search((text or "").upper())
    return m.group(1).upper() if m else None


def build_messages(row: Dict) -> List[Dict[str, str]]:
    q = str(row.get("Question", row.get("question", ""))).strip()
    options = "\n".join(f"{c}. {row.get(c, row.get(c.lower(), ''))}" for c in "ABCD")
    return [
        {"role": "system", "content": "你正在参加中文金融专业知识客观测试。只依据题目作答，不要添加无关内容。"},
        {"role": "user", "content": f"{q}\n{options}\n\n请仅输出 JSON：{{\"answer\":\"A\"}}，其中字母只能是 A、B、C、D。"},
    ]


def render(tok, messages):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def load_rows(split_name: str | None, max_samples: int):
    from datasets import load_dataset

    ds = load_dataset("Duxiaoman-DI/FinanceIQ")
    parts = [ds[split_name]] if split_name else [ds[k] for k in ds.keys()]
    rows = []
    for part in parts:
        for row in part:
            gold = row.get("Answer", row.get("answer"))
            if gold:
                rows.append(dict(row))
                if max_samples and len(rows) >= max_samples:
                    return rows
    return rows


def evaluate_one(name: str, path: str, rows: List[Dict], args, out: Path):
    tok = load_tokenizer(path, args.revision, padding_side="left")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32
    model = load_text_model(path, args.revision, dtype=dtype, device_map="auto", use_cache=True)
    model.eval()
    device = next(model.parameters()).device
    preds = []

    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        rendered = [render(tok, build_messages(r)) for r in chunk]
        batch = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        width = batch["input_ids"].shape[1]
        with torch.no_grad():
            seqs = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        raw_outputs = tok.batch_decode(seqs[:, width:], skip_special_tokens=True)
        for row, raw in zip(chunk, raw_outputs):
            pred = parse_choice(raw)
            gold = str(row.get("Answer", row.get("answer", ""))).strip().upper()
            preds.append({
                "question": row.get("Question", row.get("question", "")),
                "gold": gold,
                "prediction": pred,
                "raw_output": raw.strip(),
                "correct": pred == gold,
            })

    metrics = {
        "model": name,
        "path": path,
        "accuracy": sum(p["correct"] for p in preds) / len(preds),
        "valid_choice_rate": sum(p["prediction"] in set("ABCD") for p in preds) / len(preds),
        "samples": len(preds),
    }
    (out / f"financeiq_{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / f"financeiq_{name}_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    ap = argparse.ArgumentParser(description="FinanceIQ objective benchmark for Base/SFT/Multi-stage models.")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--split", default=None)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.split, args.max_samples)
    metrics = [evaluate_one(*parse_spec(spec), rows, args, out) for spec in args.model]
    with open(out / "financeiq_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "accuracy", "valid_choice_rate", "samples", "path"])
        w.writeheader()
        w.writerows(metrics)
    md = ["# FinanceIQ 对照评测", "", "| 模型 | 准确率↑ | 有效选项率↑ | 样本数 |", "|---|---:|---:|---:|"]
    for m in metrics:
        md.append(f"| {m['model']} | {m['accuracy']:.4f} | {m['valid_choice_rate']:.4f} | {m['samples']} |")
    (out / "financeiq_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
