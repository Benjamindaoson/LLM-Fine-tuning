#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finllm_prod import MODEL_REVISION
from finllm_prod.modeling import load_text_model, load_tokenizer


def main():
    from peft import PeftModel

    ap = argparse.ArgumentParser(description="Merge LoRA adapter into Qwen3.5 text-only causal LM.")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--revision", default=MODEL_REVISION)
    args = ap.parse_args()

    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32
    base = load_text_model(args.base_model, args.revision, dtype=dtype, device_map="cpu", use_cache=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok = load_tokenizer(args.base_model, args.revision)
    tok.save_pretrained(out)
    manifest = {"base_model": args.base_model, "adapter": args.adapter, "revision": args.revision, "output": str(out)}
    (out / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
