#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys

REQUIRED = [
    "torch", "transformers", "peft", "accelerate", "deepspeed", "datasets",
    "safetensors", "pyarrow", "httpx", "pandas", "yaml"
]


def version_of(mod):
    return getattr(mod, "__version__", "unknown")


def main():
    ap = argparse.ArgumentParser(description="Production environment preflight for the financial LLM pipeline.")
    ap.add_argument("--require_gpus", type=int, default=4)
    ap.add_argument("--teacher_url", default=None)
    args = ap.parse_args()

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
        "executables": {},
        "gpu": {},
        "errors": [],
    }
    loaded = {}
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            loaded[name] = mod
            report["packages"][name] = version_of(mod)
        except Exception as exc:
            report["packages"][name] = None
            report["errors"].append(f"missing/import failed: {name}: {exc}")

    for exe in ("deepspeed", "vllm", "nvidia-smi"):
        report["executables"][exe] = shutil.which(exe)
        if not report["executables"][exe]:
            report["errors"].append(f"missing executable: {exe}")

    torch = loaded.get("torch")
    if torch is not None:
        cuda = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if cuda else 0
        report["gpu"] = {
            "cuda_available": cuda,
            "count": count,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda else False,
            "devices": [torch.cuda.get_device_name(i) for i in range(count)] if cuda else [],
        }
        if count < args.require_gpus:
            report["errors"].append(f"need >= {args.require_gpus} GPUs, found {count}")

    if args.teacher_url:
        try:
            import httpx
            r = httpx.get(args.teacher_url.rstrip("/") + "/models", timeout=5)
            r.raise_for_status()
            report["teacher_endpoint"] = {"ok": True, "models": r.json().get("data", [])}
        except Exception as exc:
            report["teacher_endpoint"] = {"ok": False, "error": repr(exc)}
            report["errors"].append("teacher endpoint not ready")

    report["ok"] = not report["errors"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
