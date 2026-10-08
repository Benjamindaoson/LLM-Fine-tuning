#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser(description="Build a measured report from benchmark artifacts; never fills missing values.")
    ap.add_argument("--generation_dir", required=True)
    ap.add_argument("--quality_dir", required=False)
    ap.add_argument("--financeiq_dir", required=False)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gdir = Path(args.generation_dir)
    names = ["base", "sft", "multistage"]
    generation = {name: read_json(gdir / f"{name}_metrics.json") for name in names}
    report: dict = {"generation": generation}

    if args.quality_dir:
        qdir = Path(args.quality_dir)
        quality = {name: read_json(qdir / f"{name}_judged_metrics.json") for name in names}
        report["quality"] = quality
        b, m = quality["base"], quality["multistage"]
        report["quality_delta"] = {
            "task_accuracy_pp": (m["task_accuracy"] - b["task_accuracy"]) * 100,
            "numerical_factual_accuracy_pp": (m["numerical_factual_accuracy"] - b["numerical_factual_accuracy"]) * 100,
            "unsupported_number_rate_pp": (m["unsupported_number_rate"] - b["unsupported_number_rate"]) * 100,
        }

    if args.financeiq_dir:
        fdir = Path(args.financeiq_dir)
        report["financeiq"] = {name: read_json(fdir / f"financeiq_{name}_metrics.json") for name in names}

    md = ["# 金融大模型领域微调实测报告", "", "所有数字均由评测产物读取；缺失文件会使脚本直接失败。", ""]
    if "quality" in report:
        b, m = report["quality"]["base"], report["quality"]["multistage"]
        md.extend([
            "## 核心质量指标",
            "",
            f"- 金融综合任务正确率：{pct(b['task_accuracy'])} → {pct(m['task_accuracy'])}。",
            f"- 数值与事实回答准确率：{pct(b['numerical_factual_accuracy'])} → {pct(m['numerical_factual_accuracy'])}。",
            f"- 无依据数字生成率：{pct(b['unsupported_number_rate'])} → {pct(m['unsupported_number_rate'])}。",
            "",
        ])
    g0, g2 = report["generation"]["base"], report["generation"]["multistage"]
    md.extend([
        "## 自动指标",
        "",
        f"- 数值 F1：{pct(g0['number_f1'])} → {pct(g2['number_f1'])}。",
        f"- 回答区域困惑度：{g0['answer_only_ppl']:.3f} → {g2['answer_only_ppl']:.3f}。",
        f"- 平均生成延迟：{g0['latency_s']:.3f}s → {g2['latency_s']:.3f}s。",
    ])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n\n```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
