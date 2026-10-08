# 评测设计

## 三组模型

- Base：Qwen3.5-9B
- Stage 1：金融领域 SFT
- Stage 2：数值推理与多轮分析强化

三组模型使用同一测试集、同一解码参数和同一评审规则。

## 自动指标

- assistant-only perplexity
- 中文字符 ROUGE
- 数值 Precision / Recall / F1
- 无依据数字率
- 生成延迟

## 质量评审

`scripts/judge_quality.py` 使用本地强模型按固定 JSON Rubric 评估：

- task_correct
- numerical_factual_correct
- has_unsupported_number

评审模型温度为 0，输出必须通过 JSON Schema 约束；失败样本单独记录并可恢复。

## 防泄漏

数据在生成问答前按 source_id 划分。`audit_splits.py` 同时检查 source_id 和完整对话哈希的跨集合重复。
