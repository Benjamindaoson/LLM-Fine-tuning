# 数据格式

## 原始文档

```json
{
  "source_id": "24-char-stable-id",
  "split": "train|val|test",
  "title": "公司季度报告",
  "text": "原始金融文本",
  "summary": "可选摘要",
  "source": "announcement",
  "text_sha256": "full-sha256"
}
```

## SFT 样本

```json
{
  "source_id": "24-char-stable-id",
  "split": "train",
  "task_type": "numeric_reasoning",
  "source": "announcement",
  "source_text_sha256": "full-sha256",
  "generator_model": "Qwen/Qwen3.8-27B",
  "messages": [
    {"role": "system", "content": "你是金融分析助手。"},
    {"role": "user", "content": "请计算并解释同比变化。"},
    {"role": "assistant", "content": "计算过程与结论。"}
  ]
}
```

## 任务类型

- `financial_qa`
- `financial_analysis`
- `numeric_reasoning`
- `industry_research`
- `multi_turn`

## 评测样本

```json
{
  "id": "source-id",
  "source_id": "source-id",
  "task_type": "financial_analysis",
  "prompt_messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "reference": "参考答案"
}
```
