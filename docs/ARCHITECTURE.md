# 系统架构

## 分层结构

```text
Data Plane
  ├─ 多源文件读取
  ├─ 文本规范化与质量过滤
  ├─ SQLite 磁盘级去重
  └─ source_id 稳定切分

Synthesis Plane
  ├─ Qwen3.8-27B 本地教师
  ├─ 有界异步并发
  ├─ 单轮 / 多轮任务生成
  └─ 校验、去重、断点恢复

Training Plane
  ├─ Qwen3.5-9B language backbone
  ├─ assistant-only SFT
  ├─ LoRA all-linear
  ├─ DeepSpeed ZeRO-2
  └─ Checkpoint / Early Stopping / Best Model

Evaluation Plane
  ├─ Base / Stage 1 / Stage 2 同集对照
  ├─ 自动数值与文本指标
  ├─ 本地强模型质量评审
  ├─ FinanceIQ
  └─ 质量准入报告

Serving Plane
  ├─ LoRA merge
  ├─ vLLM Tensor Parallel
  ├─ Base / Candidate 双实例
  ├─ 健康检查与异步 A/B
  └─ 发布 / 回滚
```

## 核心边界

1. 数据切分先于监督样本生成，避免同源泄漏。
2. 模型只负责生成和判断，数据执行、质量门禁和版本治理由确定性程序负责。
3. 训练与服务统一使用原生 Chat Template。
4. 指标只从评测产物生成，不在报告脚本中填充缺失结果。
5. 私有语料、模型权重和 Checkpoint 不进入 Git 仓库。
