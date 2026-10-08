# LLMOps 设计

## 训练可靠性

- Checkpoint/Resume：自动发现最新检查点并恢复优化器、调度器和训练状态。
- Early Stopping：每 300 步评估；连续 5 次改善小于 0.002 时停止。
- Best Model Selection：按验证损失恢复最佳检查点，再导出 LoRA。
- Versioned Artifacts：Stage 1、Stage 2、合并模型和评测产物使用独立目录。

## 模型发布

```text
LoRA adapter
    ↓ merge
Standalone model
    ↓ offline benchmark
Quality gate
    ↓ dual serving
Base / Candidate A-B
    ↓ promote or rollback
Production alias
```

## 推理服务

- vLLM OpenAI-compatible API
- Tensor Parallel
- 基础模型与候选模型双实例
- 健康检查、PID 管理、异步 A/B 请求
- 统一的模型别名和版本追踪

## 质量门禁

候选模型必须同时检查：

1. 金融综合任务正确率；
2. 数值与事实回答准确率；
3. 无依据数字生成率；
4. FinanceIQ 外部准确率；
5. 延迟和显存预算。
