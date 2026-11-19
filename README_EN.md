# Financial Domain Adaptation & Numerical Reasoning with Qwen3.5-9B

**SFT · LoRA · DeepSpeed ZeRO-2 · vLLM · LLMOps**

This repository implements an end-to-end supervised fine-tuning and deployment workflow for financial question answering, industry research, and numerical reasoning. It covers data governance, local supervision generation, two-stage SFT, independent evaluation, inference serving, and controlled model release.

[简体中文 README](README.md)

## Research and engineering summary

- Base model: Qwen3.5-9B.
- Teacher model for local supervision generation: Qwen3.8-27B.
- Source corpus: approximately 1.6 million financial text records.
- Generated supervised data: approximately 128,000 examples.
- Validation set: 2,000 source-isolated samples.
- Frozen held-out test set: 8,000 source-isolated samples.
- Training: two-stage LoRA SFT with a four-GPU DeepSpeed ZeRO-2 configuration.
- GPU compute: rented NVIDIA RTX 4090 24GB resources.
- Serving: vLLM with tensor parallelism and separate Base/Candidate endpoints.

## Main evaluation results

| Metric | Base model | Two-stage financial SFT | Change |
|---|---:|---:|---:|
| Financial task accuracy | 51.8% | **73.6%** | +21.8 pp |
| Numerical and factual accuracy | 62.4% | **81.1%** | +18.7 pp |
| Unsupported numerical generation | 14.7% | **5.2%** | approximately 65% lower |

The evaluation uses 8,000 source-level isolated held-out samples. The repository retains the benchmark snapshot and evaluation scripts so that results can be inspected and reproduced.

## Data governance and leakage prevention

The pipeline separates source documents before generating supervised conversations. Stable source identifiers are used to prevent content derived from the same source document from crossing training, validation, and test boundaries.

The data workflow includes:

- Corpus ingestion and normalization.
- Quality filtering and global deduplication.
- Source-level train/validation/test partitioning.
- Local teacher-model generation.
- Schema and conversation validation.
- Duplicate detection and leakage auditing.
- Resumable generation with source-level checkpoints.

In a 5,000-example generation validation, 4,999 examples were generated successfully (99.98%). Generation supports bounded asynchronous concurrency, exponential backoff, structured error logging, and resumable processing.

## Two-stage SFT

### Stage 1 — Financial knowledge and common analytical tasks

The first stage focuses on financial knowledge, question answering, industry research, and financial analysis.

### Stage 2 — Numerical reasoning and multi-turn analysis

The second stage focuses on numerical reasoning, factual verification, and multi-turn analytical continuity. A 20% replay mixture of other financial tasks is retained to reduce forgetting.

Training uses the model's native chat template and assistant-only loss masking. System and user tokens, as well as padding tokens, are excluded from the loss.

## Training and serving infrastructure

The repository includes:

- DeepSpeed ZeRO-2 multi-GPU training.
- LoRA adapter checkpointing and model merging.
- Gradient checkpointing and optimizer offload.
- Validation-based checkpoint selection and early stopping.
- vLLM serving with tensor parallelism.
- Base/Candidate A/B comparison.
- Health checks, model-version tracking, promotion, and rollback.

The distributed configuration uses four GPUs. The rented RTX 4090 24GB resources are the GPU platform used for the project's GPU-backed experiments; consult the run configuration and result artifacts for the exact execution scope.

## Reproduction entry points

Install the project in a Python virtual environment:

```bash
git clone https://github.com/Benjamindaoson/LLM-Fine-tuning.git
cd LLM-Fine-tuning
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,serve]'
```

Run the environment preflight:

```bash
./run_pipeline.sh preflight
```

The pipeline can be executed end to end when the local teacher-model endpoint and data paths are configured:

```bash
./run_pipeline.sh all
```

Or run individual stages:

```bash
./run_pipeline.sh corpus
./run_pipeline.sh sample
./run_pipeline.sh generate
./run_pipeline.sh validate
./run_pipeline.sh stage1
./run_pipeline.sh stage2
./run_pipeline.sh evaluate
```

See the Chinese README for detailed dataset preparation, training commands, configuration values, serving setup, and project structure.

## Evidence and documentation

- [Benchmark snapshot](reports/benchmark_snapshot.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data pipeline](docs/DATA_PIPELINE.md)
- [Data schema](docs/DATA_SCHEMA.md)
- [Evaluation](docs/EVALUATION.md)
- [LLMOps workflow](docs/LLMOPS.md)
- [Training configuration](configs/)
- [Evaluation and training scripts](scripts/)

## Scope and limitations

This repository demonstrates a domain-specific SFT and LLMOps engineering workflow. It does not claim that the model is a general financial expert, nor that the reported internal benchmark is an external leaderboard result. The 8,000-sample held-out evaluation, source-level split policy, and benchmark definitions should be inspected when interpreting the reported improvements.

