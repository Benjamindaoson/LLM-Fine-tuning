#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/finllm_data}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/finllm_runs/qwen35_9b}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
REVISION="${MODEL_REVISION:-}"
RAW_INPUT="${RAW_INPUT:-/root/autodl-tmp/raw_financial_data}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-80}"

mkdir -p "$DATA_ROOT" "$RUN_ROOT"
REVISION_ARGS=()
if [[ -n "$REVISION" ]]; then
  REVISION_ARGS=(--revision "$REVISION")
fi

train_common=(
  --per_device_train_batch_size 3
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps 16
  --eval_steps 300
  --save_steps 300
  --early_stopping_patience 5
  --early_stopping_threshold 0.002
)

case "${1:-help}" in
  preflight)
    python "$ROOT/scripts/preflight.py" --require_gpus 4
    ;;
  corpus)
    python "$ROOT/scripts/build_corpus.py" \
      --inputs "$RAW_INPUT" \
      --output_dir "$DATA_ROOT/corpus" \
      --max_unique 1600000 \
      --min_unique 1600000
    ;;
  sample)
    python "$ROOT/scripts/sample_sources.py" --input "$DATA_ROOT/corpus/sources_train.jsonl" --output "$DATA_ROOT/selected_train_128k.jsonl" --count 128000
    python "$ROOT/scripts/sample_sources.py" --input "$DATA_ROOT/corpus/sources_val.jsonl"   --output "$DATA_ROOT/selected_val_2k.jsonl"     --count 2000
    python "$ROOT/scripts/sample_sources.py" --input "$DATA_ROOT/corpus/sources_test.jsonl"  --output "$DATA_ROOT/selected_test_8k.jsonl"    --count 8000
    ;;
  generate)
    python "$ROOT/scripts/generate_sft.py" --input "$DATA_ROOT/selected_train_128k.jsonl" --output "$DATA_ROOT/sft_train_raw.jsonl" --error_log "$DATA_ROOT/train_generation_errors.jsonl" --concurrency "$GEN_CONCURRENCY" --resume
    python "$ROOT/scripts/generate_sft.py" --input "$DATA_ROOT/selected_val_2k.jsonl" --output "$DATA_ROOT/sft_val_raw.jsonl" --error_log "$DATA_ROOT/val_generation_errors.jsonl" --concurrency "$GEN_CONCURRENCY" --resume
    python "$ROOT/scripts/generate_sft.py" --input "$DATA_ROOT/selected_test_8k.jsonl" --output "$DATA_ROOT/sft_test_raw.jsonl" --error_log "$DATA_ROOT/test_generation_errors.jsonl" --concurrency "$GEN_CONCURRENCY" --resume
    ;;
  validate)
    python "$ROOT/scripts/validate_sft.py" --inputs "$DATA_ROOT/sft_train_raw.jsonl" --output "$DATA_ROOT/sft_train.jsonl" --report "$DATA_ROOT/sft_train_report.json" --require_source_id --expected_valid 128000
    python "$ROOT/scripts/validate_sft.py" --inputs "$DATA_ROOT/sft_val_raw.jsonl" --output "$DATA_ROOT/sft_val.jsonl" --report "$DATA_ROOT/sft_val_report.json" --require_source_id --expected_valid 2000
    python "$ROOT/scripts/validate_sft.py" --inputs "$DATA_ROOT/sft_test_raw.jsonl" --output "$DATA_ROOT/sft_test.jsonl" --report "$DATA_ROOT/sft_test_report.json" --require_source_id --expected_valid 8000
    python "$ROOT/scripts/audit_splits.py" --train "$DATA_ROOT/sft_train.jsonl" --val "$DATA_ROOT/sft_val.jsonl" --test "$DATA_ROOT/sft_test.jsonl" --report "$DATA_ROOT/leakage_audit.json"
    python "$ROOT/scripts/partition_training_stages.py" --input "$DATA_ROOT/sft_train.jsonl" --stage1 "$DATA_ROOT/sft_stage1.jsonl" --stage2 "$DATA_ROOT/sft_stage2.jsonl" --report "$DATA_ROOT/stage_partition_report.json"
    python "$ROOT/scripts/make_benchmark.py" --input "$DATA_ROOT/sft_test.jsonl" --output "$DATA_ROOT/financial_test_8k.jsonl" --max_samples 8000 --require_source_id
    ;;
  stage1)
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 "$ROOT/scripts/train_sft.py" \
      --model_name_or_path "$MODEL" "${REVISION_ARGS[@]}" \
      --train_files "$DATA_ROOT/sft_stage1.jsonl" --eval_file "$DATA_ROOT/sft_val.jsonl" \
      --output_dir "$RUN_ROOT/stage1" --lora_r 32 --lora_alpha 64 --learning_rate 5e-6 --epochs 5 \
      "${train_common[@]}"
    python "$ROOT/scripts/merge_lora.py" --base_model "$MODEL" "${REVISION_ARGS[@]}" \
      --adapter "$RUN_ROOT/stage1/best_adapter" --output "$RUN_ROOT/stage1_merged"
    ;;
  stage2)
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 "$ROOT/scripts/train_sft.py" \
      --model_name_or_path "$RUN_ROOT/stage1_merged" \
      --train_files "$DATA_ROOT/sft_stage2.jsonl" --eval_file "$DATA_ROOT/sft_val.jsonl" \
      --output_dir "$RUN_ROOT/multistage" --lora_r 24 --lora_alpha 48 --learning_rate 2e-6 --epochs 4 \
      "${train_common[@]}"
    python "$ROOT/scripts/merge_lora.py" --base_model "$RUN_ROOT/stage1_merged" \
      --adapter "$RUN_ROOT/multistage/best_adapter" --output "$RUN_ROOT/multistage_merged"
    ;;
  evaluate)
    python "$ROOT/scripts/evaluate_models.py" \
      --model base="$MODEL" --model sft="$RUN_ROOT/stage1_merged" --model multistage="$RUN_ROOT/multistage_merged" \
      --benchmark "$DATA_ROOT/financial_test_8k.jsonl" --output_dir "$RUN_ROOT/benchmark"
    python "$ROOT/scripts/evaluate_financeiq.py" \
      --model base="$MODEL" --model sft="$RUN_ROOT/stage1_merged" --model multistage="$RUN_ROOT/multistage_merged" \
      --output_dir "$RUN_ROOT/financeiq"
    ;;
  report)
    python "$ROOT/scripts/build_report.py" \
      --generation_dir "$RUN_ROOT/benchmark" \
      --quality_dir "$RUN_ROOT/quality" \
      --financeiq_dir "$RUN_ROOT/financeiq" \
      --output "$RUN_ROOT/FINAL_REPORT.md"
    ;;
  all)
    "$0" preflight; "$0" corpus; "$0" sample; "$0" generate; "$0" validate; "$0" stage1; "$0" stage2; "$0" evaluate
    ;;
  all-local)
    "$0" preflight; "$0" corpus; "$0" sample
    bash "$ROOT/deploy/start_teacher.sh"
    trap 'bash "$ROOT/deploy/stop_teacher.sh" >/dev/null 2>&1 || true' EXIT
    python "$ROOT/scripts/healthcheck.py" --url "${TEACHER_BASE_URL:-http://127.0.0.1:9000/v1}" --wait_seconds 900 --interval 10
    "$0" generate
    bash "$ROOT/deploy/stop_teacher.sh"
    trap - EXIT
    "$0" validate; "$0" stage1; "$0" stage2; "$0" evaluate
    ;;
  help|*)
    echo "Usage: $0 {preflight|corpus|sample|generate|validate|stage1|stage2|evaluate|report|all|all-local}"
    ;;
esac
