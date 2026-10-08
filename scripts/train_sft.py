#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finllm_prod import MODEL_ID, MODEL_REVISION
from finllm_prod.dataset import IndexedJsonlDataset
from finllm_prod.formatting import ResponseOnlyCollator
from finllm_prod.modeling import load_text_model, load_tokenizer


def parse_args():
    ap = argparse.ArgumentParser(description="Qwen3.5-9B response-only LoRA SFT trainer for financial-domain supervised fine-tuning.")
    ap.add_argument("--model_name_or_path", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--truncation", choices=["keep_end", "keep_start", "drop"], default="keep_end")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--per_device_train_batch_size", type=int, default=3)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--eval_steps", type=int, default=300)
    ap.add_argument("--save_steps", type=int, default=300)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_total_limit", type=int, default=4)
    ap.add_argument("--dataloader_num_workers", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target", default="all-linear")
    ap.add_argument("--deepspeed", default=str(ROOT / "configs" / "ds_zero2.json"))
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto")
    ap.add_argument("--attn_implementation", choices=["auto", "sdpa", "eager", "flash_attention_2"], default="auto")
    ap.add_argument("--resume_from_checkpoint", default="auto")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument("--early_stopping_threshold", type=float, default=0.002)
    return ap.parse_args()


def main():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_name_or_path, args.revision, padding_side="right")
    train_ds = IndexedJsonlDataset(args.train_files, tokenizer, args.max_length, args.truncation, args.enable_thinking)
    eval_ds = IndexedJsonlDataset([args.eval_file], tokenizer, args.max_length, args.truncation, args.enable_thinking)
    collator = ResponseOnlyCollator(tokenizer.pad_token_id, pad_to_multiple_of=8)

    use_bf16 = args.precision == "bf16" or (
        args.precision == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    use_fp16 = args.precision == "fp16" or (args.precision == "auto" and not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = load_text_model(
        args.model_name_or_path,
        args.revision,
        dtype=dtype,
        device_map=None,
        use_cache=False,
        attn_implementation=args.attn_implementation,
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = args.lora_target if args.lora_target == "all-linear" else [x.strip() for x in args.lora_target.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(json.dumps({
            "base_model": args.model_name_or_path,
            "revision": args.revision,
            "train_records": len(train_ds),
            "eval_records": len(eval_ds),
            "bad_train_records": train_ds.bad_records,
            "bad_eval_records": eval_ds.bad_records,
            "world_size": world_size,
            "global_effective_batch": global_batch,
            "lora_target": target_modules,
        }, ensure_ascii=False, indent=2))

    kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "logs"),
        remove_unused_columns=True,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    training_args = TrainingArguments(**kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )],
    )

    resume = None
    if args.resume_from_checkpoint == "auto":
        resume = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    elif args.resume_from_checkpoint.lower() not in {"none", "false", "0"}:
        resume = args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=resume)
    adapter_dir = Path(args.output_dir) / "best_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "base_model": args.model_name_or_path,
        "revision": args.revision,
        "train_files": args.train_files,
        "eval_file": args.eval_file,
        "train_records": len(train_ds),
        "eval_records": len(eval_ds),
        "world_size": world_size,
        "global_effective_batch": global_batch,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "adapter_dir": str(adapter_dir),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "target": target_modules},
    }
    if rank == 0:
        (Path(args.output_dir) / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
