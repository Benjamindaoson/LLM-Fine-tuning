from __future__ import annotations

from pathlib import Path
from typing import Any

from . import MODEL_ID, MODEL_REVISION


def _hub_kwargs(model_name_or_path: str, revision: str | None) -> dict[str, Any]:
    if Path(model_name_or_path).exists() or not revision:
        return {}
    return {"revision": revision}


def load_tokenizer(
    model_name_or_path: str = MODEL_ID,
    revision: str | None = MODEL_REVISION,
    padding_side: str = "right",
):
    """Load the Qwen tokenizer and enforce one canonical chat template."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        **_hub_kwargs(model_name_or_path, revision),
    )
    if not getattr(tok, "chat_template", None):
        raise RuntimeError("A tokenizer chat template is required for consistent training and serving")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_text_model(
    model_name_or_path: str = MODEL_ID,
    revision: str | None = MODEL_REVISION,
    dtype: Any = None,
    device_map: Any = None,
    use_cache: bool = False,
    low_cpu_mem_usage: bool = True,
    attn_implementation: str | None = None,
):
    """Load only the Qwen3.5 language backbone for text-domain fine-tuning."""
    from transformers import Qwen3_5ForCausalLM

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "trust_remote_code": True,
        **_hub_kwargs(model_name_or_path, revision),
    }
    if attn_implementation and attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    model = Qwen3_5ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.config.use_cache = use_cache
    return model
