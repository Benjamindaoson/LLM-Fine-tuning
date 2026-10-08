from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from . import IGNORE_INDEX

_ALLOWED_ROLES = {"system", "user", "assistant"}


def _validate_dialogue(messages: Sequence[Dict[str, str]]) -> None:
    if not messages:
        raise ValueError("empty dialogue")
    i = 0
    if messages[0]["role"] == "system":
        i = 1
    if i >= len(messages) or messages[i]["role"] != "user":
        raise ValueError("dialogue must start with user after optional system")
    expected = "user"
    for m in messages[i:]:
        if m["role"] != expected:
            raise ValueError(f"dialogue roles must alternate; expected {expected}, got {m['role']}")
        expected = "assistant" if expected == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError("dialogue must end with assistant")


def normalize_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(item.get("messages"), list):
        out: List[Dict[str, str]] = []
        for raw in item["messages"]:
            if not isinstance(raw, dict):
                raise ValueError("message must be an object")
            role = str(raw.get("role", "")).strip().lower()
            content = str(raw.get("content", "")).strip()
            if role not in _ALLOWED_ROLES or not content:
                raise ValueError("invalid message role/content")
            out.append({"role": role, "content": content})
        _validate_dialogue(out)
        return out

    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    output = str(item.get("output", "")).strip()
    if not output:
        raise ValueError("legacy record has empty output")
    user = "\n\n".join(x for x in (instruction, input_text) if x)
    if not user:
        raise ValueError("legacy record has no instruction/input")
    out = [{"role": "user", "content": user}, {"role": "assistant", "content": output}]
    _validate_dialogue(out)
    return out


def _to_list(x: Any) -> List[int]:
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        if len(x) != 1:
            raise ValueError("expected one conversation")
        x = x[0]
    return list(x)


def _apply_chat_template(tokenizer: Any, messages: Sequence[Dict[str, str]], **kwargs: Any) -> Any:
    try:
        return tokenizer.apply_chat_template(list(messages), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(list(messages), **kwargs)


def encode_response_only(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    max_length: int = 4096,
    truncation: str = "keep_end",
    enable_thinking: bool = False,
) -> Dict[str, List[int]]:
    """Encode chat data while training assistant tokens only.

    Preferred path uses the tokenizer's assistant mask. A prefix-stable fallback
    is retained for templates that do not expose assistant masks.
    """
    if not messages:
        raise ValueError("empty messages")

    try:
        rendered = _apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            enable_thinking=enable_thinking,
        )
        if isinstance(rendered, dict) and "input_ids" in rendered:
            ids = _to_list(rendered["input_ids"])
            mask = rendered.get("assistant_masks", rendered.get("assistant_tokens_mask"))
            if mask is not None:
                mask = _to_list(mask)
                if len(mask) == len(ids) and any(mask):
                    labels = [tok if int(m) else IGNORE_INDEX for tok, m in zip(ids, mask)]
                    return _truncate(ids, labels, max_length, truncation)
    except Exception:
        pass

    full_ids = _to_list(
        _apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    )
    labels = [IGNORE_INDEX] * len(full_ids)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        before = _to_list(
            _apply_chat_template(
                tokenizer,
                messages[:i],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
        through = _to_list(
            _apply_chat_template(
                tokenizer,
                messages[: i + 1],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
        )
        if through != full_ids[: len(through)]:
            raise ValueError("chat template is not prefix-stable")
        if before != through[: len(before)]:
            raise ValueError("generation prefix is not aligned with assistant turn")
        for pos in range(len(before), len(through)):
            labels[pos] = full_ids[pos]
    if all(x == IGNORE_INDEX for x in labels):
        raise ValueError("no assistant tokens were marked trainable")
    return _truncate(full_ids, labels, max_length, truncation)


def _truncate(ids: List[int], labels: List[int], max_length: int, truncation: str) -> Dict[str, List[int]]:
    if max_length and len(ids) > max_length:
        if truncation == "drop":
            raise OverflowError(f"encoded length {len(ids)} exceeds max_length={max_length}")
        if truncation == "keep_start":
            ids, labels = ids[:max_length], labels[:max_length]
        elif truncation == "keep_end":
            ids, labels = ids[-max_length:], labels[-max_length:]
        else:
            raise ValueError("truncation must be keep_end, keep_start, or drop")
    if all(x == IGNORE_INDEX for x in labels):
        raise ValueError("truncation removed all assistant tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


@dataclass
class ResponseOnlyCollator:
    pad_token_id: int
    label_pad_token_id: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        if not features:
            raise ValueError("empty batch")
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        input_ids, masks, labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            masks.append(list(f.get("attention_mask", [1] * n)) + [0] * pad)
            labels.append(list(f["labels"]) + [self.label_pad_token_id] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def prompt_messages_and_reference(item: Dict[str, Any]) -> tuple[List[Dict[str, str]], str]:
    messages = normalize_messages(item)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            return messages[:i], messages[i]["content"]
    raise ValueError("record has no assistant reference")
