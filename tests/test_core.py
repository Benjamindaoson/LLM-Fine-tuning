from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finllm_prod import IGNORE_INDEX
from finllm_prod.data import basic_quality_filter, deterministic_split, source_id
from finllm_prod.formatting import encode_response_only, normalize_messages
from finllm_prod.metrics import number_metrics, rouge_l_f1, rouge_n_f1


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 3
    chat_template = "fake"

    def _enc(self, text):
        return [10 + (ord(ch) % 200) for ch in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False, return_assistant_tokens_mask=False, enable_thinking=False):
        text = "<B>"
        mask = []
        ids = []

        def add(piece, assistant=False):
            pids = self._enc(piece)
            ids.extend(pids)
            mask.extend([1 if assistant else 0] * len(pids))

        add("<B>")
        for m in messages:
            add(f"<{m['role']}>")
            add(m["content"], assistant=m["role"] == "assistant")
            if m["role"] == "assistant":
                add("<E>", assistant=True)
        if add_generation_prompt:
            add("<assistant>")
        if not tokenize:
            return text
        if return_dict:
            out = {"input_ids": ids}
            if return_assistant_tokens_mask:
                out["assistant_masks"] = mask
            return out
        return ids


def test_normalize_messages_legacy():
    m = normalize_messages({"instruction": "分析", "input": "材料", "output": "回答"})
    assert m[-1] == {"role": "assistant", "content": "回答"}


def test_assistant_mask_trains_all_assistant_turns():
    tok = FakeTokenizer()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    enc = encode_response_only(tok, msgs, max_length=1000)
    assert len(enc["labels"]) == len(enc["input_ids"])
    assert any(x == IGNORE_INDEX for x in enc["labels"])
    assert any(x != IGNORE_INDEX for x in enc["labels"])


def test_dialogue_structure_rejects_non_alternating_roles():
    bad = {"messages": [
        {"role": "user", "content": "Q1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A"},
    ]}
    try:
        normalize_messages(bad)
    except ValueError:
        return
    raise AssertionError("non-alternating dialogue should be rejected")


def test_source_id_and_split_stable():
    row = {"Article": "  财务 数据  ", "Summary": "摘要", "title": "标题"}
    assert source_id(row) == source_id({"Article": "财务 数据", "Summary": "摘要", "title": "标题"})
    sid = source_id(row)
    assert deterministic_split(sid, 42, .94, .01) == deterministic_split(sid, 42, .94, .01)


def test_quality_filter():
    ok, _ = basic_quality_filter("公司营收增长。" * 50, min_chars=20, max_chars=10000)
    assert ok
    ok, reason = basic_quality_filter("短", min_chars=20, max_chars=10000)
    assert not ok and reason == "too_short"


def test_financial_number_metrics():
    m = number_metrics("营收100亿元，同比增长12.5%。", "营收100亿元，同比增长12.5%。", "")
    assert m["number_f1"] == 1.0
    assert m["unsupported_number_rate"] == 0.0
    m2 = number_metrics("营收100亿元，利润99亿元。", "营收100亿元。", "")
    assert m2["unsupported_number_rate"] > 0


def test_rouge_metrics():
    assert rouge_n_f1("公司盈利增长", "公司盈利增长", 1) == 1.0
    assert rouge_l_f1("公司盈利增长", "公司盈利增长") == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
