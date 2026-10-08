from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

NUM_RE = re.compile(r"(?<![\d.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\d.])")


def normalize_number(s: str) -> str:
    s = s.replace(",", "").strip()
    if s.endswith("%"):
        core = s[:-1]
        try:
            return f"{float(core):.8g}%"
        except ValueError:
            return s
    try:
        return f"{float(s):.12g}"
    except ValueError:
        return s


def extract_numbers(text: str) -> List[str]:
    return [normalize_number(m.group(0)) for m in NUM_RE.finditer(text or "")]


def _multiset_prf(pred: Sequence[str], ref: Sequence[str]) -> Tuple[float, float, float]:
    pc, rc = Counter(pred), Counter(ref)
    overlap = sum((pc & rc).values())
    p = overlap / sum(pc.values()) if pc else (1.0 if not rc else 0.0)
    r = overlap / sum(rc.values()) if rc else (1.0 if not pc else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def number_metrics(prediction: str, reference: str, prompt: str = "") -> Dict[str, float]:
    pred = extract_numbers(prediction)
    ref = extract_numbers(reference)
    precision, recall, f1 = _multiset_prf(pred, ref)
    support = Counter(extract_numbers(prompt) + ref)
    unsupported = 0
    remaining = support.copy()
    for n in pred:
        if remaining[n] > 0:
            remaining[n] -= 1
        else:
            unsupported += 1
    return {
        "number_precision": precision,
        "number_recall": recall,
        "number_f1": f1,
        "unsupported_number_rate": unsupported / len(pred) if pred else 0.0,
    }


def char_ngrams(text: str, n: int) -> List[str]:
    chars = [c for c in text if not c.isspace()]
    if len(chars) < n:
        return []
    return ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]


def rouge_n_f1(pred: str, ref: str, n: int) -> float:
    _, _, f = _multiset_prf(char_ngrams(pred, n), char_ngrams(ref, n))
    return f


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    a = [c for c in pred if not c.isspace()]
    b = [c for c in ref if not c.isspace()]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    lcs = _lcs_len(a, b)
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r) if p + r else 0.0


def aggregate_generation_metrics(rows: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = ["rouge1", "rouge2", "rougeL", "number_precision", "number_recall", "number_f1", "unsupported_number_rate", "latency_s"]
    out: Dict[str, float] = {k: 0.0 for k in keys}
    for row in rows:
        for k in keys:
            out[k] += float(row.get(k, 0.0))
    return {k: v / len(rows) for k, v in out.items()}
