"""GSM8K loading and answer extraction."""

import re
from typing import Any

from datasets import load_dataset


def load_gsm8k_splits():
    """Return (train_rows, test_rows) as lists of dicts with question, answer, gold_number."""
    ds = load_dataset("gsm8k", "main")
    train = [_row(r) for r in ds["train"]]
    test = [_row(r) for r in ds["test"]]
    return train, test


def _row(r: dict[str, Any]) -> dict[str, Any]:
    gold = extract_gold_number(r["answer"])
    return {"question": r["question"].strip(), "answer": r["answer"].strip(), "gold": gold}


def extract_gold_number(answer_field: str) -> str:
    """Final numeric answer after #### in official GSM8K format."""
    if "####" in answer_field:
        tail = answer_field.split("####")[-1].strip()
        return normalize_number(tail)
    return normalize_number(answer_field)


def extract_predicted_number(text: str) -> str | None:
    """Prefer GSM8K-style #### answer; else use the last numeric token in the text."""
    if not text:
        return None
    t = text.replace(",", "")
    if "####" in t:
        tail = t.split("####")[-1]
        m = re.search(r"-?\d+(?:\.\d+)?", tail)
        if m:
            return normalize_number(m.group(0))
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    if not nums:
        return None
    return normalize_number(nums[-1])


def normalize_number(s: str) -> str:
    s = str(s).strip().replace("$", "").replace(",", "").replace("%", "")
    s = s.split()[0] if s.split() else s
    try:
        if "." in s:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return str(f).rstrip("0").rstrip(".")
        return str(int(float(s)))
    except ValueError:
        return s.strip()


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    return normalize_number(pred) == normalize_number(gold)
