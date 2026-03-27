#!/usr/bin/env python3
"""
Analyze experiment outputs saved by run_experiments.py.

Key metrics for "ours":
  - flip_good_to_bad: Generator correct but final wrong (harmful revisions)
  - flip_bad_to_good: Generator wrong but final correct (helpful revisions)

Requires that per-example entries include:
  - gold
  - generator_pred (ours only, added in updated pipelines.py)
  - judge_pred (ours only, added in updated pipelines.py)
If these fields are missing (older runs), re-run ours to collect them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def norm(x: str | None) -> str | None:
    if x is None:
        return None
    return str(x).strip().replace(",", "").replace("$", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/run.json"))
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    per = data.get("per_example", {})
    ours = per.get("ours")
    if not ours:
        raise SystemExit("No 'ours' found in per_example.")

    missing = 0
    flip_good_to_bad = 0
    flip_bad_to_good = 0
    changed_answer = 0
    total = 0

    for e in ours:
        total += 1
        gold = norm(e.get("gold"))
        gp = norm(e.get("generator_pred"))
        jp = norm(e.get("judge_pred"))
        if gold is None or gp is None or jp is None:
            missing += 1
            continue
        gen_ok = gp == gold
        fin_ok = (norm(e.get("pred")) or jp) == gold  # fin pred may be gated to generator
        if gp != jp:
            changed_answer += 1
        if gen_ok and not fin_ok:
            flip_good_to_bad += 1
        if (not gen_ok) and fin_ok:
            flip_bad_to_good += 1

    if missing > 0:
        raise SystemExit(
            f"Missing generator_pred/judge_pred in {missing}/{total} examples. "
            "Re-run: python run_experiments.py --methods ours --resume --output results/run.json"
        )

    print(json.dumps(
        {
            "total": total,
            "changed_answer_count": changed_answer,
            "flip_good_to_bad": flip_good_to_bad,
            "flip_bad_to_good": flip_bad_to_good,
            "flip_good_to_bad_rate": flip_good_to_bad / total if total else 0.0,
            "flip_bad_to_good_rate": flip_bad_to_good / total if total else 0.0,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()

