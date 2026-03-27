#!/usr/bin/env python3
"""Fuse RAG + Ours predictions with guarded override rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_file", type=Path, default=Path("results/run.json"))
    ap.add_argument("--ours_file", type=Path, default=Path("results/run_v5_ours500_gpt4omini.json"))
    ap.add_argument("--output", type=Path, default=Path("results/fused_ours500.json"))
    ap.add_argument(
        "--policy",
        type=str,
        default="judge_consensus_only",
        choices=[
            "judge_consensus_only",
            "consensus_or_agree",
            "any_judge_change",
            "oracle_upper_bound",
        ],
    )
    args = ap.parse_args()

    rag = load_json(args.rag_file)
    ours = load_json(args.ours_file)
    by_rag = {e["test_index"]: e for e in rag.get("per_example", {}).get("rag", [])}
    by_ours = {e["test_index"]: e for e in ours.get("per_example", {}).get("ours", [])}
    common = sorted(set(by_rag) & set(by_ours))
    if not common:
        raise SystemExit("No common test_index between rag and ours files.")

    fused = []
    use_ours = 0
    for ti in common:
        r = by_rag[ti]
        o = by_ours[ti]
        pick_ours = False
        vw = o.get("vote_winner")
        fallback_fail = bool(o.get("fallback_after_calc_fail"))

        if args.policy == "judge_consensus_only":
            pick_ours = (vw == "judge_consensus") and (not fallback_fail)
        elif args.policy == "consensus_or_agree":
            pick_ours = (vw in {"judge_consensus", "agree"}) and (not fallback_fail)
        elif args.policy == "any_judge_change":
            pick_ours = (o.get("judge_pred") is not None) and (not fallback_fail)
        elif args.policy == "oracle_upper_bound":
            # For analysis only (uses gold labels): choose the one that is correct.
            gold = str(r.get("gold"))
            r_ok = str(r.get("pred")) == gold
            o_ok = str(o.get("pred")) == gold
            pick_ours = o_ok and (not r_ok)

        chosen = o if pick_ours else r
        use_ours += int(pick_ours)
        fused.append(
            {
                "test_index": ti,
                "gold": chosen.get("gold"),
                "pred": chosen.get("pred"),
                "correct": bool(chosen.get("correct")),
                "source": "ours" if pick_ours else "rag",
                "rag_pred": r.get("pred"),
                "ours_pred": o.get("pred"),
                "vote_winner": vw,
            }
        )

    total = len(fused)
    correct = sum(1 for e in fused if e["correct"])
    out = {
        "policy": args.policy,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "used_ours_count": use_ours,
        "used_rag_count": total - use_ours,
        "per_example": fused,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("policy", "accuracy", "correct", "total", "used_ours_count")}, indent=2))


if __name__ == "__main__":
    main()

