#!/usr/bin/env python3
"""Train/evaluate a lightweight selector to choose between RAG and Ours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(rag_data: dict, ours_data: dict) -> list[dict]:
    rag = {e["test_index"]: e for e in rag_data.get("per_example", {}).get("rag", [])}
    ours = {e["test_index"]: e for e in ours_data.get("per_example", {}).get("ours", [])}
    common = sorted(set(rag) & set(ours))
    rows = []
    for ti in common:
        r = rag[ti]
        o = ours[ti]
        row = {
            "test_index": ti,
            "gold": str(r.get("gold")),
            "rag_pred": str(r.get("pred")),
            "ours_pred": str(o.get("pred")),
            "rag_correct": bool(r.get("correct")),
            "ours_correct": bool(o.get("correct")),
            # label: 1 => choose ours, 0 => choose rag
            "label_choose_ours": int(bool(o.get("correct")) and not bool(r.get("correct"))),
            # numeric features
            "rag_tokens": float(r.get("tokens") or 0),
            "ours_tokens": float(o.get("tokens") or 0),
            "token_gap": float((o.get("tokens") or 0) - (r.get("tokens") or 0)),
            # boolean/categorical features from ours
            "revise_triggered": int(bool(o.get("revise_triggered"))),
            "double_judge_used": int(bool(o.get("double_judge_used"))),
            "fallback_after_calc_fail": int(bool(o.get("fallback_after_calc_fail"))),
            "judge_eq_generator": int(str(o.get("judge_pred")) == str(o.get("generator_pred"))),
            "vote_winner": str(o.get("vote_winner") or "unknown"),
            "calc_verify_ok": "ok" if o.get("calc_verify_ok") is True else ("fail" if o.get("calc_verify_ok") is False else "na"),
            "judge2_present": int(o.get("judge2_pred") is not None),
            # simple critic keyword flags
            "critic_has_calc_error": int("calc" in str(o.get("calc_verify_reason", "")).lower() or "error" in str(o.get("calc_verify_reason", "")).lower()),
        }
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_file", type=Path, default=Path("results/run.json"))
    ap.add_argument("--ours_file", type=Path, default=Path("results/run_v5_ours500_gpt4omini.json"))
    ap.add_argument("--output", type=Path, default=Path("results/selector_eval.json"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "lr"],
        help="Selector model: rf (random forest) or lr (logistic regression)",
    )
    args = ap.parse_args()

    rag_data = load_json(args.rag_file)
    ours_data = load_json(args.ours_file)
    rows = build_rows(rag_data, ours_data)
    if not rows:
        raise SystemExit("No overlapping examples between rag_file and ours_file.")

    y = np.array([r["label_choose_ours"] for r in rows], dtype=int)

    # Build feature dicts for vectorization
    feat_dicts = []
    for r in rows:
        feat_dicts.append(
            {
                "rag_tokens": r["rag_tokens"],
                "ours_tokens": r["ours_tokens"],
                "token_gap": r["token_gap"],
                "revise_triggered": r["revise_triggered"],
                "double_judge_used": r["double_judge_used"],
                "fallback_after_calc_fail": r["fallback_after_calc_fail"],
                "judge_eq_generator": r["judge_eq_generator"],
                "judge2_present": r["judge2_present"],
                "critic_has_calc_error": r["critic_has_calc_error"],
                "vote_winner": r["vote_winner"],
                "calc_verify_ok": r["calc_verify_ok"],
            }
        )

    # DictVectorizer handles mixed numeric/categorical in a sparse matrix.
    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    model = Pipeline(steps=[("vec", DictVectorizer(sparse=True)), ("clf", clf)])

    # CV predictions for "deployable" estimate
    folds = max(3, int(args.folds))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    choose_ours_pred = cross_val_predict(model, feat_dicts, y, cv=cv, method="predict")

    fused_correct = []
    for r, choose_o in zip(rows, choose_ours_pred):
        use_ours = bool(choose_o)
        correct = bool(r["ours_correct"]) if use_ours else bool(r["rag_correct"])
        fused_correct.append(correct)

    rag_acc = accuracy_score([1] * len(rows), [1 if r["rag_correct"] else 0 for r in rows])
    ours_acc = accuracy_score([1] * len(rows), [1 if r["ours_correct"] else 0 for r in rows])
    fused_acc = accuracy_score([1] * len(rows), [1 if x else 0 for x in fused_correct])
    oracle_acc = accuracy_score([1] * len(rows), [1 if (r["rag_correct"] or r["ours_correct"]) else 0 for r in rows])

    out = {
        "n_examples": len(rows),
        "selector_model": args.model,
        "rag_baseline_acc": rag_acc,
        "ours_baseline_acc": ours_acc,
        "selector_cv_acc": fused_acc,
        "oracle_upper_bound_acc": oracle_acc,
        "choose_ours_rate": float(np.mean(choose_ours_pred)),
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

