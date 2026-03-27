#!/usr/bin/env python3
"""
Run GSM8K experiments: Zero-shot, RAG (Retrieve+Reuse), and full CBR+MAS (Ours).

Environment:
  UniAPI (https://uniapi.ai/dashboard/key): USE_UNIAPI=1 and OPENAI_API_KEY or UNIAPI_API_KEY;
    default base https://api.uniapi.io/v1 (override with UNIAPI_BASE_URL, e.g. https://hk.uniapi.io/v1).
  Other: OPENAI_API_KEY + optional OPENAI_BASE_URL (DeepSeek, OpenRouter, etc.).
  OPENAI_MODEL         default gpt-4o-mini

Example (UniAPI):
  export USE_UNIAPI=1 OPENAI_API_KEY=<key from dashboard>
  cd /path/to/iccbr && source .venv/bin/activate   # macOS 上通常没有 python 命令，用 python3 或激活 venv
  python run_experiments.py --methods zeroshot,rag,ours --num_samples 300 --seed 42

  # 不激活 venv 时可直接：
  # .venv/bin/python run_experiments.py --methods zeroshot,rag,ours --num_samples 300

  # 全测试集 + 强约束（KEY_CALC 算术可验证，失败则回退 Generator）：
  # python run_experiments.py --full_test --methods ours --strong_constraints --output results/full_ours.json

  # 分阶段（默认 500 题；全量设 FULL_TEST=1）：bash run_full_phased.sh
  # NUM_SAMPLES=300 OUT=results/p300.json bash run_full_phased.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from tqdm import tqdm

from cbr_mas.config import get_llm_config, get_retrieval_config
from cbr_mas.gsm8k_utils import answers_match, load_gsm8k_splits
from cbr_mas.llm_client import ChatLLM
from cbr_mas.pipelines import maybe_retain, pipeline_full, pipeline_rag, pipeline_zeroshot
from cbr_mas.retrieval import CaseBase


def _write_results(
    path: Path,
    summary: dict,
    report_methods: list[str],
    all_logs: dict[str, list],
    take: int,
    wall_by_method: dict[str, float | None] | None = None,
) -> None:
    """Persist JSON; methods stats use full take when a method has take entries else partial."""
    wall_by_method = wall_by_method or {}
    methods_out: dict = {}
    for m in report_methods:
        logs = all_logs.get(m, [])
        n = len(logs)
        if n == 0:
            continue
        c = sum(1 for e in logs if e.get("correct"))
        tok = sum(int(e.get("tokens") or 0) for e in logs)
        entry = {
            "accuracy": (c / take) if n >= take else (c / n if n else 0.0),
            "correct": c,
            "total": take if n >= take else n,
            "total_tokens": tok,
            "avg_tokens_per_problem": tok / n if n else 0.0,
        }
        ws = wall_by_method.get(m)
        if ws is not None:
            entry["wall_seconds"] = ws
        if n < take:
            entry["incomplete"] = True
            entry["completed_examples"] = n
        methods_out[m] = entry
    out = {**summary, "methods": methods_out, "per_example": all_logs}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_methods(s: str) -> list[str]:
    allowed = {"zeroshot", "rag", "ours"}
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    for p in parts:
        if p not in allowed:
            raise ValueError(f"Unknown method {p}; choose from {allowed}")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description="GSM8K CBR multi-agent experiments")
    ap.add_argument(
        "--num_samples",
        type=int,
        default=300,
        help="Test subset size (ignored if --full_test)",
    )
    ap.add_argument(
        "--full_test",
        action="store_true",
        help="Evaluate on the entire GSM8K test split (~1319 examples)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--methods",
        type=str,
        default="zeroshot,rag,ours",
        help="Comma-separated: zeroshot, rag, ours",
    )
    ap.add_argument("--top_k", type=int, default=None, help="Override TOP_K env / default 3")
    ap.add_argument("--debate_rounds", type=int, default=1)
    ap.add_argument(
        "--retain",
        action="store_true",
        help="After each *ours* correct (oracle), append case to the index (sequential learning)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/run.json"),
        help="Write JSON summary and per-example logs here",
    )
    ap.add_argument(
        "--case_study_idx",
        type=int,
        default=None,
        help="If set, dump full trace for this sample index (0-based in shuffled test subset)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If --output exists with same seed/num_samples, skip examples already in per_example",
    )
    ap.add_argument(
        "--resume_extend",
        action="store_true",
        help="Allow resuming from a smaller previous num_samples if previous test order is a prefix of current order",
    )
    ap.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Write results JSON every N new examples (0=every example). Always writes when a method finishes.",
    )
    ap.add_argument(
        "--strong_constraints",
        action="store_true",
        help="Ours only: require Judge KEY_CALC equalities; on failure fall back to Generator",
    )
    ap.add_argument(
        "--gate_and_vote",
        action="store_true",
        help="Ours only: run critic gate + generator/judge fallback voting (C setting)",
    )
    ap.add_argument(
        "--double_judge_consensus",
        action="store_true",
        help="Ours only: when generator/judge disagree, require judge2==judge1 to override generator",
    )
    args = ap.parse_args()

    methods = parse_methods(args.methods)
    llm_cfg = get_llm_config()
    ret_cfg = get_retrieval_config()
    if args.top_k is not None:
        from dataclasses import replace

        ret_cfg = replace(ret_cfg, top_k=args.top_k)

    train, test = load_gsm8k_splits()
    rng = random.Random(args.seed)
    indices = list(range(len(test)))
    rng.shuffle(indices)
    if args.full_test:
        take = len(indices)
    else:
        take = min(args.num_samples, len(indices))
    chosen = indices[:take]

    strong_constraints = args.strong_constraints or os.environ.get(
        "STRONG_CONSTRAINTS", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    gate_and_vote = args.gate_and_vote or os.environ.get(
        "GATE_AND_VOTE", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    double_judge_consensus = args.double_judge_consensus or os.environ.get(
        "DOUBLE_JUDGE_CONSENSUS", ""
    ).strip().lower() in ("1", "true", "yes", "on")

    prev: dict | None = None
    if args.resume and args.output.is_file():
        prev = json.loads(args.output.read_text(encoding="utf-8"))
        if prev.get("seed") != args.seed:
            raise SystemExit("resume: existing output has different seed; use same seed or new --output")
        prev_n = int(prev.get("num_samples", 0))
        if prev_n != take:
            if not (
                args.resume_extend
                and prev_n < take
                and isinstance(prev.get("test_indices_order"), list)
                and prev.get("test_indices_order") == chosen[:prev_n]
            ):
                raise SystemExit(
                    "resume: existing output has different num_samples. "
                    "Use same args/new --output, or pass --resume_extend when extending from smaller sample."
                )
        prev_order = prev.get("test_indices_order")
        if prev_order is not None:
            if prev.get("num_samples") == take:
                if prev_order != chosen:
                    raise SystemExit("resume: test_indices_order mismatch (seed/subset changed?)")
            else:
                # resume_extend path: previous order must match current prefix
                if prev_order != chosen[: len(prev_order)]:
                    raise SystemExit("resume_extend: previous test order is not a prefix of current order")
        if args.retain and "ours" in methods and prev.get("per_example", {}).get("ours"):
            print(
                "Warning: --resume with --retain may desync the case base vs a full uninterrupted run; "
                "for strict retain semantics, rerun `ours` from scratch without resume."
            )

    llm = ChatLLM(llm_cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "num_samples": take,
        "seed": args.seed,
        "test_indices_order": chosen,
        "methods": dict(prev.get("methods", {})) if prev else {},
        "config": {
            "model": llm_cfg.model,
            "llm_base_url": llm_cfg.base_url,
            "generator_temperature": llm_cfg.generator_temperature,
            "critic_temperature": llm_cfg.critic_temperature,
            "judge_temperature": llm_cfg.judge_temperature,
            "embedding_model": ret_cfg.embedding_model,
            "top_k": ret_cfg.top_k,
            "debate_rounds": args.debate_rounds,
            "retain": args.retain,
            "checkpoint_every": args.checkpoint_every,
            "full_test": args.full_test,
            "strong_constraints": strong_constraints,
            "gate_and_vote": gate_and_vote,
            "double_judge_consensus": double_judge_consensus,
        },
    }
    prev_pe = (prev or {}).get("per_example", {})
    all_logs: dict[str, list] = {}
    for m in ("zeroshot", "rag", "ours"):
        if m not in methods and prev_pe.get(m):
            by_prev = {e["test_index"]: e for e in prev_pe[m]}
            all_logs[m] = [by_prev[ti] for ti in chosen if ti in by_prev]
    for m in methods:
        if m not in all_logs:
            all_logs[m] = []
        if prev:
            raw = prev_pe.get(m, [])
            by_prev = {e["test_index"]: e for e in raw}
            all_logs[m] = [by_prev[ti] for ti in chosen if ti in by_prev]
    pos_map = {ti: pos for pos, ti in enumerate(chosen)}
    wall_by_method: dict[str, float | None] = {}
    if prev:
        for m, stats in prev.get("methods", {}).items():
            wall_by_method[m] = stats.get("wall_seconds")
    for m in methods:
        if m not in wall_by_method:
            wall_by_method[m] = None

    def _report_methods() -> list[str]:
        return sorted(all_logs.keys())

    for method in methods:
        case_base = CaseBase(ret_cfg.embedding_model)
        case_base.build(train)
        by_ti = {e["test_index"]: e for e in all_logs[method]}
        missing = [ti for ti in chosen if ti not in by_ti]
        correct = sum(1 for e in by_ti.values() if e.get("correct"))
        tokens = sum(int(e.get("tokens") or 0) for e in by_ti.values())
        t0 = time.perf_counter()

        for idx, ti in enumerate(tqdm(missing, desc=method)):
            row = test[ti]
            q = row["question"]
            gold = row["gold"]
            pos = pos_map[ti]

            if method == "zeroshot":
                tr = pipeline_zeroshot(llm, llm_cfg, q)
            elif method == "rag":
                tr = pipeline_rag(llm, llm_cfg, case_base, ret_cfg, q)
            else:
                tr = pipeline_full(
                    llm,
                    llm_cfg,
                    case_base,
                    ret_cfg,
                    q,
                    debate_rounds=args.debate_rounds,
                    strong_constraints=strong_constraints,
                    gate_and_vote=gate_and_vote,
                    double_judge_consensus=double_judge_consensus,
                )
                if args.retain:
                    maybe_retain(
                        case_base,
                        q,
                        tr.final_text,
                        gold,
                        tr.pred,
                        answers_match,
                    )

            ok = answers_match(tr.pred, gold)
            correct += int(ok)
            tokens += tr.total_tokens or 0

            log_entry = {
                "test_index": ti,
                "gold": gold,
                "pred": tr.pred,
                "correct": ok,
                "tokens": tr.total_tokens,
            }
            if method == "ours":
                log_entry["generator_pred"] = tr.generator_pred
                log_entry["judge_pred"] = tr.judge_pred
                log_entry["calc_verify_ok"] = tr.calc_verify_ok
                log_entry["calc_verify_reason"] = tr.calc_verify_reason
                log_entry["fallback_after_calc_fail"] = tr.fallback_after_calc_fail
                log_entry["revise_triggered"] = tr.revise_triggered
                log_entry["vote_winner"] = tr.vote_winner
                log_entry["double_judge_used"] = tr.double_judge_used
                log_entry["judge2_pred"] = tr.judge2_pred
            if args.case_study_idx is not None and pos == args.case_study_idx:
                log_entry["trace"] = {
                    "retrieved": tr.retrieved_titles,
                    "generator": tr.generator_out,
                    "critic": tr.critic_out,
                    "judge": tr.judge_out,
                    "final_text": tr.final_text,
                }
            by_ti[ti] = log_entry
            all_logs[method] = [by_ti[tii] for tii in chosen if tii in by_ti]
            step = idx + 1
            if (
                args.checkpoint_every <= 0
                or step % args.checkpoint_every == 0
                or step == len(missing)
            ):
                _write_results(
                    args.output, summary, _report_methods(), all_logs, take, wall_by_method
                )

        elapsed = time.perf_counter() - t0
        all_logs[method] = [by_ti[tii] for tii in chosen]
        wall_by_method[method] = elapsed
        acc = correct / take if take else 0.0
        summary["methods"][method] = {
            "accuracy": acc,
            "correct": correct,
            "total": take,
            "total_tokens": tokens,
            "avg_tokens_per_problem": tokens / take if take else 0,
            "wall_seconds": elapsed,
        }
        _write_results(
            args.output, summary, _report_methods(), all_logs, take, wall_by_method
        )

    summary["per_example"] = all_logs
    print(json.dumps({k: summary["methods"][k] for k in sorted(summary["methods"])}, indent=2))


if __name__ == "__main__":
    main()
