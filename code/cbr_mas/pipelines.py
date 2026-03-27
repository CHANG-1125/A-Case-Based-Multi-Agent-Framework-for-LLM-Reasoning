"""Baseline and full CBR+MAS pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

from cbr_mas import agents
from cbr_mas.config import LLMConfig, RetrievalConfig
from cbr_mas.gsm8k_utils import extract_predicted_number
from cbr_mas.key_calc_verify import verify_key_calc
from cbr_mas.llm_client import ChatLLM
from cbr_mas.retrieval import CaseBase


@dataclass
class Trace:
    method: str
    retrieved_titles: list[str] = field(default_factory=list)
    generator_out: str = ""
    critic_out: str = ""
    judge_out: str = ""
    final_text: str = ""
    generator_pred: str | None = None
    judge_pred: str | None = None
    pred: str | None = None
    total_tokens: int = 0
    calc_verify_ok: bool | None = None
    calc_verify_reason: str = ""
    fallback_after_calc_fail: bool = False
    revise_triggered: bool = True
    vote_winner: str = ""
    judge2_pred: str | None = None
    double_judge_used: bool = False


def _add_tokens(acc: int, t: int | None) -> int:
    return acc + (t or 0)


def _critic_says_correct(critic_text: str) -> bool:
    c = (critic_text or "").lower()
    return any(
        key in c
        for key in (
            "fully correct",
            "is correct",
            "looks correct",
            "no error",
            "no errors",
            "no issues",
            "correct as written",
        )
    )


def pipeline_zeroshot(
    llm: ChatLLM,
    cfg: LLMConfig,
    question: str,
) -> Trace:
    tr = Trace(method="zeroshot")
    text, tok = llm.complete(
        agents.GEN_SYSTEM,
        agents.run_zeroshot_user(question),
        temperature=cfg.generator_temperature,
    )
    tr.total_tokens = _add_tokens(0, tok)
    tr.generator_out = text
    tr.final_text = text
    tr.pred = extract_predicted_number(text)
    return tr


def pipeline_rag(
    llm: ChatLLM,
    llm_cfg: LLMConfig,
    case_base: CaseBase,
    ret_cfg: RetrievalConfig,
    question: str,
) -> Trace:
    tr = Trace(method="rag")
    hits = case_base.retrieve(question, ret_cfg.top_k)
    tr.retrieved_titles = [h[0]["question"][:80] + "..." for h in hits]
    few = agents.format_few_shot([(c, s) for c, s in hits], question)
    text, tok = agents.run_generator(llm, llm_cfg, question, few)
    tr.total_tokens = _add_tokens(0, tok)
    tr.generator_out = text
    tr.final_text = text
    tr.pred = extract_predicted_number(text)
    return tr


def pipeline_full(
    llm: ChatLLM,
    llm_cfg: LLMConfig,
    case_base: CaseBase,
    ret_cfg: RetrievalConfig,
    question: str,
    debate_rounds: int = 1,
    *,
    strong_constraints: bool = False,
    gate_and_vote: bool = False,
    double_judge_consensus: bool = False,
) -> Trace:
    tr = Trace(method="ours")
    hits = case_base.retrieve(question, ret_cfg.top_k)
    tr.retrieved_titles = [h[0]["question"][:80] + "..." for h in hits]
    few = agents.format_few_shot([(c, s) for c, s in hits], question)
    draft, tok = agents.run_generator(llm, llm_cfg, question, few)
    tr.total_tokens = _add_tokens(0, tok)
    tr.generator_out = draft
    tr.generator_pred = extract_predicted_number(draft)
    final_draft = draft
    critique = ""
    # Always run critic first; when gate_and_vote=True, only revise if critic flags likely issues.
    critique, ct = agents.run_critic(llm, llm_cfg, question, final_draft)
    tr.total_tokens = _add_tokens(tr.total_tokens, ct)
    tr.critic_out = critique
    critic_says_correct = _critic_says_correct(critique)

    revise_triggered = True
    if gate_and_vote and critic_says_correct:
        revise_triggered = False
    tr.revise_triggered = revise_triggered

    if revise_triggered:
        rounds = max(1, debate_rounds)
        for _ in range(rounds):
            judged, jt = agents.run_judge(
                llm,
                llm_cfg,
                question,
                final_draft,
                critique,
                strong_constraints=strong_constraints,
            )
            tr.total_tokens = _add_tokens(tr.total_tokens, jt)
            tr.judge_out = judged
            final_draft = judged
            if rounds > 1:
                critique, ct = agents.run_critic(llm, llm_cfg, question, final_draft)
                tr.total_tokens = _add_tokens(tr.total_tokens, ct)
                tr.critic_out = critique
        tr.judge_pred = extract_predicted_number(tr.judge_out)
    else:
        tr.judge_out = tr.generator_out
        tr.judge_pred = tr.generator_pred

    # Voting / fallback
    # - If gate_and_vote: keep generator when critic says draft is correct.
    # - Else original conservative behavior.
    final_from_judge = True
    if gate_and_vote:
        if critic_says_correct:
            tr.final_text = tr.generator_out
            tr.pred = tr.generator_pred
            final_from_judge = False
            tr.vote_winner = "generator_by_gate"
        elif tr.judge_pred is None and tr.generator_pred is not None:
            tr.final_text = tr.generator_out
            tr.pred = tr.generator_pred
            final_from_judge = False
            tr.vote_winner = "generator_by_judge_missing"
        elif tr.generator_pred is None and tr.judge_pred is not None:
            tr.final_text = tr.judge_out
            tr.pred = tr.judge_pred
            tr.vote_winner = "judge_by_generator_missing"
        elif tr.generator_pred == tr.judge_pred:
            tr.final_text = tr.judge_out
            tr.pred = tr.judge_pred
            tr.vote_winner = "agree"
        else:
            # critic says likely wrong; optionally require second-judge consensus before override.
            if double_judge_consensus:
                tr.double_judge_used = True
                judged2, jt2 = agents.run_judge(
                    llm,
                    llm_cfg,
                    question,
                    tr.generator_out,
                    tr.critic_out,
                    strong_constraints=strong_constraints,
                )
                tr.total_tokens = _add_tokens(tr.total_tokens, jt2)
                tr.judge2_pred = extract_predicted_number(judged2)
                if tr.judge2_pred is not None and tr.judge2_pred == tr.judge_pred:
                    tr.final_text = tr.judge_out
                    tr.pred = tr.judge_pred
                    tr.vote_winner = "judge_consensus"
                else:
                    tr.final_text = tr.generator_out
                    tr.pred = tr.generator_pred
                    final_from_judge = False
                    tr.vote_winner = "generator_by_no_judge_consensus"
            else:
                tr.final_text = tr.judge_out
                tr.pred = tr.judge_pred
                tr.vote_winner = "judge_by_critic"
    else:
        if tr.generator_pred is not None and tr.judge_pred is not None and tr.judge_pred != tr.generator_pred:
            if critic_says_correct:
                tr.final_text = tr.generator_out
                tr.pred = tr.generator_pred
                final_from_judge = False
                tr.vote_winner = "generator_by_conservative_gate"
            else:
                tr.final_text = tr.judge_out
                tr.pred = tr.judge_pred
                tr.vote_winner = "judge"
        else:
            tr.final_text = tr.judge_out
            tr.pred = tr.judge_pred
            tr.vote_winner = "agree_or_single"

    if strong_constraints and final_from_judge:
        ok, msg = verify_key_calc(tr.judge_out)
        tr.calc_verify_ok = ok
        tr.calc_verify_reason = msg
        if not ok:
            tr.final_text = tr.generator_out
            tr.pred = tr.generator_pred
            tr.fallback_after_calc_fail = True
            tr.vote_winner = "generator_by_calc_fail"
    elif strong_constraints:
        tr.calc_verify_ok = None
        tr.calc_verify_reason = "skipped (conservative gate kept generator)"
    return tr


def maybe_retain(
    case_base: CaseBase,
    question: str,
    judge_solution: str,
    gold: str,
    pred: str | None,
    match_fn,
) -> bool:
    """Append verified case if prediction matches gold (evaluation oracle)."""
    if pred is None or not match_fn(pred, gold):
        return False
    case_base.add_case(question, judge_solution)
    return True
