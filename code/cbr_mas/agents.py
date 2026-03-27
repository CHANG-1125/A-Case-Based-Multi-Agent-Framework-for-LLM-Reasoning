"""Generator, Critic, Judge prompts and rollouts."""

from cbr_mas.config import LLMConfig
from cbr_mas.llm_client import ChatLLM


GEN_SYSTEM = """You are an expert at grade-school math word problems.
Solve the problem step by step. End your response with a single final line:
#### <integer or decimal answer>
"""


CRITIC_SYSTEM = """You are a strict mathematical verifier.
Given a problem and a proposed solution, find calculation errors, logical gaps,
or unjustified claims. If the solution is fully correct, say so briefly.
Be specific about what is wrong and where."""


JUDGE_SYSTEM = """You are a careful solver of grade-school math word problems.
You receive the problem, an initial solution, and a critic's feedback.

Your job is to produce the final correct solution, but you must:
- Re-solve the problem from scratch (do NOT edit the initial solution in-place).
- Use the critic feedback only as a checklist of possible issues.
- If the critic is wrong, ignore it.

End your response with a single final line:
#### <final numeric answer>
"""

JUDGE_SYSTEM_STRONG = JUDGE_SYSTEM + """
Before the #### line, you MUST include a KEY_CALC block so your arithmetic can be machine-checked.

Copy this structure (1–8 lines of equalities; use ONLY plain ASCII operators + - * / // %):

KEY_CALC:
6 + 4 = 10
12 * 3 = 36
(100 - 25) // 5 = 15

Rules:
- Header must contain the exact text KEY_CALC: (colon). Prefer a newline after it; you may put the first equality on the same line as KEY_CALC:.
- Each line: <expression> = <number> (spaces optional around =).
- You may prefix lines with "- " or "* " (e.g. "- 2+2 = 4"); avoid other prose inside KEY_CALC.
- Use × or ÷ only if needed; ASCII * and / are preferred.
- Put your word reasoning OUTSIDE this block (above it). KEY_CALC is only for equalities.
- The final answer line must still be: #### <number>
"""


def format_few_shot(examples: list[tuple[dict, float]], target_question: str) -> str:
    parts = []
    for j, (case, _) in enumerate(examples, start=1):
        parts.append(f"Example {j}:\nQuestion: {case['question']}\nSolution:\n{case['answer']}\n")
    parts.append(f"Now solve this problem.\nQuestion: {target_question}\nSolution:")
    return "\n".join(parts)


def run_generator(
    llm: ChatLLM,
    cfg: LLMConfig,
    question: str,
    few_shot_block: str | None,
) -> tuple[str, int | None]:
    user = few_shot_block if few_shot_block else f"Question: {question}\nSolution:"
    return llm.complete(GEN_SYSTEM, user, cfg.generator_temperature)


def run_critic(
    llm: ChatLLM,
    cfg: LLMConfig,
    question: str,
    draft_solution: str,
) -> tuple[str, int | None]:
    user = (
        f"Problem:\n{question}\n\nProposed solution:\n{draft_solution}\n\n"
        "Provide your critique."
    )
    return llm.complete(CRITIC_SYSTEM, user, cfg.critic_temperature)


def run_judge(
    llm: ChatLLM,
    cfg: LLMConfig,
    question: str,
    draft_solution: str,
    critique: str,
    *,
    strong_constraints: bool = False,
) -> tuple[str, int | None]:
    system = JUDGE_SYSTEM_STRONG if strong_constraints else JUDGE_SYSTEM
    user = (
        f"Problem:\n{question}\n\n"
        f"Initial solution:\n{draft_solution}\n\n"
        f"Critic's feedback:\n{critique}\n\n"
        "Re-solve from scratch and provide the final solution."
    )
    return llm.complete(system, user, cfg.judge_temperature)


def run_zeroshot_user(question: str) -> str:
    return f"Question: {question}\nSolution:"
