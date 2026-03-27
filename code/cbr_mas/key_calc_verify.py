"""Verify KEY_CALC blocks in model output (hard constraint for revision path)."""

from __future__ import annotations

import math
import os
import re

from cbr_mas.safe_math import safe_eval_arithmetic


def extract_key_calc_block(text: str) -> str | None:
    """
    Accept common variants:
    - KEY_CALC:\\n...\\n####
    - KEY_CALC: first line on same line as header (no newline after colon)
    """
    if not text:
        return None
    # Looser: allow optional blank lines after colon; body until #### or EOF
    m = re.search(
        r"(?is)KEY_CALC\s*:\s*(?:\n\s*)*(?P<body>.*?)(?=\n\s*####|\Z)",
        text,
    )
    if not m:
        return None
    body = m.group("body").strip()
    return body or None


def _normalize_calc_line(ln: str) -> str:
    s = ln.strip()
    if not s or s.startswith("#"):
        return ""
    s = s.strip("`").strip()
    s = re.sub(r"^[-*•]\s+", "", s)
    return s.strip()


def _normalize_expression(expr: str) -> str:
    s = expr.strip().replace(",", "")
    s = s.replace("×", "*").replace("÷", "/")
    return s


def verify_key_calc(
    text: str,
    min_lines: int | None = None,
    max_lines: int | None = None,
) -> tuple[bool, str]:
    """
    Each kept line must be: <arithmetic expr> = <number>
    All equalities must hold numerically (tolerant float compare).

    Defaults can be overridden with env KEY_CALC_MIN_LINES / KEY_CALC_MAX_LINES.
    """
    min_l = min_lines if min_lines is not None else int(os.environ.get("KEY_CALC_MIN_LINES", "1"))
    max_l = max_lines if max_lines is not None else int(os.environ.get("KEY_CALC_MAX_LINES", "10"))
    min_l = max(1, min_l)
    max_l = max(min_l, max_l)

    body = extract_key_calc_block(text)
    if body is None:
        return False, "missing KEY_CALC block"
    raw_lines = body.splitlines()
    lines = []
    for ln in raw_lines:
        n = _normalize_calc_line(ln)
        if n:
            lines.append(n)
    if len(lines) < min_l:
        return False, f"KEY_CALC line count {len(lines)} < min {min_l}"
    if len(lines) > max_l:
        lines = lines[:max_l]
    for i, ln in enumerate(lines, start=1):
        if "=" not in ln:
            return False, f"line {i}: no '='"
        left, right = ln.rsplit("=", 1)
        left = _normalize_expression(left)
        right = right.strip().replace(",", "")
        if not left or not right:
            return False, f"line {i}: empty side"
        try:
            ev = safe_eval_arithmetic(left)
            rv = float(right)
        except Exception as e:
            return False, f"line {i}: parse/eval error ({e})"
        if not math.isclose(float(ev), rv, rel_tol=0.0, abs_tol=1e-6):
            return False, f"line {i}: {left} -> {ev} != {rv}"
    return True, "ok"
