"""Safely evaluate arithmetic expressions (+ - * / // %, parentheses, unary +/-)."""

from __future__ import annotations

import ast
import operator as op


_ALLOWED_BINOPS: dict[type[ast.operator], type] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
}


def safe_eval_arithmetic(expr: str) -> float | int:
    """Evaluate a single arithmetic expression; no names, no calls, no attributes."""
    s = expr.strip().replace(",", "")
    if not s:
        raise ValueError("empty expression")
    tree = ast.parse(s, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("boolean not allowed")
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("unsupported constant")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval_node(node.operand)
        return -v
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_node(node.operand)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"operator not allowed: {op_type.__name__}")
        fn = _ALLOWED_BINOPS[op_type]
        return fn(_eval_node(node.left), _eval_node(node.right))
    raise ValueError(f"disallowed syntax: {type(node).__name__}")
