"""Terminating bottom-up canonicalization independent of task semantics."""

from __future__ import annotations

from world_model_search.dsl.ast import (
    MAX_SMALL_INT,
    MIN_SMALL_INT,
    AddConst,
    And,
    At,
    Between,
    BitExpr,
    Const,
    Count,
    Eq,
    Ge,
    If,
    IntConst,
    IntExpr,
    Le,
    Majority,
    Not,
    Or,
    Parity,
    PredExpr,
    TruthTable,
    Xor,
)
from world_model_search.dsl.json_schema import ast_canonical_json


def ordering_key(expr: BitExpr | IntExpr | PredExpr) -> tuple[str, str]:
    """Declared commutative ordering: node class, then canonical JSON bytes."""

    return (type(expr).__name__, ast_canonical_json(expr))


def canonicalize_int(expr: IntExpr) -> IntExpr:
    if isinstance(expr, IntConst | Count):
        return expr
    if isinstance(expr, AddConst):
        child = canonicalize_int(expr.expr)
        if expr.amount == 0:
            return child
        if isinstance(child, IntConst):
            value = child.value + expr.amount
            if MIN_SMALL_INT <= value <= MAX_SMALL_INT:
                return IntConst(value)
        if isinstance(child, AddConst):
            amount = child.amount + expr.amount
            if MIN_SMALL_INT <= amount <= MAX_SMALL_INT:
                return canonicalize_int(AddConst(child.expr, amount))
        return AddConst(child, expr.amount)
    raise TypeError(f"unsupported IntExpr: {type(expr).__name__}")


def canonicalize_pred(expr: PredExpr) -> PredExpr:
    if isinstance(expr, Eq):
        left, right = canonicalize_int(expr.left), canonicalize_int(expr.right)
        if ordering_key(right) < ordering_key(left):
            left, right = right, left
        return Eq(left, right)
    if isinstance(expr, Le):
        return Le(canonicalize_int(expr.left), canonicalize_int(expr.right))
    if isinstance(expr, Ge):
        return Ge(canonicalize_int(expr.left), canonicalize_int(expr.right))
    if isinstance(expr, Between):
        return Between(
            canonicalize_int(expr.value),
            canonicalize_int(expr.lower),
            canonicalize_int(expr.upper),
        )
    raise TypeError(f"unsupported PredExpr: {type(expr).__name__}")


def _constant_predicate(expr: PredExpr) -> bool | None:
    if isinstance(expr, Eq | Le | Ge):
        if not isinstance(expr.left, IntConst) or not isinstance(expr.right, IntConst):
            return None
        if isinstance(expr, Eq):
            return expr.left.value == expr.right.value
        if isinstance(expr, Le):
            return expr.left.value <= expr.right.value
        return expr.left.value >= expr.right.value
    if isinstance(expr, Between) and all(
        isinstance(item, IntConst) for item in (expr.value, expr.lower, expr.upper)
    ):
        if not isinstance(expr.value, IntConst):  # pragma: no cover - narrowed by all()
            raise AssertionError
        if not isinstance(expr.lower, IntConst) or not isinstance(expr.upper, IntConst):
            raise AssertionError
        return expr.lower.value <= expr.value.value <= expr.upper.value
    return None


def _ordered(left: BitExpr, right: BitExpr) -> tuple[BitExpr, BitExpr]:
    return (right, left) if ordering_key(right) < ordering_key(left) else (left, right)


def canonicalize(expr: BitExpr) -> BitExpr:
    """Apply only the published oriented rewrites in one bottom-up traversal."""

    if isinstance(expr, Const | At | TruthTable):
        return expr
    if isinstance(expr, Parity | Majority):
        if len(expr.mask) == 1:
            return At(expr.mask[0])
        return expr
    if isinstance(expr, Not):
        child = canonicalize(expr.expr)
        if isinstance(child, Const):
            return Const(1 - child.value)
        if isinstance(child, Not):
            return child.expr
        return Not(child)
    if isinstance(expr, And):
        left, right = _ordered(canonicalize(expr.left), canonicalize(expr.right))
        if left == right:
            return left
        if isinstance(left, Const):
            return Const(0) if left.value == 0 else right
        if isinstance(right, Const):
            return Const(0) if right.value == 0 else left
        return And(left, right)
    if isinstance(expr, Or):
        left, right = _ordered(canonicalize(expr.left), canonicalize(expr.right))
        if left == right:
            return left
        if isinstance(left, Const):
            return right if left.value == 0 else Const(1)
        if isinstance(right, Const):
            return left if right.value == 0 else Const(1)
        return Or(left, right)
    if isinstance(expr, Xor):
        left, right = _ordered(canonicalize(expr.left), canonicalize(expr.right))
        if left == right:
            return Const(0)
        if isinstance(left, Const):
            return right if left.value == 0 else canonicalize(Not(right))
        if isinstance(right, Const):
            return left if right.value == 0 else canonicalize(Not(left))
        return Xor(left, right)
    if isinstance(expr, If):
        condition = canonicalize_pred(expr.condition)
        then_branch = canonicalize(expr.then_branch)
        else_branch = canonicalize(expr.else_branch)
        if then_branch == else_branch:
            return then_branch
        condition_value = _constant_predicate(condition)
        if condition_value is not None:
            return then_branch if condition_value else else_branch
        return If(condition, then_branch, else_branch)
    raise TypeError(f"unsupported BitExpr: {type(expr).__name__}")
