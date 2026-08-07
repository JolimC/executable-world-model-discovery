"""Deterministic total interpreter for the finite Phase 2 AST."""

from __future__ import annotations

from world_model_search.dsl.ast import (
    DEFAULT_LIMITS,
    AddConst,
    And,
    AstLimits,
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
    validate_ast,
)
from world_model_search.dsl.versions import SEMANTIC_HASH_VERSION
from world_model_search.serialization import sha256_json

type Neighborhood = tuple[int, int, int]


def _neighborhood(value: object) -> Neighborhood:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(cell, bool) or cell not in (0, 1) for cell in value)
    ):
        raise ValueError("neighborhood must be three integer bits in left, center, right order")
    return value


def _at(neighborhood: Neighborhood, offset: int) -> int:
    return neighborhood[offset + 1]


def _eval_int(expr: IntExpr, neighborhood: Neighborhood) -> int:
    if isinstance(expr, IntConst):
        return expr.value
    if isinstance(expr, Count):
        return sum(_at(neighborhood, offset) for offset in expr.mask)
    if isinstance(expr, AddConst):
        return _eval_int(expr.expr, neighborhood) + expr.amount
    raise TypeError(f"unsupported IntExpr: {type(expr).__name__}")


def _eval_pred(expr: PredExpr, neighborhood: Neighborhood) -> bool:
    if isinstance(expr, Eq):
        return _eval_int(expr.left, neighborhood) == _eval_int(expr.right, neighborhood)
    if isinstance(expr, Le):
        return _eval_int(expr.left, neighborhood) <= _eval_int(expr.right, neighborhood)
    if isinstance(expr, Ge):
        return _eval_int(expr.left, neighborhood) >= _eval_int(expr.right, neighborhood)
    if isinstance(expr, Between):
        value = _eval_int(expr.value, neighborhood)
        return _eval_int(expr.lower, neighborhood) <= value <= _eval_int(expr.upper, neighborhood)
    raise TypeError(f"unsupported PredExpr: {type(expr).__name__}")


def _eval_bit(expr: BitExpr, neighborhood: Neighborhood) -> int:
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, At):
        return _at(neighborhood, expr.offset)
    if isinstance(expr, Not):
        return 1 - _eval_bit(expr.expr, neighborhood)
    if isinstance(expr, And):
        return _eval_bit(expr.left, neighborhood) & _eval_bit(expr.right, neighborhood)
    if isinstance(expr, Or):
        return _eval_bit(expr.left, neighborhood) | _eval_bit(expr.right, neighborhood)
    if isinstance(expr, Xor):
        return _eval_bit(expr.left, neighborhood) ^ _eval_bit(expr.right, neighborhood)
    if isinstance(expr, If):
        branch = expr.then_branch if _eval_pred(expr.condition, neighborhood) else expr.else_branch
        return _eval_bit(branch, neighborhood)
    if isinstance(expr, Parity):
        return sum(_at(neighborhood, offset) for offset in expr.mask) % 2
    if isinstance(expr, Majority):
        count = sum(_at(neighborhood, offset) for offset in expr.mask)
        return int(count >= (len(expr.mask) + 1) // 2)
    if isinstance(expr, TruthTable):
        left, center, right = neighborhood
        return expr.outputs[(left << 2) | (center << 1) | right]
    raise TypeError(f"unsupported BitExpr: {type(expr).__name__}")


def evaluate(
    expr: BitExpr,
    neighborhood: Neighborhood,
    *,
    limits: AstLimits = DEFAULT_LIMITS,
) -> int:
    """Evaluate one case after explicit finite-structure and input validation."""

    validate_ast(expr, limits)
    checked = _neighborhood(neighborhood)
    return _eval_bit(expr, checked)


def ordered_neighborhoods() -> tuple[Neighborhood, ...]:
    return tuple((left, center, right) for left in (0, 1) for center in (0, 1) for right in (0, 1))


def truth_table(expr: BitExpr, *, limits: AstLimits = DEFAULT_LIMITS) -> tuple[int, ...]:
    """Return outputs for ``000, 001, ..., 111`` with an explicit case bound."""

    cases = ordered_neighborhoods()
    if len(cases) > limits.max_cases:
        raise ValueError("exhaustive case count exceeds configured limit")
    validate_ast(expr, limits)
    return tuple(_eval_bit(expr, case) for case in cases)


def int_truth_table(expr: IntExpr) -> tuple[int, ...]:
    """Internal enumeration signature over the same eight public neighborhoods."""

    return tuple(_eval_int(expr, case) for case in ordered_neighborhoods())


def predicate_truth_table(expr: PredExpr) -> tuple[bool, ...]:
    """Internal enumeration signature over the same eight public neighborhoods."""

    return tuple(_eval_pred(expr, case) for case in ordered_neighborhoods())


def semantic_hash(expr: BitExpr, *, limits: AstLimits = DEFAULT_LIMITS) -> str:
    """Use the exact Phase 1 semantic-hash domain and ordered truth-table payload."""

    return sha256_json(
        {"domain": SEMANTIC_HASH_VERSION, "ordered_000_to_111": truth_table(expr, limits=limits)}
    )


def step(
    expr: BitExpr, state: tuple[int, ...], *, limits: AstLimits = DEFAULT_LIMITS
) -> tuple[int, ...]:
    """Apply synchronous periodic candidate semantics to one nonempty binary lattice."""

    if not state or any(isinstance(cell, bool) or cell not in (0, 1) for cell in state):
        raise ValueError("state must be a nonempty tuple of integer bits")
    validate_ast(expr, limits)
    size = len(state)
    return tuple(
        _eval_bit(expr, (state[(index - 1) % size], state[index], state[(index + 1) % size]))
        for index in range(size)
    )


def rollout(
    expr: BitExpr,
    initial: tuple[int, ...],
    horizon: int,
    *,
    limits: AstLimits = DEFAULT_LIMITS,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    states = [initial]
    for _ in range(horizon):
        states.append(step(expr, states[-1], limits=limits))
    return tuple(states)
