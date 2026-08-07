"""Immutable, loop-free typed AST for the Phase 2 binary radius-1 DSL."""

from __future__ import annotations

from dataclasses import dataclass

NEIGHBORHOOD_OFFSETS = (-1, 0, 1)
MIN_SMALL_INT = -3
MAX_SMALL_INT = 3
DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_NODES = 63
DEFAULT_MAX_CASES = 8

type Mask = tuple[int, ...]


class BitExpr:
    """Marker base for expressions that produce one binary cell."""


class IntExpr:
    """Marker base for bounded, total integer expressions."""


class PredExpr:
    """Marker base for Boolean conditions accepted by ``If``."""


def _small_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{location} must be an integer")
    if not MIN_SMALL_INT <= value <= MAX_SMALL_INT:
        raise ValueError(f"{location} must be in [{MIN_SMALL_INT}, {MAX_SMALL_INT}]")
    return value


def normalize_mask(mask: object) -> Mask:
    """Validate the one canonical mask form: nonempty, sorted, unique offsets."""

    if not isinstance(mask, tuple):
        raise TypeError("mask must be a tuple")
    if not mask:
        raise ValueError("mask must not be empty")
    if any(isinstance(offset, bool) or not isinstance(offset, int) for offset in mask):
        raise TypeError("mask offsets must be integers")
    if tuple(sorted(set(mask))) != mask:
        raise ValueError("mask offsets must be sorted and unique")
    if any(offset not in NEIGHBORHOOD_OFFSETS for offset in mask):
        raise ValueError("mask offsets must be drawn from (-1, 0, 1)")
    return mask


@dataclass(frozen=True, slots=True)
class Const(BitExpr):
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or self.value not in (0, 1):
            raise ValueError("Const value must be integer 0 or 1")


@dataclass(frozen=True, slots=True)
class At(BitExpr):
    offset: int

    def __post_init__(self) -> None:
        if isinstance(self.offset, bool) or self.offset not in NEIGHBORHOOD_OFFSETS:
            raise ValueError("At offset must be -1, 0, or 1")


@dataclass(frozen=True, slots=True)
class Not(BitExpr):
    expr: BitExpr

    def __post_init__(self) -> None:
        if not isinstance(self.expr, BitExpr):
            raise TypeError("Not expr must be BitExpr")


@dataclass(frozen=True, slots=True)
class And(BitExpr):
    left: BitExpr
    right: BitExpr

    def __post_init__(self) -> None:
        if not isinstance(self.left, BitExpr) or not isinstance(self.right, BitExpr):
            raise TypeError("And children must be BitExpr")


@dataclass(frozen=True, slots=True)
class Or(BitExpr):
    left: BitExpr
    right: BitExpr

    def __post_init__(self) -> None:
        if not isinstance(self.left, BitExpr) or not isinstance(self.right, BitExpr):
            raise TypeError("Or children must be BitExpr")


@dataclass(frozen=True, slots=True)
class Xor(BitExpr):
    left: BitExpr
    right: BitExpr

    def __post_init__(self) -> None:
        if not isinstance(self.left, BitExpr) or not isinstance(self.right, BitExpr):
            raise TypeError("Xor children must be BitExpr")


@dataclass(frozen=True, slots=True)
class If(BitExpr):
    condition: PredExpr
    then_branch: BitExpr
    else_branch: BitExpr

    def __post_init__(self) -> None:
        if not isinstance(self.condition, PredExpr):
            raise TypeError("If condition must be PredExpr")
        if not isinstance(self.then_branch, BitExpr) or not isinstance(self.else_branch, BitExpr):
            raise TypeError("If branches must be BitExpr")


@dataclass(frozen=True, slots=True)
class Parity(BitExpr):
    mask: Mask

    def __post_init__(self) -> None:
        normalize_mask(self.mask)


@dataclass(frozen=True, slots=True)
class Majority(BitExpr):
    mask: Mask

    def __post_init__(self) -> None:
        normalize_mask(self.mask)


@dataclass(frozen=True, slots=True)
class TruthTable(BitExpr):
    """Fully charged baseline constructor; structured enumeration excludes it."""

    outputs: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.outputs) != 8 or any(
            isinstance(value, bool) or value not in (0, 1) for value in self.outputs
        ):
            raise ValueError("TruthTable requires eight integer bits ordered 000 through 111")


@dataclass(frozen=True, slots=True)
class IntConst(IntExpr):
    value: int

    def __post_init__(self) -> None:
        _small_int(self.value, "IntConst value")


@dataclass(frozen=True, slots=True)
class Count(IntExpr):
    mask: Mask

    def __post_init__(self) -> None:
        normalize_mask(self.mask)


@dataclass(frozen=True, slots=True)
class AddConst(IntExpr):
    expr: IntExpr
    amount: int

    def __post_init__(self) -> None:
        if not isinstance(self.expr, IntExpr):
            raise TypeError("AddConst expr must be IntExpr")
        _small_int(self.amount, "AddConst amount")


@dataclass(frozen=True, slots=True)
class Eq(PredExpr):
    left: IntExpr
    right: IntExpr

    def __post_init__(self) -> None:
        if not isinstance(self.left, IntExpr) or not isinstance(self.right, IntExpr):
            raise TypeError("Eq children must be IntExpr")


@dataclass(frozen=True, slots=True)
class Le(PredExpr):
    left: IntExpr
    right: IntExpr

    def __post_init__(self) -> None:
        if not isinstance(self.left, IntExpr) or not isinstance(self.right, IntExpr):
            raise TypeError("Le children must be IntExpr")


@dataclass(frozen=True, slots=True)
class Ge(PredExpr):
    left: IntExpr
    right: IntExpr

    def __post_init__(self) -> None:
        if not isinstance(self.left, IntExpr) or not isinstance(self.right, IntExpr):
            raise TypeError("Ge children must be IntExpr")


@dataclass(frozen=True, slots=True)
class Between(PredExpr):
    value: IntExpr
    lower: IntExpr
    upper: IntExpr

    def __post_init__(self) -> None:
        if not all(isinstance(item, IntExpr) for item in (self.value, self.lower, self.upper)):
            raise TypeError("Between children must be IntExpr")


type Expr = BitExpr | IntExpr | PredExpr


@dataclass(frozen=True, slots=True)
class AstLimits:
    max_depth: int = DEFAULT_MAX_DEPTH
    max_nodes: int = DEFAULT_MAX_NODES
    max_cases: int = DEFAULT_MAX_CASES

    def __post_init__(self) -> None:
        for name, value in (
            ("max_depth", self.max_depth),
            ("max_nodes", self.max_nodes),
            ("max_cases", self.max_cases),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_cases > 8:
            raise ValueError("Phase 2 binary radius-1 mechanics have at most eight cases")


DEFAULT_LIMITS = AstLimits()


def children(expr: Expr) -> tuple[Expr, ...]:
    if isinstance(expr, Not):
        return (expr.expr,)
    if isinstance(expr, And | Or | Xor):
        return (expr.left, expr.right)
    if isinstance(expr, If):
        return (expr.condition, expr.then_branch, expr.else_branch)
    if isinstance(expr, AddConst):
        return (expr.expr,)
    if isinstance(expr, Eq | Le | Ge):
        return (expr.left, expr.right)
    if isinstance(expr, Between):
        return (expr.value, expr.lower, expr.upper)
    return ()


def ast_size(expr: Expr) -> tuple[int, int]:
    """Return ``(node_count, depth)`` with leaf depth one."""

    child_sizes = tuple(ast_size(child) for child in children(expr))
    if not child_sizes:
        return (1, 1)
    return (1 + sum(size[0] for size in child_sizes), 1 + max(size[1] for size in child_sizes))


def validate_ast(expr: Expr, limits: AstLimits = DEFAULT_LIMITS) -> None:
    """Audit runtime types and structural bounds for an already constructed AST."""

    if not isinstance(expr, BitExpr | IntExpr | PredExpr):
        raise TypeError("candidate root is not a DSL expression")
    nodes, depth = ast_size(expr)
    if nodes > limits.max_nodes:
        raise ValueError(f"AST node count {nodes} exceeds limit {limits.max_nodes}")
    if depth > limits.max_depth:
        raise ValueError(f"AST depth {depth} exceeds limit {limits.max_depth}")
