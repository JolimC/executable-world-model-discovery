"""Strict, versioned JSON data contract for Phase 2 DSL candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass

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
from world_model_search.dsl.versions import CANDIDATE_SCHEMA_VERSION, DSL_VERSION
from world_model_search.serialization import JsonObject, canonical_json

ALLOWED_MACROS = frozenset({"Parity", "Majority"})


class CandidateJsonError(ValueError):
    """Raised when candidate data violates the strict schema."""


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateJsonError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _strict_object(data: str) -> dict[str, object]:
    try:
        value: object = json.loads(data, object_pairs_hook=_object_pairs)
    except json.JSONDecodeError as exc:
        raise CandidateJsonError(f"invalid candidate JSON: {exc.msg}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CandidateJsonError("candidate document must be a JSON object")
    return value


def _fields(raw: dict[str, object], expected: set[str], location: str) -> None:
    missing = expected - raw.keys()
    unknown = raw.keys() - expected
    if missing:
        raise CandidateJsonError(f"{location} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise CandidateJsonError(f"{location} has unknown fields: {', '.join(sorted(unknown))}")


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateJsonError(f"{location} must be an integer")
    return value


def _mask(value: object, location: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CandidateJsonError(f"{location} must be an array")
    return tuple(_integer(item, f"{location} item") for item in value)


@dataclass(slots=True)
class _ParseState:
    limits: AstLimits
    allowed_macros: frozenset[str]
    nodes: int = 0

    def enter(self, depth: int) -> None:
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise CandidateJsonError("candidate exceeds the configured node limit")
        if depth > self.limits.max_depth:
            raise CandidateJsonError("candidate exceeds the configured depth limit")


def _node_object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CandidateJsonError(f"{location} must be an object")
    return value


def _parse_bit(value: object, state: _ParseState, depth: int) -> BitExpr:
    raw = _node_object(value, "BitExpr")
    state.enter(depth)
    op = raw.get("op")
    if not isinstance(op, str):
        raise CandidateJsonError("BitExpr op must be a string")
    try:
        if op == "Const":
            _fields(raw, {"op", "value"}, op)
            return Const(_integer(raw["value"], "Const.value"))
        if op == "At":
            _fields(raw, {"op", "offset"}, op)
            return At(_integer(raw["offset"], "At.offset"))
        if op == "Not":
            _fields(raw, {"op", "expr"}, op)
            return Not(_parse_bit(raw["expr"], state, depth + 1))
        if op in {"And", "Or", "Xor"}:
            _fields(raw, {"op", "left", "right"}, op)
            left = _parse_bit(raw["left"], state, depth + 1)
            right = _parse_bit(raw["right"], state, depth + 1)
            constructors = {"And": And, "Or": Or, "Xor": Xor}
            return constructors[op](left, right)
        if op == "If":
            _fields(raw, {"op", "condition", "then", "else"}, op)
            return If(
                _parse_pred(raw["condition"], state, depth + 1),
                _parse_bit(raw["then"], state, depth + 1),
                _parse_bit(raw["else"], state, depth + 1),
            )
        if op in {"Parity", "Majority"}:
            _fields(raw, {"op", "mask"}, op)
            if op not in state.allowed_macros:
                raise CandidateJsonError(f"macro {op} is forbidden by configuration")
            mask = _mask(raw["mask"], f"{op}.mask")
            return Parity(mask) if op == "Parity" else Majority(mask)
        if op == "TruthTable":
            _fields(raw, {"op", "outputs"}, op)
            outputs_raw = raw["outputs"]
            if not isinstance(outputs_raw, list):
                raise CandidateJsonError("TruthTable.outputs must be an array")
            return TruthTable(
                tuple(_integer(item, "TruthTable.outputs item") for item in outputs_raw)
            )
    except (TypeError, ValueError) as exc:
        raise CandidateJsonError(str(exc)) from exc
    raise CandidateJsonError(f"unknown or wrong-type BitExpr opcode: {op}")


def _parse_int(value: object, state: _ParseState, depth: int) -> IntExpr:
    raw = _node_object(value, "IntExpr")
    state.enter(depth)
    op = raw.get("op")
    if not isinstance(op, str):
        raise CandidateJsonError("IntExpr op must be a string")
    try:
        if op == "IntConst":
            _fields(raw, {"op", "value"}, op)
            return IntConst(_integer(raw["value"], "IntConst.value"))
        if op == "Count":
            _fields(raw, {"op", "mask"}, op)
            return Count(_mask(raw["mask"], "Count.mask"))
        if op == "AddConst":
            _fields(raw, {"op", "expr", "amount"}, op)
            return AddConst(
                _parse_int(raw["expr"], state, depth + 1),
                _integer(raw["amount"], "AddConst.amount"),
            )
    except (TypeError, ValueError) as exc:
        raise CandidateJsonError(str(exc)) from exc
    raise CandidateJsonError(f"unknown or wrong-type IntExpr opcode: {op}")


def _parse_pred(value: object, state: _ParseState, depth: int) -> PredExpr:
    raw = _node_object(value, "PredExpr")
    state.enter(depth)
    op = raw.get("op")
    if not isinstance(op, str):
        raise CandidateJsonError("PredExpr op must be a string")
    try:
        if op in {"Eq", "Le", "Ge"}:
            _fields(raw, {"op", "left", "right"}, op)
            left = _parse_int(raw["left"], state, depth + 1)
            right = _parse_int(raw["right"], state, depth + 1)
            constructors = {"Eq": Eq, "Le": Le, "Ge": Ge}
            return constructors[op](left, right)
        if op == "Between":
            _fields(raw, {"op", "value", "lower", "upper"}, op)
            return Between(
                _parse_int(raw["value"], state, depth + 1),
                _parse_int(raw["lower"], state, depth + 1),
                _parse_int(raw["upper"], state, depth + 1),
            )
    except (TypeError, ValueError) as exc:
        raise CandidateJsonError(str(exc)) from exc
    raise CandidateJsonError(f"unknown or wrong-type PredExpr opcode: {op}")


def ast_to_value(expr: BitExpr | IntExpr | PredExpr) -> JsonObject:
    """Return the declared external AST mapping, independent of dataclass layout."""

    if isinstance(expr, Const):
        return {"op": "Const", "value": expr.value}
    if isinstance(expr, At):
        return {"op": "At", "offset": expr.offset}
    if isinstance(expr, Not):
        return {"op": "Not", "expr": ast_to_value(expr.expr)}
    if isinstance(expr, And | Or | Xor):
        return {
            "op": type(expr).__name__,
            "left": ast_to_value(expr.left),
            "right": ast_to_value(expr.right),
        }
    if isinstance(expr, If):
        return {
            "op": "If",
            "condition": ast_to_value(expr.condition),
            "then": ast_to_value(expr.then_branch),
            "else": ast_to_value(expr.else_branch),
        }
    if isinstance(expr, Parity | Majority | Count):
        return {"op": type(expr).__name__, "mask": list(expr.mask)}
    if isinstance(expr, TruthTable):
        return {"op": "TruthTable", "outputs": list(expr.outputs)}
    if isinstance(expr, IntConst):
        return {"op": "IntConst", "value": expr.value}
    if isinstance(expr, AddConst):
        return {
            "op": "AddConst",
            "expr": ast_to_value(expr.expr),
            "amount": expr.amount,
        }
    if isinstance(expr, Eq | Le | Ge):
        return {
            "op": type(expr).__name__,
            "left": ast_to_value(expr.left),
            "right": ast_to_value(expr.right),
        }
    if isinstance(expr, Between):
        return {
            "op": "Between",
            "value": ast_to_value(expr.value),
            "lower": ast_to_value(expr.lower),
            "upper": ast_to_value(expr.upper),
        }
    raise TypeError(f"unsupported AST node: {type(expr).__name__}")


@dataclass(frozen=True, slots=True)
class DslCandidateDocument:
    ast: BitExpr
    candidate_schema_version: int = CANDIDATE_SCHEMA_VERSION
    dsl_version: str = DSL_VERSION

    def to_value(self) -> JsonObject:
        return {
            "candidate_schema_version": self.candidate_schema_version,
            "dsl_version": self.dsl_version,
            "ast": ast_to_value(self.ast),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_value())

    @classmethod
    def from_json(
        cls,
        data: str,
        *,
        limits: AstLimits = DEFAULT_LIMITS,
        allowed_macros: frozenset[str] = ALLOWED_MACROS,
    ) -> DslCandidateDocument:
        raw = _strict_object(data)
        _fields(raw, {"candidate_schema_version", "dsl_version", "ast"}, "candidate")
        schema = _integer(raw["candidate_schema_version"], "candidate_schema_version")
        if schema != CANDIDATE_SCHEMA_VERSION:
            raise CandidateJsonError(f"unsupported candidate schema version: {schema}")
        version = raw["dsl_version"]
        if version != DSL_VERSION:
            raise CandidateJsonError(f"unsupported DSL version: {version!r}")
        state = _ParseState(limits=limits, allowed_macros=allowed_macros)
        ast = _parse_bit(raw["ast"], state, 1)
        validate_ast(ast, limits)
        return cls(ast=ast)


def ast_canonical_json(expr: BitExpr | IntExpr | PredExpr) -> str:
    return canonical_json(ast_to_value(expr))
