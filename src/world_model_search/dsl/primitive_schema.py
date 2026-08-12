"""Strict JSON parsing and structured-output schema for learned primitive calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from world_model_search.domain.types import ProposalRole
from world_model_search.dsl.ast import (
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
    Xor,
    validate_ast,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.primitives import PrimitiveCall, PrimitiveRegistry, expand_primitives
from world_model_search.model.schema import candidate_batch_json_schema
from world_model_search.serialization import JsonObject, canonical_json


class PrimitiveCandidateJsonError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PrimitiveBatchItem:
    ordinal: int
    submitted_document: JsonObject | None
    source_ast: BitExpr | None
    expanded_ast: BitExpr | None
    rejection_reason: str | None

    @property
    def accepted(self) -> bool:
        return self.expanded_ast is not None


@dataclass(frozen=True, slots=True)
class PrimitiveCandidateBatch:
    role: ProposalRole
    items: tuple[PrimitiveBatchItem, ...]
    normalized_envelope: JsonObject


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PrimitiveCandidateJsonError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _nonfinite(value: str) -> object:
    raise PrimitiveCandidateJsonError(f"non-finite JSON number is forbidden: {value}")


def _object(value: object, expected: set[str], location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PrimitiveCandidateJsonError(f"{location} must be an object")
    if set(value) != expected:
        raise PrimitiveCandidateJsonError(f"{location} has missing or unknown fields")
    return value


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrimitiveCandidateJsonError(f"{location} must be an integer")
    return value


def _mask(value: object, location: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise PrimitiveCandidateJsonError(f"{location} must be an array")
    return tuple(_integer(item, location) for item in value)


def _parse_int(value: object) -> IntExpr:
    if not isinstance(value, dict):
        raise PrimitiveCandidateJsonError("IntExpr must be an object")
    op = value.get("op")
    try:
        if op == "IntConst":
            raw = _object(value, {"op", "value"}, "IntConst")
            return IntConst(_integer(raw["value"], "IntConst.value"))
        if op == "Count":
            raw = _object(value, {"op", "mask"}, "Count")
            return Count(_mask(raw["mask"], "Count.mask"))
        if op == "AddConst":
            raw = _object(value, {"op", "expr", "amount"}, "AddConst")
            return AddConst(_parse_int(raw["expr"]), _integer(raw["amount"], "AddConst.amount"))
    except (TypeError, ValueError) as exc:
        raise PrimitiveCandidateJsonError(str(exc)) from exc
    raise PrimitiveCandidateJsonError(f"unknown IntExpr opcode: {op}")


def _parse_pred(value: object) -> PredExpr:
    if not isinstance(value, dict):
        raise PrimitiveCandidateJsonError("PredExpr must be an object")
    op = value.get("op")
    try:
        if op in {"Eq", "Le", "Ge"}:
            raw = _object(value, {"op", "left", "right"}, str(op))
            constructors = {"Eq": Eq, "Le": Le, "Ge": Ge}
            return constructors[str(op)](_parse_int(raw["left"]), _parse_int(raw["right"]))
        if op == "Between":
            raw = _object(value, {"op", "value", "lower", "upper"}, "Between")
            return Between(
                _parse_int(raw["value"]),
                _parse_int(raw["lower"]),
                _parse_int(raw["upper"]),
            )
    except (TypeError, ValueError) as exc:
        raise PrimitiveCandidateJsonError(str(exc)) from exc
    raise PrimitiveCandidateJsonError(f"unknown PredExpr opcode: {op}")


def _parse_bit(value: object, registry: PrimitiveRegistry) -> BitExpr:
    if not isinstance(value, dict):
        raise PrimitiveCandidateJsonError("BitExpr must be an object")
    op = value.get("op")
    try:
        if op == "PrimitiveCall":
            raw = _object(value, {"op", "primitive_id"}, "PrimitiveCall")
            primitive_id = raw["primitive_id"]
            if not isinstance(primitive_id, str):
                raise PrimitiveCandidateJsonError("primitive_id must be a string")
            registry.definition(primitive_id)
            return PrimitiveCall(primitive_id)
        if op == "Const":
            raw = _object(value, {"op", "value"}, "Const")
            return Const(_integer(raw["value"], "Const.value"))
        if op == "At":
            raw = _object(value, {"op", "offset"}, "At")
            return At(_integer(raw["offset"], "At.offset"))
        if op == "Not":
            raw = _object(value, {"op", "expr"}, "Not")
            return Not(_parse_bit(raw["expr"], registry))
        if op in {"And", "Or", "Xor"}:
            raw = _object(value, {"op", "left", "right"}, str(op))
            constructors = {"And": And, "Or": Or, "Xor": Xor}
            return constructors[str(op)](
                _parse_bit(raw["left"], registry),
                _parse_bit(raw["right"], registry),
            )
        if op == "If":
            raw = _object(value, {"op", "condition", "then", "else"}, "If")
            return If(
                _parse_pred(raw["condition"]),
                _parse_bit(raw["then"], registry),
                _parse_bit(raw["else"], registry),
            )
        if op in {"Parity", "Majority"}:
            raw = _object(value, {"op", "mask"}, str(op))
            mask = _mask(raw["mask"], f"{op}.mask")
            return Parity(mask) if op == "Parity" else Majority(mask)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, PrimitiveCandidateJsonError):
            raise
        raise PrimitiveCandidateJsonError(str(exc)) from exc
    raise PrimitiveCandidateJsonError(f"unknown BitExpr opcode: {op}")


@dataclass(frozen=True, slots=True)
class PrimitiveCandidateDocument:
    source_ast: BitExpr
    expanded_ast: BitExpr

    @classmethod
    def from_json(
        cls,
        data: str,
        *,
        registry: PrimitiveRegistry,
        limits: AstLimits,
    ) -> PrimitiveCandidateDocument:
        try:
            value: object = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PrimitiveCandidateJsonError("invalid primitive candidate JSON") from exc
        raw = _object(
            value,
            {"candidate_schema_version", "dsl_version", "ast"},
            "primitive candidate",
        )
        if raw["candidate_schema_version"] != 1 or raw["dsl_version"] != "binary-ca-radius1-dsl-v1":
            raise PrimitiveCandidateJsonError("candidate schema or DSL version mismatch")
        source = _parse_bit(raw["ast"], registry)
        expanded = expand_primitives(source, registry)
        validate_ast(expanded, limits)
        if canonicalize(expanded) != expanded:
            raise PrimitiveCandidateJsonError("primitive candidate expansion is noncanonical")
        return cls(source, expanded)


def parse_primitive_candidate_batch(
    raw_text: str,
    *,
    expected_role: ProposalRole,
    requested_batch_size: int,
    registry: PrimitiveRegistry,
    limits: AstLimits,
) -> PrimitiveCandidateBatch:
    """Parse the exact batch envelope while allowing only frozen primitive symbols."""

    if not raw_text or raw_text.lstrip().startswith("```"):
        raise PrimitiveCandidateJsonError("model response must be an unfenced JSON object")
    try:
        value: object = json.loads(
            raw_text,
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise PrimitiveCandidateJsonError(f"invalid batch JSON: {exc.msg}") from exc
    raw = _object(value, {"batch_schema_version", "role", "candidates"}, "candidate batch")
    if raw["batch_schema_version"] != 1 or raw["role"] != expected_role.value:
        raise PrimitiveCandidateJsonError("candidate batch version or role mismatch")
    candidates = raw["candidates"]
    if not isinstance(candidates, list) or len(candidates) != requested_batch_size:
        raise PrimitiveCandidateJsonError("candidate batch size does not match the request")
    items: list[PrimitiveBatchItem] = []
    normalized: list[object] = []
    for ordinal, submitted in enumerate(candidates):
        if not isinstance(submitted, dict) or not all(isinstance(key, str) for key in submitted):
            items.append(
                PrimitiveBatchItem(ordinal, None, None, None, "candidate item must be an object")
            )
            normalized.append({"invalid_item_ordinal": ordinal})
            continue
        document_value = cast(JsonObject, submitted)
        normalized.append(document_value)
        try:
            document = PrimitiveCandidateDocument.from_json(
                canonical_json(document_value), registry=registry, limits=limits
            )
            items.append(
                PrimitiveBatchItem(
                    ordinal, document_value, document.source_ast, document.expanded_ast, None
                )
            )
        except (PrimitiveCandidateJsonError, TypeError, ValueError) as exc:
            items.append(PrimitiveBatchItem(ordinal, document_value, None, None, str(exc)))
    envelope = cast(
        JsonObject,
        {
            "batch_schema_version": 1,
            "role": expected_role.value,
            "candidates": normalized,
        },
    )
    return PrimitiveCandidateBatch(expected_role, tuple(items), envelope)


def primitive_candidate_batch_json_schema(
    *, role: ProposalRole, batch_size: int, registry: PrimitiveRegistry
) -> JsonObject:
    schema = candidate_batch_json_schema(role=role, batch_size=batch_size)
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise AssertionError("base candidate schema has no definitions")
    bit_expr = definitions.get("bit_expr")
    if not isinstance(bit_expr, dict) or not isinstance(bit_expr.get("anyOf"), list):
        raise AssertionError("base candidate schema has no BitExpr alternatives")
    ids = [item.primitive_id for item in registry.definitions]
    if ids:
        cast(list[object], bit_expr["anyOf"]).append(
            {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["PrimitiveCall"]},
                    "primitive_id": {"type": "string", "enum": ids},
                },
                "required": ["op", "primitive_id"],
                "additionalProperties": False,
            }
        )
    return cast(JsonObject, json.loads(canonical_json(schema)))
