"""Strict Phase 4 candidate-batch envelope and structured-output schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from world_model_search.domain.types import ProposalRole
from world_model_search.dsl.ast import AstLimits, BitExpr, TruthTable, children
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.json_schema import CandidateJsonError, DslCandidateDocument
from world_model_search.serialization import JsonObject, JsonValue, canonical_json

BATCH_SCHEMA_VERSION = 1
BATCH_SCHEMA_NAME = "world_model_candidate_batch_v1"


class BatchEnvelopeError(ValueError):
    """The response root is not the exact bounded Phase 4 envelope."""


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise BatchEnvelopeError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _nonfinite_constant(value: str) -> object:
    raise BatchEnvelopeError(f"non-finite JSON number is forbidden: {value}")


@dataclass(frozen=True, slots=True)
class BatchItem:
    ordinal: int
    submitted_document: JsonObject | None
    source_ast: BitExpr | None
    canonical_ast: BitExpr | None
    rejection_reason: str | None

    @property
    def accepted(self) -> bool:
        return self.canonical_ast is not None


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    role: ProposalRole
    items: tuple[BatchItem, ...]
    normalized_envelope: JsonObject


def _has_truth_table(expr: BitExpr) -> bool:
    return isinstance(expr, TruthTable) or any(
        isinstance(child, BitExpr) and _has_truth_table(child) for child in children(expr)
    )


def parse_candidate_batch(
    raw_text: str,
    *,
    expected_role: ProposalRole,
    requested_batch_size: int,
    limits: AstLimits,
    allowed_macros: frozenset[str],
) -> CandidateBatch:
    """Validate the envelope once and every candidate independently, preserving order."""

    if not raw_text or raw_text.lstrip().startswith("```"):
        raise BatchEnvelopeError("model response must be an unfenced JSON object")
    try:
        raw: object = json.loads(
            raw_text,
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise BatchEnvelopeError(f"invalid batch JSON: {exc.msg}") from exc
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise BatchEnvelopeError("batch must be a JSON object")
    if set(raw) != {"batch_schema_version", "role", "candidates"}:
        raise BatchEnvelopeError("batch has missing or unknown root fields")
    version = raw["batch_schema_version"]
    role = raw["role"]
    candidates = raw["candidates"]
    if isinstance(version, bool) or version != BATCH_SCHEMA_VERSION:
        raise BatchEnvelopeError("unsupported batch schema version")
    if role != expected_role.value:
        raise BatchEnvelopeError("batch role does not match the request")
    if not isinstance(candidates, list) or not candidates:
        raise BatchEnvelopeError("candidate batch must be a nonempty array")
    if len(candidates) != requested_batch_size:
        raise BatchEnvelopeError("candidate batch size does not match the request")
    items: list[BatchItem] = []
    normalized_candidates: list[JsonValue] = []
    for ordinal, submitted in enumerate(candidates):
        if not isinstance(submitted, dict) or not all(isinstance(key, str) for key in submitted):
            items.append(BatchItem(ordinal, None, None, None, "candidate item must be an object"))
            normalized_candidates.append({"invalid_item_ordinal": ordinal})
            continue
        submitted_value = cast(JsonObject, submitted)
        normalized_candidates.append(submitted_value)
        try:
            document = DslCandidateDocument.from_json(
                canonical_json(submitted_value), limits=limits, allowed_macros=allowed_macros
            )
            if _has_truth_table(document.ast):
                raise CandidateJsonError("TruthTable is reserved for the charged non-LLM baseline")
            canonical = canonicalize(document.ast)
            items.append(BatchItem(ordinal, submitted_value, document.ast, canonical, None))
        except (CandidateJsonError, TypeError, ValueError) as exc:
            items.append(BatchItem(ordinal, submitted_value, None, None, str(exc)))
    envelope = cast(
        JsonObject,
        {
            "batch_schema_version": BATCH_SCHEMA_VERSION,
            "role": expected_role.value,
            "candidates": normalized_candidates,
        },
    )
    return CandidateBatch(expected_role, tuple(items), envelope)


def candidate_batch_json_schema(*, role: ProposalRole, batch_size: int) -> JsonObject:
    """Return the exact recursive JSON Schema sent through Responses `text.format`."""

    if batch_size < 1 or batch_size > 16:
        raise ValueError("Phase 4 batch size must be in [1, 16]")
    mask: dict[str, object] = {
        "type": "array",
        "items": {"type": "integer", "enum": [-1, 0, 1]},
        "minItems": 1,
        "maxItems": 3,
    }
    int_expr: dict[str, object] = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["IntConst"]},
                    "value": {"type": "integer", "minimum": -3, "maximum": 3},
                },
                "required": ["op", "value"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"op": {"type": "string", "enum": ["Count"]}, "mask": mask},
                "required": ["op", "mask"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["AddConst"]},
                    "expr": {"$ref": "#/$defs/int_expr"},
                    "amount": {"type": "integer", "minimum": -3, "maximum": 3},
                },
                "required": ["op", "expr", "amount"],
                "additionalProperties": False,
            },
        ]
    }
    pred_expr: dict[str, object] = {
        "anyOf": [
            *[
                {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": [op]},
                        "left": {"$ref": "#/$defs/int_expr"},
                        "right": {"$ref": "#/$defs/int_expr"},
                    },
                    "required": ["op", "left", "right"],
                    "additionalProperties": False,
                }
                for op in ("Eq", "Le", "Ge")
            ],
            {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["Between"]},
                    "value": {"$ref": "#/$defs/int_expr"},
                    "lower": {"$ref": "#/$defs/int_expr"},
                    "upper": {"$ref": "#/$defs/int_expr"},
                },
                "required": ["op", "value", "lower", "upper"],
                "additionalProperties": False,
            },
        ]
    }
    bit_variants: list[dict[str, object]] = [
        {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["Const"]},
                "value": {"type": "integer", "enum": [0, 1]},
            },
            "required": ["op", "value"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["At"]},
                "offset": {"type": "integer", "enum": [-1, 0, 1]},
            },
            "required": ["op", "offset"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["Not"]},
                "expr": {"$ref": "#/$defs/bit_expr"},
            },
            "required": ["op", "expr"],
            "additionalProperties": False,
        },
        *[
            {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": [op]},
                    "left": {"$ref": "#/$defs/bit_expr"},
                    "right": {"$ref": "#/$defs/bit_expr"},
                },
                "required": ["op", "left", "right"],
                "additionalProperties": False,
            }
            for op in ("And", "Or", "Xor")
        ],
        {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["If"]},
                "condition": {"$ref": "#/$defs/pred_expr"},
                "then": {"$ref": "#/$defs/bit_expr"},
                "else": {"$ref": "#/$defs/bit_expr"},
            },
            "required": ["op", "condition", "then", "else"],
            "additionalProperties": False,
        },
        *[
            {
                "type": "object",
                "properties": {"op": {"type": "string", "enum": [op]}, "mask": mask},
                "required": ["op", "mask"],
                "additionalProperties": False,
            }
            for op in ("Parity", "Majority")
        ],
    ]
    candidate: dict[str, object] = {
        "type": "object",
        "properties": {
            "candidate_schema_version": {"type": "integer", "enum": [1]},
            "dsl_version": {"type": "string", "enum": ["binary-ca-radius1-dsl-v1"]},
            "ast": {"$ref": "#/$defs/bit_expr"},
        },
        "required": ["candidate_schema_version", "dsl_version", "ast"],
        "additionalProperties": False,
    }
    return cast(
        JsonObject,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "batch_schema_version": {"type": "integer", "enum": [BATCH_SCHEMA_VERSION]},
                "role": {"type": "string", "enum": [role.value]},
                "candidates": {
                    "type": "array",
                    "items": candidate,
                    "minItems": batch_size,
                    "maxItems": batch_size,
                },
            },
            "required": ["batch_schema_version", "role", "candidates"],
            "additionalProperties": False,
            "$defs": {
                "bit_expr": {"anyOf": bit_variants},
                "int_expr": int_expr,
                "pred_expr": pred_expr,
            },
        },
    )
