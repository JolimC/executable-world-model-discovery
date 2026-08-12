"""Safe, typed, exactly coded Phase 5 learned primitives.

Learned calls are zero-arity symbols whose definitions are canonical base-DSL ``BitExpr``
values.  Every call expands before canonicalization or oracle evaluation; the language has
no source-code or evaluator capability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import cast

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
    validate_ast,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import OPCODES, CodecError, decode, encode
from world_model_search.dsl.interpreter import truth_table
from world_model_search.dsl.json_schema import DslCandidateDocument, ast_to_value
from world_model_search.errors import PersistenceError
from world_model_search.persistence.artifacts import write_content_artifact
from world_model_search.phase5_versions import (
    PHASE5_PRIMITIVE_LANGUAGE_VERSION,
    PHASE5_PRIMITIVE_REGISTRY_VERSION,
)
from world_model_search.serialization import JsonObject, canonical_json, sha256_json

PRIMITIVE_ESCAPE = "11111111111"


@dataclass(frozen=True, slots=True)
class PrimitiveCall(BitExpr):
    primitive_id: str

    def __post_init__(self) -> None:
        if len(self.primitive_id) != 64 or set(self.primitive_id) - set("0123456789abcdef"):
            raise ValueError("primitive call needs a complete content identity")


def _builtin_macro_tables() -> frozenset[tuple[int, ...]]:
    masks = tuple(mask for size in range(1, 4) for mask in combinations((-1, 0, 1), size))
    return frozenset(truth_table(node) for mask in masks for node in (Parity(mask), Majority(mask)))


@dataclass(frozen=True, slots=True)
class PrimitiveDefinition:
    ast: BitExpr

    def __post_init__(self) -> None:
        validate_ast(self.ast)
        if self.ast != canonicalize(self.ast):
            raise ValueError("learned primitive definition must be a canonical base AST")
        if isinstance(self.ast, TruthTable):
            raise ValueError("TruthTable cannot be promoted as a learned primitive")
        if truth_table(self.ast) in _builtin_macro_tables():
            raise ValueError("learned primitive is semantically equivalent to a built-in macro")

    @property
    def primitive_id(self) -> str:
        return sha256_json(
            {
                "primitive_identity_version": PHASE5_PRIMITIVE_LANGUAGE_VERSION,
                "definition": ast_to_value(self.ast),
            }
        )

    @property
    def base_definition_bits(self) -> int:
        return len(encode(self.ast))

    def to_value(self) -> JsonObject:
        return {
            "primitive_id": self.primitive_id,
            "type": "BitExpr",
            "arity": 0,
            "definition": ast_to_value(self.ast),
            "base_definition_bits": self.base_definition_bits,
        }


@dataclass(frozen=True, slots=True)
class PrimitiveRegistry:
    split_registry_hash: str
    analysis_plan_hash: str
    source_evidence_ids: tuple[str, ...]
    definitions: tuple[PrimitiveDefinition, ...]

    def __post_init__(self) -> None:
        for value in (self.split_registry_hash, self.analysis_plan_hash, *self.source_evidence_ids):
            if len(value) != 64 or set(value) - set("0123456789abcdef"):
                raise ValueError("primitive registry hashes must be complete SHA-256 values")
        ordered = tuple(sorted(self.definitions, key=lambda item: item.primitive_id))
        if ordered != self.definitions or len({item.primitive_id for item in ordered}) != len(
            ordered
        ):
            raise ValueError("primitive definitions must be unique and content ordered")
        if tuple(sorted(set(self.source_evidence_ids))) != self.source_evidence_ids:
            raise ValueError("primitive source evidence must be unique and ordered")

    def to_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "registry_version": PHASE5_PRIMITIVE_REGISTRY_VERSION,
                "language_version": PHASE5_PRIMITIVE_LANGUAGE_VERSION,
                "split_registry_hash": self.split_registry_hash,
                "analysis_plan_hash": self.analysis_plan_hash,
                "source_evidence_ids": list(self.source_evidence_ids),
                "definitions": [item.to_value() for item in self.definitions],
                "definition_code": encode_library(self.definitions),
                "definition_cost_bits": library_definition_cost(self.definitions),
                "invocation_escape": PRIMITIVE_ESCAPE,
            },
        )

    @property
    def registry_hash(self) -> str:
        return sha256_json(self.to_value())

    def safe_value(self) -> JsonObject:
        """Return definitions and symbols, omitting all evaluator provenance."""

        return cast(
            JsonObject,
            {
                "language_version": PHASE5_PRIMITIVE_LANGUAGE_VERSION,
                "registry_hash": self.registry_hash,
                "definition_cost_bits": library_definition_cost(self.definitions),
                "primitives": [
                    {
                        "primitive_id": definition.primitive_id,
                        "invocation_index": index,
                        "type": "BitExpr",
                        "arity": 0,
                        "expansion": ast_to_value(definition.ast),
                    }
                    for index, definition in enumerate(self.definitions, 1)
                ],
            },
        )

    def definition(self, primitive_id: str) -> PrimitiveDefinition:
        matches = [item for item in self.definitions if item.primitive_id == primitive_id]
        if len(matches) != 1:
            raise ValueError("primitive call is absent from the frozen registry")
        return matches[0]


def empty_primitive_registry(
    split_registry_hash: str, analysis_plan_hash: str
) -> PrimitiveRegistry:
    return PrimitiveRegistry(split_registry_hash, analysis_plan_hash, (), ())


def gamma_encode(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("gamma code accepts positive integers")
    binary = format(value, "b")
    return "0" * (len(binary) - 1) + binary


@dataclass(slots=True)
class _Reader:
    bits: str
    position: int = 0

    def take(self, count: int) -> str:
        end = self.position + count
        if end > len(self.bits):
            raise CodecError("truncated learned-primitive bitstream")
        value = self.bits[self.position : end]
        self.position = end
        return value

    def gamma(self) -> int:
        zeros = 0
        while self.position < len(self.bits) and self.bits[self.position] == "0":
            zeros += 1
            self.position += 1
        if self.position >= len(self.bits):
            raise CodecError("truncated gamma code")
        binary = self.take(zeros + 1)
        return int(binary, 2)

    def opcode(self) -> str:
        codes = {**{code: name for name, code in OPCODES.items()}, PRIMITIVE_ESCAPE: "Call"}
        for length in range(1, max(map(len, codes)) + 1):
            prefix = self.bits[self.position : self.position + length]
            if prefix in codes:
                self.position += length
                return codes[prefix]
        raise CodecError("invalid learned-primitive opcode")


def encode_library(definitions: tuple[PrimitiveDefinition, ...]) -> str:
    """Encode a counted list of length-delimited, prefix-coded base definitions."""

    ordered = tuple(sorted(definitions, key=lambda item: item.primitive_id))
    if ordered != definitions:
        raise ValueError("library definitions must be content ordered")
    result = gamma_encode(len(definitions) + 1)
    for definition in definitions:
        bits = encode(definition.ast)
        result += gamma_encode(len(bits)) + bits
    return result


def decode_library(bits: str) -> tuple[PrimitiveDefinition, ...]:
    if not bits or set(bits) - {"0", "1"}:
        raise CodecError("primitive library code must contain bits")
    reader = _Reader(bits)
    count = reader.gamma() - 1
    definitions: list[PrimitiveDefinition] = []
    for _ in range(count):
        length = reader.gamma()
        definitions.append(PrimitiveDefinition(decode(reader.take(length))))
    result = tuple(definitions)
    if (
        reader.position != len(bits)
        or tuple(sorted(result, key=lambda item: item.primitive_id)) != result
    ):
        raise CodecError("primitive library code is extended or noncanonical")
    if encode_library(result) != bits:
        raise CodecError("primitive library does not re-encode identically")
    return result


def library_definition_cost(definitions: tuple[PrimitiveDefinition, ...]) -> int:
    return len(encode_library(definitions))


def expand_primitives(expr: BitExpr, registry: PrimitiveRegistry) -> BitExpr:
    if isinstance(expr, PrimitiveCall):
        return registry.definition(expr.primitive_id).ast
    if isinstance(expr, Const | At | Parity | Majority | TruthTable):
        return expr
    if isinstance(expr, Not):
        return Not(expand_primitives(expr.expr, registry))
    if isinstance(expr, And):
        return And(expand_primitives(expr.left, registry), expand_primitives(expr.right, registry))
    if isinstance(expr, Or):
        return Or(expand_primitives(expr.left, registry), expand_primitives(expr.right, registry))
    if isinstance(expr, Xor):
        return Xor(expand_primitives(expr.left, registry), expand_primitives(expr.right, registry))
    if isinstance(expr, If):
        return If(
            expr.condition,
            expand_primitives(expr.then_branch, registry),
            expand_primitives(expr.else_branch, registry),
        )
    raise TypeError(f"unsupported extended BitExpr: {type(expr).__name__}")


def replace_subtree(expr: BitExpr, target: BitExpr, primitive_id: str) -> BitExpr:
    """Hygienically replace exact typed subtrees; no variables or capture exist."""

    if expr == target:
        return PrimitiveCall(primitive_id)
    if isinstance(expr, Const | At | Parity | Majority | TruthTable | PrimitiveCall):
        return expr
    if isinstance(expr, Not):
        return Not(replace_subtree(expr.expr, target, primitive_id))
    if isinstance(expr, And):
        return And(
            replace_subtree(expr.left, target, primitive_id),
            replace_subtree(expr.right, target, primitive_id),
        )
    if isinstance(expr, Or):
        return Or(
            replace_subtree(expr.left, target, primitive_id),
            replace_subtree(expr.right, target, primitive_id),
        )
    if isinstance(expr, Xor):
        return Xor(
            replace_subtree(expr.left, target, primitive_id),
            replace_subtree(expr.right, target, primitive_id),
        )
    if isinstance(expr, If):
        return If(
            expr.condition,
            replace_subtree(expr.then_branch, target, primitive_id),
            replace_subtree(expr.else_branch, target, primitive_id),
        )
    raise TypeError(f"unsupported BitExpr for replacement: {type(expr).__name__}")


def _offset_bits(offset: int) -> str:
    return {-1: "00", 0: "01", 1: "10"}[offset]


def _mask_bits(mask: tuple[int, ...]) -> str:
    return "".join("1" if offset in mask else "0" for offset in (-1, 0, 1))


def _small_int_bits(value: int) -> str:
    if not MIN_SMALL_INT <= value <= MAX_SMALL_INT:
        raise CodecError("small integer is outside the coding range")
    return format(value - MIN_SMALL_INT, "03b")


def _encode_extended(expr: BitExpr | IntExpr | PredExpr, registry: PrimitiveRegistry) -> str:
    if isinstance(expr, PrimitiveCall):
        ids = [item.primitive_id for item in registry.definitions]
        try:
            index = ids.index(expr.primitive_id) + 1
        except ValueError as exc:
            raise CodecError("call is absent from the primitive registry") from exc
        return PRIMITIVE_ESCAPE + gamma_encode(index)
    opcode = OPCODES[type(expr).__name__]
    if isinstance(expr, Const):
        return opcode + str(expr.value)
    if isinstance(expr, At):
        return opcode + _offset_bits(expr.offset)
    if isinstance(expr, Not):
        return opcode + _encode_extended(expr.expr, registry)
    if isinstance(expr, And | Or | Xor):
        return (
            opcode + _encode_extended(expr.left, registry) + _encode_extended(expr.right, registry)
        )
    if isinstance(expr, If):
        return (
            opcode
            + _encode_extended(expr.condition, registry)
            + _encode_extended(expr.then_branch, registry)
            + _encode_extended(expr.else_branch, registry)
        )
    if isinstance(expr, Parity | Majority | Count):
        return opcode + _mask_bits(expr.mask)
    if isinstance(expr, TruthTable):
        return opcode + "".join(map(str, expr.outputs))
    if isinstance(expr, IntConst):
        return opcode + _small_int_bits(expr.value)
    if isinstance(expr, AddConst):
        return opcode + _encode_extended(expr.expr, registry) + _small_int_bits(expr.amount)
    if isinstance(expr, Eq | Le | Ge):
        return (
            opcode + _encode_extended(expr.left, registry) + _encode_extended(expr.right, registry)
        )
    if isinstance(expr, Between):
        return (
            opcode
            + _encode_extended(expr.value, registry)
            + _encode_extended(expr.lower, registry)
            + _encode_extended(expr.upper, registry)
        )
    raise CodecError(f"unsupported learned-language node: {type(expr).__name__}")


def encode_program(expr: BitExpr, registry: PrimitiveRegistry) -> str:
    expanded = expand_primitives(expr, registry)
    validate_ast(expanded)
    if canonicalize(expanded) != expanded:
        raise CodecError("learned program must expand to a canonical base AST")
    return _encode_extended(expr, registry)


def _decode_offset(reader: _Reader) -> int:
    try:
        return {"00": -1, "01": 0, "10": 1}[reader.take(2)]
    except KeyError as exc:
        raise CodecError("invalid offset code") from exc


def _decode_mask(reader: _Reader) -> tuple[int, ...]:
    bits = reader.take(3)
    mask = tuple(offset for offset, bit in zip((-1, 0, 1), bits, strict=True) if bit == "1")
    if not mask:
        raise CodecError("empty mask code is invalid")
    return mask


def _decode_small_int(reader: _Reader) -> int:
    raw = int(reader.take(3), 2)
    if raw == 7:
        raise CodecError("invalid small-integer code")
    return raw + MIN_SMALL_INT


def _decode_int(reader: _Reader, registry: PrimitiveRegistry) -> IntExpr:
    op = reader.opcode()
    if op == "IntConst":
        return IntConst(_decode_small_int(reader))
    if op == "Count":
        return Count(_decode_mask(reader))
    if op == "AddConst":
        return AddConst(_decode_int(reader, registry), _decode_small_int(reader))
    raise CodecError(f"opcode {op} is not an IntExpr")


def _decode_pred(reader: _Reader, registry: PrimitiveRegistry) -> PredExpr:
    op = reader.opcode()
    if op in {"Eq", "Le", "Ge"}:
        left, right = _decode_int(reader, registry), _decode_int(reader, registry)
        return {"Eq": Eq, "Le": Le, "Ge": Ge}[op](left, right)
    if op == "Between":
        return Between(
            _decode_int(reader, registry),
            _decode_int(reader, registry),
            _decode_int(reader, registry),
        )
    raise CodecError(f"opcode {op} is not a PredExpr")


def _decode_bit(reader: _Reader, registry: PrimitiveRegistry) -> BitExpr:
    op = reader.opcode()
    if op == "Call":
        index = reader.gamma()
        if index > len(registry.definitions):
            raise CodecError("primitive invocation index is out of range")
        return PrimitiveCall(registry.definitions[index - 1].primitive_id)
    if op == "Const":
        return Const(int(reader.take(1), 2))
    if op == "At":
        return At(_decode_offset(reader))
    if op == "Not":
        return Not(_decode_bit(reader, registry))
    if op in {"And", "Or", "Xor"}:
        left, right = _decode_bit(reader, registry), _decode_bit(reader, registry)
        return {"And": And, "Or": Or, "Xor": Xor}[op](left, right)
    if op == "If":
        return If(
            _decode_pred(reader, registry),
            _decode_bit(reader, registry),
            _decode_bit(reader, registry),
        )
    if op == "Parity":
        return Parity(_decode_mask(reader))
    if op == "Majority":
        return Majority(_decode_mask(reader))
    if op == "TruthTable":
        return TruthTable(tuple(int(bit) for bit in reader.take(8)))
    raise CodecError(f"opcode {op} is not a BitExpr")


def decode_program(bits: str, registry: PrimitiveRegistry) -> BitExpr:
    if not bits or set(bits) - {"0", "1"}:
        raise CodecError("learned program code must contain bits")
    reader = _Reader(bits)
    try:
        expr = _decode_bit(reader, registry)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CodecError):
            raise
        raise CodecError(str(exc)) from exc
    if reader.position != len(bits):
        raise CodecError("trailing bits after learned program")
    if encode_program(expr, registry) != bits:
        raise CodecError("learned program does not re-encode identically")
    return expr


def write_primitive_registry(path: Path, registry: PrimitiveRegistry) -> str:
    artifact: JsonObject = {**registry.to_value(), "registry_hash": registry.registry_hash}
    return write_content_artifact(path, canonical_json(artifact))


def load_primitive_registry(path: Path) -> PrimitiveRegistry:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceError("primitive registry artifact is unavailable or corrupt") from exc
    if (
        not isinstance(value, dict)
        or value.get("registry_version") != PHASE5_PRIMITIVE_REGISTRY_VERSION
    ):
        raise PersistenceError("primitive registry version is unsupported")
    definitions_raw = value.get("definitions")
    evidence_raw = value.get("source_evidence_ids")
    if not isinstance(definitions_raw, list) or not isinstance(evidence_raw, list):
        raise PersistenceError("primitive registry fields are malformed")
    try:
        definitions = tuple(
            PrimitiveDefinition(
                DslCandidateDocument.from_json(
                    canonical_json(
                        {
                            "candidate_schema_version": 1,
                            "dsl_version": "binary-ca-radius1-dsl-v1",
                            "ast": item["definition"],
                        }
                    )
                ).ast
            )
            for item in definitions_raw
            if isinstance(item, dict)
        )
        registry = PrimitiveRegistry(
            str(value["split_registry_hash"]),
            str(value["analysis_plan_hash"]),
            tuple(str(item) for item in evidence_raw),
            definitions,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistenceError("primitive registry definitions are invalid") from exc
    if (
        len(definitions) != len(definitions_raw)
        or registry.registry_hash != value.get("registry_hash")
        or encode_library(definitions) != value.get("definition_code")
    ):
        raise PersistenceError("primitive registry content identity mismatch")
    return registry
