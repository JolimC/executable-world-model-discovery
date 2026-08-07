"""Versioned prefix code for canonical Phase 2 ASTs."""

from __future__ import annotations

from dataclasses import dataclass

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
from world_model_search.dsl.canonicalize import canonicalize


class CodecError(ValueError):
    """Raised for malformed, ill-typed, noncanonical, or extended streams."""


OPCODES = {
    "Const": "000",
    "At": "001",
    "Not": "0100",
    "And": "0101",
    "Or": "0110",
    "Xor": "0111",
    "If": "10000",
    "Parity": "10001",
    "Majority": "10010",
    "IntConst": "10011",
    "Count": "10100",
    "AddConst": "10101",
    "Eq": "10110",
    "Le": "10111",
    "Ge": "11000",
    "Between": "11001",
    "TruthTable": "11111111110",
}


def _offset_bits(offset: int) -> str:
    return {-1: "00", 0: "01", 1: "10"}[offset]


def _mask_bits(mask: tuple[int, ...]) -> str:
    return "".join("1" if offset in mask else "0" for offset in (-1, 0, 1))


def _small_int_bits(value: int) -> str:
    if not MIN_SMALL_INT <= value <= MAX_SMALL_INT:
        raise CodecError("small integer is outside the coding range")
    return format(value - MIN_SMALL_INT, "03b")


def _encode(expr: BitExpr | IntExpr | PredExpr) -> str:
    name = type(expr).__name__
    opcode = OPCODES[name]
    if isinstance(expr, Const):
        return opcode + str(expr.value)
    if isinstance(expr, At):
        return opcode + _offset_bits(expr.offset)
    if isinstance(expr, Not):
        return opcode + _encode(expr.expr)
    if isinstance(expr, And | Or | Xor):
        return opcode + _encode(expr.left) + _encode(expr.right)
    if isinstance(expr, If):
        return (
            opcode + _encode(expr.condition) + _encode(expr.then_branch) + _encode(expr.else_branch)
        )
    if isinstance(expr, Parity | Majority | Count):
        return opcode + _mask_bits(expr.mask)
    if isinstance(expr, TruthTable):
        return opcode + "".join(map(str, expr.outputs))
    if isinstance(expr, IntConst):
        return opcode + _small_int_bits(expr.value)
    if isinstance(expr, AddConst):
        return opcode + _encode(expr.expr) + _small_int_bits(expr.amount)
    if isinstance(expr, Eq | Le | Ge):
        return opcode + _encode(expr.left) + _encode(expr.right)
    if isinstance(expr, Between):
        return opcode + _encode(expr.value) + _encode(expr.lower) + _encode(expr.upper)
    raise CodecError(f"unsupported node: {name}")


def encode(expr: BitExpr) -> str:
    """Encode exactly one canonical BitExpr into its unpadded bit string."""

    if expr != canonicalize(expr):
        raise CodecError("only canonical ASTs have a primary encoding")
    return _encode(expr)


def encoded_length(expr: BitExpr) -> int:
    return len(encode(canonicalize(expr)))


def value_encoded_length(expr: BitExpr | IntExpr | PredExpr) -> int:
    """Return structural code length for a canonical typed subtree."""

    return len(_encode(expr))


@dataclass(slots=True)
class _Reader:
    bits: str
    position: int = 0

    def take(self, count: int) -> str:
        end = self.position + count
        if end > len(self.bits):
            raise CodecError("truncated AST bitstream")
        result = self.bits[self.position : end]
        self.position = end
        return result

    def opcode(self) -> str:
        for length in range(1, max(map(len, OPCODES.values())) + 1):
            prefix = self.bits[self.position : self.position + length]
            for name, code in OPCODES.items():
                if prefix == code:
                    self.position += length
                    return name
        raise CodecError("invalid or truncated opcode")


def _decode_offset(reader: _Reader) -> int:
    bits = reader.take(2)
    try:
        return {"00": -1, "01": 0, "10": 1}[bits]
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


def _decode_int(reader: _Reader) -> IntExpr:
    op = reader.opcode()
    if op == "IntConst":
        return IntConst(_decode_small_int(reader))
    if op == "Count":
        return Count(_decode_mask(reader))
    if op == "AddConst":
        return AddConst(_decode_int(reader), _decode_small_int(reader))
    raise CodecError(f"opcode {op} is not valid where IntExpr is required")


def _decode_pred(reader: _Reader) -> PredExpr:
    op = reader.opcode()
    if op in {"Eq", "Le", "Ge"}:
        left, right = _decode_int(reader), _decode_int(reader)
        return {"Eq": Eq, "Le": Le, "Ge": Ge}[op](left, right)
    if op == "Between":
        return Between(_decode_int(reader), _decode_int(reader), _decode_int(reader))
    raise CodecError(f"opcode {op} is not valid where PredExpr is required")


def _decode_bit(reader: _Reader) -> BitExpr:
    op = reader.opcode()
    if op == "Const":
        return Const(int(reader.take(1), 2))
    if op == "At":
        return At(_decode_offset(reader))
    if op == "Not":
        return Not(_decode_bit(reader))
    if op in {"And", "Or", "Xor"}:
        left, right = _decode_bit(reader), _decode_bit(reader)
        return {"And": And, "Or": Or, "Xor": Xor}[op](left, right)
    if op == "If":
        return If(_decode_pred(reader), _decode_bit(reader), _decode_bit(reader))
    if op == "Parity":
        return Parity(_decode_mask(reader))
    if op == "Majority":
        return Majority(_decode_mask(reader))
    if op == "TruthTable":
        return TruthTable(tuple(int(bit) for bit in reader.take(8)))
    raise CodecError(f"opcode {op} is not valid where BitExpr is required")


def decode(bits: str) -> BitExpr:
    """Decode exactly one canonical AST, rejecting any remaining structural data."""

    if not isinstance(bits, str) or not bits or any(bit not in "01" for bit in bits):
        raise CodecError("encoded AST must be a nonempty string of bits")
    reader = _Reader(bits)
    try:
        expr = _decode_bit(reader)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CodecError):
            raise
        raise CodecError(str(exc)) from exc
    if reader.position != len(bits):
        raise CodecError("trailing bits after complete AST")
    if expr != canonicalize(expr):
        raise CodecError("bitstream encodes a noncanonical AST")
    if _encode(expr) != bits:
        raise CodecError("bitstream does not re-encode identically")
    return expr


def opcodes_are_prefix_free() -> bool:
    codes = tuple(OPCODES.values())
    return len(codes) == len(set(codes)) and not any(
        right.startswith(left) for left in codes for right in codes if left != right
    )
