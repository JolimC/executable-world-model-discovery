from __future__ import annotations

import pytest

from world_model_search.dsl.ast import (
    AddConst,
    And,
    At,
    Between,
    Const,
    Count,
    Eq,
    Ge,
    If,
    IntConst,
    Le,
    Majority,
    Not,
    Or,
    Parity,
    TruthTable,
    Xor,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import (
    OPCODES,
    CodecError,
    decode,
    encode,
    opcodes_are_prefix_free,
)


def _constructor_samples() -> tuple[object, ...]:
    return (
        Const(0),
        Const(1),
        At(-1),
        At(0),
        At(1),
        If(
            Eq(AddConst(Count((-1, 0, 1)), 1), IntConst(2)),
            And(At(-1), Not(At(0))),
            Or(At(0), At(1)),
        ),
        If(Le(Count((-1,)), IntConst(0)), Xor(At(-1), At(1)), Majority((-1, 0, 1))),
        If(Ge(Count((1,)), IntConst(1)), Parity((-1, 0, 1)), TruthTable((0, 1, 1, 1, 1, 0, 0, 0))),
        If(Between(Count((-1, 0)), IntConst(-3), IntConst(3)), At(0), At(1)),
    )


def test_codec_covers_every_constructor_and_field_boundary() -> None:
    constructor_names: set[str] = set()

    def collect(node: object) -> None:
        constructor_names.add(type(node).__name__)
        for name in (
            "expr",
            "left",
            "right",
            "condition",
            "then_branch",
            "else_branch",
            "value",
            "lower",
            "upper",
        ):
            child = getattr(node, name, None)
            if not isinstance(child, int) and child is not None:
                collect(child)

    canonical_samples = tuple(canonicalize(sample) for sample in _constructor_samples())  # type: ignore[arg-type]
    for source, canonical in zip(_constructor_samples(), canonical_samples, strict=True):
        collect(source)
        bits = encode(canonical)
        assert bits == encode(canonical)
        assert decode(bits) == canonical
        assert encode(decode(bits)) == bits
    assert constructor_names == set(OPCODES)
    assert opcodes_are_prefix_free()
    codes = tuple(encode(sample) for sample in canonical_samples)
    assert all(not right.startswith(left) for left in codes for right in codes if left != right)


def test_decoder_rejects_truncation_extension_malformed_types_and_noncanonical_data() -> None:
    canonical = canonicalize(Xor(At(-1), At(1)))
    bits = encode(canonical)
    for end in range(len(bits)):
        with pytest.raises(CodecError):
            decode(bits[:end])
    with pytest.raises(CodecError, match="trailing"):
        decode(bits + "0")
    with pytest.raises(CodecError):
        decode("not-bits")
    with pytest.raises(CodecError, match="not valid"):
        decode(OPCODES["Eq"] + OPCODES["IntConst"] + "011" + OPCODES["IntConst"] + "011")
    with pytest.raises(CodecError, match="offset"):
        decode(OPCODES["At"] + "11")
    noncanonical = OPCODES["And"] + encode(At(1)) + encode(At(-1))
    with pytest.raises(CodecError, match="noncanonical"):
        decode(noncanonical)
    with pytest.raises(CodecError, match="canonical"):
        encode(And(At(1), At(-1)))
