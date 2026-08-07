"""Published Phase 2 residual and two-part MDL code lengths."""

from __future__ import annotations

import math

from world_model_search.dsl.ast import BitExpr
from world_model_search.dsl.codec import encoded_length


def nonnegative_integer_bits(value: int) -> int:
    """Elias-gamma length of ``value + 1`` for a nonnegative integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("integer code input must be a nonnegative integer")
    return 2 * ((value + 1).bit_length() - 1) + 1


def residual_bits(errors: int, cases: int, *, alphabet_size: int = 2) -> int:
    """Return ``L_N(e) + ceil(log2(C(N,e)))`` plus nonbinary corrections."""

    for value, name in ((errors, "errors"), (cases, "cases"), (alphabet_size, "alphabet_size")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if cases < 0 or errors < 0 or errors > cases:
        raise ValueError("residual inputs require 0 <= errors <= cases")
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least 2")
    choices = math.comb(cases, errors)
    locations = 0 if choices == 1 else (choices - 1).bit_length()
    corrections = errors * ((alphabet_size - 2).bit_length())
    return nonnegative_integer_bits(errors) + locations + corrections


def two_part_description_bits(
    expr: BitExpr, errors: int, cases: int, *, alphabet_size: int = 2
) -> int:
    return encoded_length(expr) + residual_bits(errors, cases, alphabet_size=alphabet_size)
