from __future__ import annotations

from dataclasses import dataclass

import pytest

from world_model_search.serialization import canonical_json, derive_seed, sha256_json


@dataclass(frozen=True)
class Example:
    z: int
    a: tuple[str, ...]


def test_canonical_json_has_stable_key_order_and_compact_encoding() -> None:
    assert canonical_json({"z": 2, "a": [1, True]}) == '{"a":[1,true],"z":2}'
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})
    assert canonical_json(Example(z=3, a=("x",))) == '{"a":["x"],"z":3}'


def test_non_finite_float_is_not_hashable() -> None:
    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})


def test_seed_derivation_is_deterministic_and_domain_separated() -> None:
    assert derive_seed(7, "task") == derive_seed(7, "task")
    assert derive_seed(7, "task") != derive_seed(7, "proposer")
