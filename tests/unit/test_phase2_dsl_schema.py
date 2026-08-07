from __future__ import annotations

import json

import pytest

from world_model_search.dsl.ast import AstLimits, At, Xor
from world_model_search.dsl.json_schema import CandidateJsonError, DslCandidateDocument


def _candidate(ast: object) -> str:
    return json.dumps(
        {
            "candidate_schema_version": 1,
            "dsl_version": "binary-ca-radius1-dsl-v1",
            "ast": ast,
        }
    )


def test_candidate_schema_round_trips_declared_external_shape() -> None:
    document = DslCandidateDocument(ast=Xor(At(-1), At(1)))
    assert DslCandidateDocument.from_json(document.to_json()) == document
    assert document.to_json() == (
        '{"ast":{"left":{"offset":-1,"op":"At"},"op":"Xor",'
        '"right":{"offset":1,"op":"At"}},"candidate_schema_version":1,'
        '"dsl_version":"binary-ca-radius1-dsl-v1"}'
    )


@pytest.mark.parametrize(
    "data",
    [
        "[]",
        _candidate({"op": "Const"}),
        _candidate({"op": "Const", "value": 0, "extra": 1}),
        _candidate({"op": "Const", "value": True}),
        _candidate({"op": "At", "offset": 2}),
        _candidate({"op": "Parity", "mask": []}),
        _candidate({"op": "Parity", "mask": [1, -1]}),
        _candidate({"op": "AddConst", "expr": {"op": "IntConst", "value": 0}, "amount": 1}),
        _candidate(
            {
                "op": "If",
                "condition": {"op": "Const", "value": 1},
                "then": {"op": "Const", "value": 1},
                "else": {"op": "Const", "value": 0},
            }
        ),
        _candidate({"op": "__import__", "value": "os"}),
        '{"candidate_schema_version":1,"candidate_schema_version":1,'
        '"dsl_version":"binary-ca-radius1-dsl-v1","ast":{"op":"Const","value":0}}',
        _candidate({"op": "Const", "value": 0}) + " trailing",
        '{"candidate_schema_version":2,"dsl_version":"binary-ca-radius1-dsl-v1",'
        '"ast":{"op":"Const","value":0}}',
        '{"candidate_schema_version":1,"dsl_version":"future","ast":{"op":"Const","value":0}}',
    ],
)
def test_malformed_candidates_fail_closed(data: str) -> None:
    with pytest.raises(CandidateJsonError):
        DslCandidateDocument.from_json(data)


def test_forbidden_macros_and_structural_limits_fail_closed() -> None:
    with pytest.raises(CandidateJsonError, match="forbidden"):
        DslCandidateDocument.from_json(
            _candidate({"op": "Parity", "mask": [-1, 0, 1]}),
            allowed_macros=frozenset(),
        )
    nested: object = {"op": "At", "offset": 0}
    for _ in range(4):
        nested = {"op": "Not", "expr": nested}
    with pytest.raises(CandidateJsonError, match="depth"):
        DslCandidateDocument.from_json(_candidate(nested), limits=AstLimits(3, 63, 8))
    wide = {
        "op": "And",
        "left": {"op": "At", "offset": -1},
        "right": {"op": "At", "offset": 1},
    }
    with pytest.raises(CandidateJsonError, match="node"):
        DslCandidateDocument.from_json(_candidate(wide), limits=AstLimits(8, 2, 8))
