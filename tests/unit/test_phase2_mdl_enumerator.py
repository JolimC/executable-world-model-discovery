from __future__ import annotations

import math
import random

import pytest

from world_model_search.domain.types import OracleFeedback, OracleResponseMode, OracleResult
from world_model_search.dsl.ast import Majority, Parity, TruthTable
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.interpreter import truth_table
from world_model_search.evaluation.rank import rank_result
from world_model_search.oracle.elementary import ElementaryRule
from world_model_search.oracle.residual import (
    nonnegative_integer_bits,
    residual_bits,
    two_part_description_bits,
)
from world_model_search.proposer.enumerative import EnumerationBounds, enumerate_programs


def test_residual_and_two_part_mdl_match_published_formula_for_all_eight_cases() -> None:
    candidate = TruthTable(ElementaryRule(30).ordered_semantics)
    assert encoded_length(candidate) == 19
    for errors in range(9):
        choices = math.comb(8, errors)
        independent = nonnegative_integer_bits(errors) + (
            0 if choices == 1 else math.ceil(math.log2(choices))
        )
        assert residual_bits(errors, 8) == independent
        assert two_part_description_bits(candidate, errors, 8) == 19 + independent
    assert residual_bits(3, 8, alphabet_size=4) == residual_bits(3, 8) + 6
    for invalid in ((-1, 8), (9, 8), (1, 0)):
        with pytest.raises(ValueError):
            residual_bits(*invalid)
    with pytest.raises(ValueError):
        residual_bits(True, 8)


def _result(*, errors: int, exact: bool, bits: int, runtime: int = 0) -> OracleResult:
    return OracleResult(
        type_valid=True,
        total=True,
        local_errors=errors,
        local_cases=8,
        rollout_pass=exact,
        exact=exact,
        ast_bits=bits,
        residual_bits=residual_bits(errors, 8),
        runtime_ns=runtime,
        response=OracleFeedback(mode=OracleResponseMode.SCORE_ONLY),
    )


def test_rank_is_correctness_first_on_boundaries_and_seeded_cases() -> None:
    assert rank_result(_result(errors=0, exact=True, bits=100)) > rank_result(
        _result(errors=1, exact=False, bits=1)
    )
    assert rank_result(_result(errors=0, exact=True, bits=8, runtime=1000)) > rank_result(
        _result(errors=0, exact=True, bits=9, runtime=1)
    )
    assert rank_result(
        _result(errors=0, exact=True, bits=8, runtime=1000), include_diagnostic_runtime=True
    ) < rank_result(
        _result(errors=0, exact=True, bits=8, runtime=1), include_diagnostic_runtime=True
    )
    for seed in range(256):
        rng = random.Random(seed)
        fewer = rng.randrange(8)
        more = rng.randrange(fewer + 1, 9)
        assert rank_result(_result(errors=fewer, exact=fewer == 0, bits=1000)) > rank_result(
            _result(errors=more, exact=False, bits=1)
        )


def test_enumerator_is_deterministic_monotone_canonical_and_target_blind() -> None:
    bounds = EnumerationBounds(max_bits=36, max_depth=8, max_nodes=15, max_candidates=50_000)
    first = enumerate_programs(bounds)
    second = enumerate_programs(bounds)
    first_snapshot = tuple(
        (program.discovery_index, program.ast, program.ast_bits, program.semantic_hash)
        for program in first.programs
    )
    second_snapshot = tuple(
        (program.discovery_index, program.ast, program.ast_bits, program.semantic_hash)
        for program in second.programs
    )
    assert first_snapshot == second_snapshot
    assert first.candidates_examined == second.candidates_examined == 29_529
    assert len(first.programs) == 256
    assert [program.ast_bits for program in first.programs] == sorted(
        program.ast_bits for program in first.programs
    )
    assert all(canonicalize(program.ast) == program.ast for program in first.programs)
    assert len({program.semantic_hash for program in first.programs}) == 256
    assert first.canonical_duplicates > 0 and first.semantic_duplicates > 0
    assert not first.stopped_at_candidate_bound
    for target in (
        ElementaryRule(90).ordered_semantics,
        ElementaryRule(150).ordered_semantics,
        truth_table(Majority((-1, 0, 1))),
        truth_table(Parity((-1, 0, 1))),
    ):
        recovered = next(
            program for program in first.programs if program.ordered_semantics == target
        )
        assert recovered.ast_bits <= 8


def test_truth_table_baseline_is_fully_charged_and_not_structurally_enumerated() -> None:
    result = enumerate_programs(
        EnumerationBounds(max_bits=20, max_depth=6, max_nodes=11, max_candidates=20_000)
    )
    assert all(not isinstance(program.ast, TruthTable) for program in result.programs)
    assert {
        encoded_length(TruthTable(ElementaryRule(number).ordered_semantics))
        for number in range(256)
    } == {19}
