from __future__ import annotations

import random

from world_model_search.dsl.ast import (
    And,
    At,
    Const,
    Eq,
    If,
    IntConst,
    Majority,
    Not,
    Or,
    Parity,
    TruthTable,
    Xor,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import encode
from world_model_search.dsl.interpreter import rollout as dsl_rollout
from world_model_search.dsl.interpreter import semantic_hash, truth_table
from world_model_search.dsl.json_schema import ast_canonical_json
from world_model_search.oracle.elementary import ElementaryRule, rollout


def test_explicit_canonicalization_rewrites_are_semantics_preserving() -> None:
    examples = (
        (Not(Const(1)), Const(0)),
        (And(At(0), At(0)), At(0)),
        (Or(Const(0), At(1)), At(1)),
        (Not(Not(At(-1))), At(-1)),
        (If(Eq(IntConst(1), IntConst(1)), At(-1), At(1)), At(-1)),
        (And(At(1), At(-1)), And(At(-1), At(1))),
    )
    for source, expected in examples:
        assert canonicalize(source) == expected
        assert truth_table(source) == truth_table(expected)
        assert canonicalize(canonicalize(source)) == canonicalize(source)


def test_seeded_canonicalization_properties_and_stable_identity() -> None:
    leaves = (Const(0), Const(1), At(-1), At(0), At(1), Parity((-1, 1)), Majority((-1, 0, 1)))
    for seed in range(256):
        rng = random.Random(seed)
        expr = leaves[rng.randrange(len(leaves))]
        for _ in range(4):
            other = leaves[rng.randrange(len(leaves))]
            expr = rng.choice((And(expr, other), Or(expr, other), Xor(expr, other), Not(expr)))
        canonical = canonicalize(expr)
        assert truth_table(expr) == truth_table(canonical)
        assert canonicalize(canonical) == canonical
        assert ast_canonical_json(canonical) == ast_canonical_json(canonicalize(expr))
        assert semantic_hash(expr) == semantic_hash(canonical)
        assert encode(canonical) == encode(canonicalize(expr))


def test_interpreter_matches_all_elementary_tables_hashes_and_rollouts() -> None:
    for number in range(256):
        rule = ElementaryRule(number)
        candidate = TruthTable(rule.ordered_semantics)
        assert truth_table(candidate) == rule.ordered_semantics
        assert semantic_hash(candidate) == rule.semantic_hash
        initial = tuple((number >> (index % 8)) & 1 for index in range(13))
        assert dsl_rollout(candidate, initial, 4) == rollout(rule, initial, 4)


def test_declared_short_forms_have_expected_semantics() -> None:
    assert truth_table(Xor(At(-1), At(1))) == ElementaryRule(90).ordered_semantics
    assert truth_table(Xor(Xor(At(-1), At(0)), At(1))) == ElementaryRule(150).ordered_semantics
    assert truth_table(Parity((-1, 0, 1))) == ElementaryRule(150).ordered_semantics
    assert truth_table(Majority((-1, 0, 1))) == ElementaryRule(232).ordered_semantics
