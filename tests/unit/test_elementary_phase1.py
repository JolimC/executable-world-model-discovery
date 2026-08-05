from __future__ import annotations

import random

from world_model_search.oracle.elementary import (
    ElementaryRule,
    independent_rollout_matches,
    local_errors,
    rollout,
    scalar_step,
    vectorized_step,
)


def test_rule_numbering_and_named_semantics() -> None:
    assert ElementaryRule(30).ordered_semantics == (0, 1, 1, 1, 1, 0, 0, 0)
    assert ElementaryRule(90).output(1, 0, 1) == 0
    assert ElementaryRule(110).output(0, 1, 1) == 1
    assert ElementaryRule(150).output(1, 1, 1) == 1


def test_all_rules_reference_and_mutations() -> None:
    for number in range(256):
        reference = ElementaryRule(number)
        assert not local_errors(reference, reference)
        for bit in range(8):
            assert local_errors(ElementaryRule(number ^ (1 << bit)), reference) == (bit,)


def test_scalar_vector_and_local_implies_rollout_over_1000_seeded_cases() -> None:
    for seed in range(1_024):
        rng = random.Random(seed)
        rule = ElementaryRule(rng.randrange(256))
        size = rng.randrange(1, 65)
        initial = tuple(rng.randrange(2) for _ in range(size))
        horizon = rng.randrange(1, 17)
        scalar = rollout(rule, initial, horizon)
        assert scalar_step(rule, initial) == vectorized_step(rule, initial)
        assert scalar == rollout(rule, initial, horizon, vectorized=True)
        assert independent_rollout_matches(rule, scalar)
