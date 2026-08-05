"""Exact Phase 1 elementary-cellular-automaton semantics and verification."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

from world_model_search.serialization import sha256_json

SIMULATOR_VERSION = "elementary-ca-scalar-vector-v1"
ORACLE_VERSION = "elementary-exact-v1"
ROLLOUT_VERSION = "elementary-rollout-independent-v1"
SEMANTIC_HASH_VERSION = "elementary-local-semantics-v1"


@dataclass(frozen=True, slots=True)
class ElementaryRule:
    """A rule number using Wolfram numbering: bit 4L+2C+R is the output."""

    number: int

    def __post_init__(self) -> None:
        if not 0 <= self.number <= 255:
            raise ValueError("elementary rule number must be in [0, 255]")

    def output(self, left: int, center: int, right: int) -> int:
        if (left, center, right) not in tuple(
            (a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)
        ):
            raise ValueError("neighborhood cells must be binary")
        return (self.number >> (4 * left + 2 * center + right)) & 1

    @property
    def ordered_semantics(self) -> tuple[int, ...]:
        """Outputs for neighborhoods 000,001,...,111."""

        return tuple((self.number >> index) & 1 for index in range(8))

    @property
    def semantic_hash(self) -> str:
        return sha256_json(
            {"domain": SEMANTIC_HASH_VERSION, "ordered_000_to_111": self.ordered_semantics}
        )


RULE_30 = ElementaryRule(30)
RULE_90 = ElementaryRule(90)
RULE_110 = ElementaryRule(110)
RULE_150 = ElementaryRule(150)


def scalar_step(rule: ElementaryRule, state: tuple[int, ...]) -> tuple[int, ...]:
    """Readable reference update using explicit modular indexing."""

    if not state or any(cell not in (0, 1) for cell in state):
        raise ValueError("state must be a nonempty binary tuple")
    size = len(state)
    result: list[int] = []
    for index in range(size):
        result.append(
            rule.output(state[(index - 1) % size], state[index], state[(index + 1) % size])
        )
    return tuple(result)


def vectorized_step(rule: ElementaryRule, state: tuple[int, ...]) -> tuple[int, ...]:
    """Independent bulk implementation based on rotated sequences and a lookup table."""

    if not state or any(cell not in (0, 1) for cell in state):
        raise ValueError("state must be a nonempty binary tuple")
    left = state[-1:] + state[:-1]
    right = state[1:] + state[:1]
    table = rule.ordered_semantics
    return tuple(table[(a << 2) | (b << 1) | c] for a, b, c in zip(left, state, right, strict=True))


def rollout(
    rule: ElementaryRule, initial: tuple[int, ...], horizon: int, *, vectorized: bool = False
) -> tuple[tuple[int, ...], ...]:
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    update = vectorized_step if vectorized else scalar_step
    states = [initial]
    for _ in range(horizon):
        states.append(update(rule, states[-1]))
    return tuple(states)


def local_errors(candidate: ElementaryRule, reference: ElementaryRule) -> tuple[int, ...]:
    return tuple(
        i for i in range(8) if candidate.ordered_semantics[i] != reference.ordered_semantics[i]
    )


def independent_rollout_matches(
    candidate: ElementaryRule, reference_states: Iterable[tuple[int, ...]]
) -> bool:
    """Verify a locked trajectory without calling either simulator or local equivalence."""

    states = tuple(reference_states)
    for before, expected in pairwise(states):
        produced = tuple(
            (
                candidate.number
                >> ((before[i - 1] << 2) + (before[i] << 1) + before[(i + 1) % len(before)])
            )
            & 1
            for i in range(len(before))
        )
        if produced != expected:
            return False
    return True
