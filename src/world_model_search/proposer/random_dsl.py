"""Target-blind deterministic random DSL generation baseline."""

from __future__ import annotations

import random
from collections.abc import Sequence

from world_model_search.domain.types import ProposalBudget, ProposalContext
from world_model_search.dsl.ast import (
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
    Xor,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.json_schema import DslCandidateDocument
from world_model_search.serialization import derive_seed

RANDOM_DSL_VERSION = "target-blind-random-dsl-v1"


def _mask(rng: random.Random) -> tuple[int, ...]:
    values = tuple(offset for offset in (-1, 0, 1) if rng.randrange(2))
    return values or (rng.choice((-1, 0, 1)),)


def _int_expr(rng: random.Random, depth: int) -> IntExpr:
    if depth <= 1 or rng.randrange(3) < 2:
        return IntConst(rng.randint(-3, 3)) if rng.randrange(2) else Count(_mask(rng))
    return AddConst(_int_expr(rng, depth - 1), rng.randint(-3, 3))


def _pred(rng: random.Random, depth: int) -> PredExpr:
    if rng.randrange(4) == 0:
        return Between(_int_expr(rng, depth), _int_expr(rng, depth), _int_expr(rng, depth))
    cls = rng.choice((Eq, Le, Ge))
    return cls(_int_expr(rng, depth), _int_expr(rng, depth))


def _bit(rng: random.Random, depth: int) -> BitExpr:
    if depth <= 1:
        return Const(rng.randrange(2)) if rng.randrange(2) else At(rng.choice((-1, 0, 1)))
    choice = rng.randrange(9)
    if choice == 0:
        return Const(rng.randrange(2))
    if choice == 1:
        return At(rng.choice((-1, 0, 1)))
    if choice == 2:
        return Not(_bit(rng, depth - 1))
    if choice in {3, 4, 5}:
        cls = (And, Or, Xor)[choice - 3]
        return cls(_bit(rng, depth - 1), _bit(rng, depth - 1))
    if choice == 6:
        return Parity(_mask(rng))
    if choice == 7:
        return Majority(_mask(rng))
    return If(
        _pred(rng, max(1, depth - 2)),
        _bit(rng, depth - 1),
        _bit(rng, depth - 1),
    )


class RandomDslProposer:
    proposer_id = "random-dsl"
    proposer_version = RANDOM_DSL_VERSION

    def propose(
        self, context: ProposalContext, budget: ProposalBudget
    ) -> Sequence[DslCandidateDocument]:
        # Public task data, parents, and all oracle data are deliberately ignored.
        del context
        return tuple(
            DslCandidateDocument(
                canonicalize(
                    _bit(
                        random.Random(
                            derive_seed(
                                budget.proposer_seed,
                                f"{RANDOM_DSL_VERSION}:{budget.start_index + ordinal}",
                            )
                        ),
                        4,
                    )
                )
            )
            for ordinal in range(budget.max_candidates)
        )
