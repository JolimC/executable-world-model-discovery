"""Exact Phase 2 oracle composition for typed DSL candidates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter_ns

from world_model_search.domain.types import (
    OracleFeedback,
    OracleResponseMode,
    OracleResult,
)
from world_model_search.dsl.ast import AstLimits, BitExpr
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.interpreter import evaluate, semantic_hash, truth_table
from world_model_search.dsl.versions import INTERPRETER_VERSION
from world_model_search.oracle.residual import residual_bits
from world_model_search.tasks import HiddenTaskBundle

EXACT_ORACLE_VERSION = "typed-elementary-exact-v1"


@dataclass(frozen=True, slots=True)
class ExactEvaluation:
    """Internal evaluation metadata plus the stable public ``OracleResult``."""

    result: OracleResult
    canonical_ast: BitExpr
    semantic_hash: str
    interpreter_version: str = INTERPRETER_VERSION


def _locked_rollout_matches(
    candidate: BitExpr, hidden: HiddenTaskBundle, limits: AstLimits
) -> bool:
    """Check each locked transition directly through candidate local semantics."""

    states = hidden.locked_rollout.states
    for before, expected in pairwise(states):
        size = len(before)
        produced = tuple(
            evaluate(
                candidate,
                (before[(index - 1) % size], before[index], before[(index + 1) % size]),
                limits=limits,
            )
            for index in range(size)
        )
        if produced != expected:
            return False
    return True


class ExactDslOracle:
    oracle_id = EXACT_ORACLE_VERSION

    def __init__(
        self,
        hidden_task: HiddenTaskBundle,
        *,
        limits: AstLimits,
        response_mode: OracleResponseMode,
    ) -> None:
        self.hidden_task = hidden_task
        self.limits = limits
        self.response_mode = response_mode

    def evaluate(self, candidate: BitExpr) -> ExactEvaluation:
        started = perf_counter_ns()
        canonical = canonicalize(candidate)
        outputs = truth_table(canonical, limits=self.limits)
        errors = tuple(
            index
            for index, (actual, expected) in enumerate(
                zip(outputs, self.hidden_task.ordered_semantics, strict=True)
            )
            if actual != expected
        )
        rollout_pass = _locked_rollout_matches(canonical, self.hidden_task, self.limits)
        bits = encoded_length(canonical)
        feedback = OracleFeedback(mode=self.response_mode)
        if self.response_mode is OracleResponseMode.DIAGNOSTIC:
            feedback = OracleFeedback(
                mode=self.response_mode,
                summary=(
                    f"local-errors:{len(errors)}",
                    f"rollout:{'pass' if rollout_pass else 'fail'}",
                ),
            )
        elif self.response_mode is OracleResponseMode.COUNTEREXAMPLE and errors:
            feedback = OracleFeedback(
                mode=self.response_mode,
                counterexample=f"ordered-neighborhood-index:{errors[0]}",
            )
        exact = not errors and rollout_pass
        elapsed = max(0, perf_counter_ns() - started)
        result = OracleResult(
            type_valid=True,
            total=True,
            local_errors=len(errors),
            local_cases=8,
            rollout_pass=rollout_pass,
            exact=exact,
            ast_bits=bits,
            residual_bits=residual_bits(len(errors), 8),
            runtime_ns=elapsed,
            response=feedback,
        )
        return ExactEvaluation(
            result=result,
            canonical_ast=canonical,
            semantic_hash=semantic_hash(canonical, limits=self.limits),
        )
