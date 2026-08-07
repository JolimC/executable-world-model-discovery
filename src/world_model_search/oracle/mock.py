"""No-CA fixture evaluator for exercising the deterministic shell."""

from __future__ import annotations

from time import perf_counter_ns

from world_model_search.domain.types import (
    Candidate,
    OracleFeedback,
    OracleResponseMode,
    OracleResult,
    RuleExpr,
)


class MockOracle:
    oracle_id = "mock-v1"

    def __init__(self, exact_index: int) -> None:
        self._exact_index = exact_index

    def evaluate(self, candidate: Candidate) -> OracleResult:
        started = perf_counter_ns()
        if not isinstance(candidate.ast, RuleExpr):
            raise TypeError("MockOracle accepts only the Phase 0 fixture AST")
        arguments = dict(candidate.ast.arguments)
        index = arguments.get("index")
        type_valid = candidate.ast.node == "Phase0Fixture" and isinstance(index, int)
        exact = type_valid and index == self._exact_index
        elapsed = max(0, perf_counter_ns() - started)
        return OracleResult(
            type_valid=type_valid,
            total=type_valid,
            local_errors=0 if exact else 1,
            local_cases=1,
            rollout_pass=exact,
            exact=exact,
            ast_bits=len(candidate.ast.node.encode("utf-8")) * 8,
            residual_bits=0 if exact else 1,
            runtime_ns=elapsed,
            response=OracleFeedback(mode=OracleResponseMode.SCORE_ONLY),
        )
