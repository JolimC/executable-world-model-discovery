"""Correctness-first lexicographic Phase 2 candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass

from world_model_search.domain.types import OracleResult


@dataclass(frozen=True, order=True, slots=True)
class CandidateRank:
    """A directly comparable tuple where larger values are always better."""

    type_valid: int
    total: int
    negative_local_errors: int
    exact: int
    negative_ast_bits: int
    negative_runtime_ns: int


def rank_result(result: OracleResult, *, include_diagnostic_runtime: bool = False) -> CandidateRank:
    """Rank runtime only as an explicitly local, non-replay-stable final tie breaker."""

    runtime = result.runtime_ns if include_diagnostic_runtime else 0
    return CandidateRank(
        type_valid=int(result.type_valid),
        total=int(result.total),
        negative_local_errors=-result.local_errors,
        exact=int(result.exact),
        negative_ast_bits=-result.ast_bits,
        negative_runtime_ns=-runtime,
    )
