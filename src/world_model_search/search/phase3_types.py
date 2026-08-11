"""Frozen Phase 3 condition, budget, proposal, and identity records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from world_model_search.domain.types import Candidate
from world_model_search.dsl.ast import BitExpr
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.json_schema import ast_to_value
from world_model_search.dsl.versions import (
    PHASE3_BUDGET_VERSION,
    PHASE3_CANDIDATE_IDENTITY_VERSION,
)
from world_model_search.serialization import JsonObject, sha256_json


class SearchCondition(StrEnum):
    INCUMBENT = "single-incumbent-v1"
    DIVERSE = "uniform-diverse-archive-v1"


@dataclass(frozen=True, slots=True)
class BudgetState:
    proposal_attempt_cap: int
    oracle_call_cap: int
    proposal_attempts: int = 0
    operator_attempts: int = 0
    invalid_outputs: int = 0
    noop_outputs: int = 0
    parsed_proposals: int = 0
    type_valid_proposals: int = 0
    canonical_proposals: int = 0
    semantically_distinct_proposals: int = 0
    canonical_duplicates: int = 0
    semantic_duplicates: int = 0
    oracle_invocations: int = 0
    oracle_cache_hits: int = 0
    archive_insertions: int = 0
    archive_replacements: int = 0
    archive_reserves: int = 0
    archive_duplicates: int = 0
    archive_rejections: int = 0
    scheduler_selections: int = 0
    evaluated_candidates: int = 0
    language_model_calls: int = 0
    language_model_tokens: int = 0

    @property
    def remaining_proposal_attempts(self) -> int:
        return self.proposal_attempt_cap - self.proposal_attempts

    @property
    def remaining_oracle_calls(self) -> int:
        return self.oracle_call_cap - self.oracle_invocations

    @property
    def exhausted(self) -> bool:
        return self.remaining_proposal_attempts <= 0 or self.remaining_oracle_calls <= 0

    def updated(self, **increments: int) -> BudgetState:
        values: dict[str, int] = {}
        for name, increment in increments.items():
            current = getattr(self, name)
            if not isinstance(current, int):
                raise TypeError(f"budget field is not numeric: {name}")
            values[name] = current + increment
        result = replace(self, **values)
        result.validate()
        return result

    def validate(self) -> None:
        if self.proposal_attempts > self.proposal_attempt_cap:
            raise ValueError("proposal attempt budget exceeded")
        if self.oracle_invocations > self.oracle_call_cap:
            raise ValueError("oracle call budget exceeded")
        if (
            self.oracle_cache_hits != 0
            or self.language_model_calls != 0
            or self.language_model_tokens != 0
        ):
            raise ValueError("Phase 3 forbids oracle caching and language-model use")
        if self.evaluated_candidates != self.oracle_invocations:
            raise ValueError("every Phase 3 oracle invocation evaluates exactly one candidate")
        if self.scheduler_selections != self.proposal_attempts:
            raise ValueError("every proposal attempt requires one scheduler selection")
        if self.operator_attempts > self.proposal_attempts:
            raise ValueError("operator attempts cannot exceed proposal attempts")

    def to_value(self) -> JsonObject:
        return {
            "budget_version": PHASE3_BUDGET_VERSION,
            "caps": {
                "proposal_attempts": self.proposal_attempt_cap,
                "oracle_calls": self.oracle_call_cap,
            },
            "counters": {
                "proposal_attempts": self.proposal_attempts,
                "operator_attempts": self.operator_attempts,
                "invalid_outputs": self.invalid_outputs,
                "noop_outputs": self.noop_outputs,
                "parsed_proposals": self.parsed_proposals,
                "type_valid_proposals": self.type_valid_proposals,
                "canonical_proposals": self.canonical_proposals,
                "semantically_distinct_proposals": self.semantically_distinct_proposals,
                "canonical_duplicates": self.canonical_duplicates,
                "semantic_duplicates": self.semantic_duplicates,
                "oracle_invocations": self.oracle_invocations,
                "oracle_cache_hits": self.oracle_cache_hits,
                "archive_insertions": self.archive_insertions,
                "archive_replacements": self.archive_replacements,
                "archive_reserves": self.archive_reserves,
                "archive_duplicates": self.archive_duplicates,
                "archive_rejections": self.archive_rejections,
                "scheduler_selections": self.scheduler_selections,
                "evaluated_candidates": self.evaluated_candidates,
                "language_model_calls": self.language_model_calls,
                "language_model_tokens": self.language_model_tokens,
            },
            "remaining": {
                "proposal_attempts": self.remaining_proposal_attempts,
                "oracle_calls": self.remaining_oracle_calls,
            },
            "primary_normalized_cost": {
                "quantum": "one-actual-oracle-invocation",
                "charged": self.oracle_invocations,
            },
            "diagnostics": {"cpu_seconds": None, "elapsed_seconds": None},
        }

    @classmethod
    def from_value(cls, value: object) -> BudgetState:
        if not isinstance(value, dict):
            raise ValueError("budget state must be an object")
        caps = value.get("caps")
        counters = value.get("counters")
        if value.get("budget_version") != PHASE3_BUDGET_VERSION:
            raise ValueError("unsupported budget version")
        if not isinstance(caps, dict) or not isinstance(counters, dict):
            raise ValueError("budget state caps/counters are malformed")
        expected = {
            field
            for field in cls.__dataclass_fields__
            if field not in {"proposal_attempt_cap", "oracle_call_cap"}
        }
        if set(counters) != expected:
            raise ValueError("budget counter set is malformed")
        raw_values = {
            "proposal_attempt_cap": caps.get("proposal_attempts"),
            "oracle_call_cap": caps.get("oracle_calls"),
            **counters,
        }
        if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_values.values()):
            raise ValueError("budget values must be integers")
        result = cls(**raw_values)
        result.validate()
        return result


def phase3_candidate(
    *,
    task_id: str,
    ast: BitExpr,
    parent_ids: tuple[str, ...],
    proposer_id: str,
    operator_id: str,
    context_hash: str,
    payload_hash: str,
    coding_version: str,
    semantic_hash: str,
) -> Candidate:
    canonical = canonicalize(ast)
    identity: JsonObject = {
        "candidate_identity_schema": PHASE3_CANDIDATE_IDENTITY_VERSION,
        "task_id": task_id,
        "canonical_ast": ast_to_value(canonical),
        "ordered_parent_ids": list(parent_ids),
        "proposer_id": proposer_id,
        "operator_id": operator_id,
        "public_context_hash": context_hash,
        "payload_hash": payload_hash,
        "coding_version": coding_version,
    }
    return Candidate(
        candidate_id=sha256_json(identity),
        task_id=task_id,
        ast=canonical,
        parent_ids=parent_ids,
        proposer_id=proposer_id,
        operator_id=operator_id,
        context_hash=context_hash,
        payload_hash=payload_hash,
        semantic_hash=semantic_hash,
    )
