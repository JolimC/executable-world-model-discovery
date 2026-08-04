"""Immutable Phase 0 contracts for tasks, proposals, evaluation, and search."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Protocol

from world_model_search.serialization import (
    JsonObject,
    canonical_json,
    parse_json_object,
    sha256_text,
)


class SplitLabel(StrEnum):
    """Immutable task split metadata; Phase 0 does not act on this label."""

    TRAINING = "training"
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class PublicDemonstration:
    """A proposer-visible observation with no oracle-only fields."""

    observation: str
    successor: str


@dataclass(frozen=True, slots=True)
class PublicTask:
    """The only task representation authorized for proposer context."""

    task_id: str
    family: str
    split: SplitLabel
    demonstrations: tuple[PublicDemonstration, ...]
    active_queries_enabled: bool
    query_budget: int


@dataclass(frozen=True, slots=True)
class Task:
    """Internal task contract, including oracle-only artifact references."""

    task_id: str
    family: str
    split: SplitLabel
    public_demonstrations: tuple[PublicDemonstration, ...]
    active_queries_enabled: bool
    query_budget: int
    exact_case_set_id: str
    rollout_suite_id: str
    public_artifact_hash: str
    hidden_artifact_id: str
    generator_version: str
    seed: int

    def public_view(self) -> PublicTask:
        """Return a capability-safe view that structurally omits hidden data."""

        return PublicTask(
            task_id=self.task_id,
            family=self.family,
            split=self.split,
            demonstrations=self.public_demonstrations,
            active_queries_enabled=self.active_queries_enabled,
            query_budget=self.query_budget,
        )


@dataclass(frozen=True, slots=True)
class RuleExpr:
    """Opaque, typed Phase 0 AST stub; the real DSL begins in Phase 2."""

    node: str
    arguments: tuple[tuple[str, str | int | bool], ...]


class ProposalRole(StrEnum):
    EXPLOIT = "exploit"
    SIMPLIFY = "simplify"
    RECOMBINE = "recombine"
    FALSIFY = "falsify"
    TRANSFER = "transfer"


@dataclass(frozen=True, slots=True)
class CandidatePayload:
    """A parsed proposer output before candidate lineage is attached."""

    ast: RuleExpr
    role: ProposalRole

    def to_json(self) -> str:
        return canonical_json(self)

    @classmethod
    def from_json(cls, data: str) -> CandidatePayload:
        raw = parse_json_object(data)
        if set(raw) != {"ast", "role"}:
            raise ValueError("candidate payload has missing or unknown fields")
        ast_raw = raw.get("ast")
        if not isinstance(ast_raw, dict):
            raise ValueError("candidate payload ast must be an object")
        if set(ast_raw) != {"node", "arguments"}:
            raise ValueError("candidate payload ast has missing or unknown fields")
        node = ast_raw.get("node")
        arguments = ast_raw.get("arguments")
        role = raw.get("role")
        if (
            not isinstance(node, str)
            or not isinstance(arguments, list)
            or not isinstance(role, str)
        ):
            raise ValueError("invalid candidate payload")
        parsed_arguments: list[tuple[str, str | int | bool]] = []
        for pair in arguments:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not isinstance(pair[1], str | int | bool)
            ):
                raise ValueError("invalid candidate AST argument")
            parsed_arguments.append((pair[0], pair[1]))
        return cls(
            ast=RuleExpr(node=node, arguments=tuple(parsed_arguments)),
            role=ProposalRole(role),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    task_id: str
    ast: RuleExpr
    parent_ids: tuple[str, ...]
    proposer_id: str
    operator_id: str
    context_hash: str
    payload_hash: str


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """Bounded candidate information safe to expose to another proposal."""

    candidate_id: str
    ast: RuleExpr


class OracleResponseMode(StrEnum):
    SCORE_ONLY = "score-only"
    DIAGNOSTIC = "diagnostic"
    COUNTEREXAMPLE = "counterexample"


@dataclass(frozen=True, slots=True)
class OracleFeedback:
    mode: OracleResponseMode
    summary: tuple[str, ...] = ()
    counterexample: str | None = None


@dataclass(frozen=True, slots=True)
class OracleResult:
    type_valid: bool
    total: bool
    local_errors: int
    local_cases: int
    rollout_pass: bool
    exact: bool
    ast_bits: int
    residual_bits: int
    runtime_ns: int
    response: OracleFeedback

    def deterministic_payload(self) -> JsonObject:
        """Return correctness data while excluding diagnostic runtime."""

        return {
            "type_valid": self.type_valid,
            "total": self.total,
            "local_errors": self.local_errors,
            "local_cases": self.local_cases,
            "rollout_pass": self.rollout_pass,
            "exact": self.exact,
            "ast_bits": self.ast_bits,
            "residual_bits": self.residual_bits,
            "response": json.loads(canonical_json(self.response)),
        }


@dataclass(frozen=True, slots=True)
class ProposalContext:
    """Complete proposer-visible context, intentionally public-only."""

    task: PublicTask
    parents: tuple[CandidateSummary, ...] = ()
    feedback: tuple[OracleFeedback, ...] = ()

    @property
    def content_hash(self) -> str:
        return sha256_text(canonical_json(self))


@dataclass(frozen=True, slots=True)
class ProposalBudget:
    max_candidates: int
    start_index: int
    proposer_seed: int


@dataclass(frozen=True, slots=True)
class SearchEvent:
    """Append-only event with deterministic payload and separate audit time."""

    sequence: int
    event_type: str
    logical_cost: int
    payload_json: str
    payload_hash: str
    audit_timestamp: str

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        event_type: str,
        logical_cost: int,
        payload: JsonObject,
        audit_timestamp: str,
    ) -> SearchEvent:
        payload_json = canonical_json(payload)
        return cls(
            sequence=sequence,
            event_type=event_type,
            logical_cost=logical_cost,
            payload_json=payload_json,
            payload_hash=sha256_text(payload_json),
            audit_timestamp=audit_timestamp,
        )

    @property
    def payload(self) -> JsonObject:
        return parse_json_object(self.payload_json)


BranchId = NewType("BranchId", str)


@dataclass(frozen=True, slots=True)
class BranchView:
    branch_id: BranchId
    task_id: str
    remaining_budget: int
    logical_cost: int


class Proposer(Protocol):
    proposer_id: str

    def propose(
        self, context: ProposalContext, budget: ProposalBudget
    ) -> Sequence[CandidatePayload]: ...


class Archive(Protocol):
    """Phase 0 interface only; archive policy begins in Phase 3."""

    def insert_if_elite(
        self, candidate: Candidate, result: OracleResult, event: SearchEvent
    ) -> bool: ...

    def candidates(self, task_id: str) -> Sequence[CandidateSummary]: ...


class Scheduler(Protocol):
    """Phase 0 interface only; concrete scheduling begins in Phase 3."""

    def select(self, branches: Sequence[BranchView]) -> BranchId: ...

    def observe(self, event: SearchEvent) -> None: ...
