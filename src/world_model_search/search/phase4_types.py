"""Phase 4 conditions, joint budgets, candidate identity, and request lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from world_model_search.domain.types import Candidate
from world_model_search.dsl.ast import BitExpr
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.json_schema import ast_to_value
from world_model_search.phase4_versions import (
    PHASE4_BUDGET_VERSION,
    PHASE4_CANDIDATE_IDENTITY_VERSION,
)
from world_model_search.serialization import JsonObject, sha256_json


class Phase4Condition(StrEnum):
    DIRECT = "direct-llm-v1"
    INCUMBENT = "single-incumbent-v1"
    DIVERSE = "uniform-diverse-archive-v1"


class RequestState(StrEnum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    RESPONDED = "responded"
    SCHEMA_FAILURE = "schema-failure"
    FAILED = "failed"
    USAGE_UNCERTAIN = "usage-uncertain"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class Phase4BudgetState:
    model_request_cap: int
    input_token_cap: int
    output_token_cap: int
    total_token_cap: int
    proposal_item_cap: int
    oracle_call_cap: int
    child_nano_usd_cap: int
    logical_model_calls: int = 0
    model_request_attempts: int = 0
    physical_provider_calls: int = 0
    exact_cache_hits: int = 0
    retries: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    proposal_items: int = 0
    valid_items: int = 0
    invalid_items: int = 0
    canonical_duplicates: int = 0
    semantic_duplicates: int = 0
    oracle_invocations: int = 0
    evaluated_candidates: int = 0
    scheduler_selections: int = 0
    actual_nano_usd: int = 0
    uncertain_nano_usd: int = 0
    released_nano_usd: int = 0

    def updated(self, **increments: int) -> Phase4BudgetState:
        values: dict[str, int] = {}
        for name, increment in increments.items():
            current = getattr(self, name)
            if not isinstance(current, int):
                raise TypeError(f"Phase 4 budget field is not numeric: {name}")
            values[name] = current + increment
        value = replace(self, **values)
        value.validate()
        return value

    @property
    def remaining_model_requests(self) -> int:
        return self.model_request_cap - self.model_request_attempts

    @property
    def remaining_proposal_items(self) -> int:
        return self.proposal_item_cap - self.proposal_items

    @property
    def remaining_oracle_calls(self) -> int:
        return self.oracle_call_cap - self.oracle_invocations

    @property
    def exhausted(self) -> bool:
        return (
            self.remaining_model_requests <= 0
            or self.remaining_proposal_items <= 0
            or self.remaining_oracle_calls <= 0
            or self.input_tokens >= self.input_token_cap
            or self.output_tokens >= self.output_token_cap
            or self.total_tokens >= self.total_token_cap
            or self.actual_nano_usd + self.uncertain_nano_usd >= self.child_nano_usd_cap > 0
        )

    def preflight(self, *, input_token_bound: int, max_output_tokens: int) -> None:
        self.validate()
        if self.remaining_model_requests < 1:
            raise ValueError("model request cap exhausted")
        if self.input_tokens + input_token_bound > self.input_token_cap:
            raise ValueError("input token cap cannot reserve this request")
        if self.output_tokens + max_output_tokens > self.output_token_cap:
            raise ValueError("output token cap cannot reserve this request")
        if self.total_tokens + input_token_bound + max_output_tokens > self.total_token_cap:
            raise ValueError("total token cap cannot reserve this request")

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Phase 4 budget {name} must be a nonnegative integer")
        if self.model_request_attempts > self.model_request_cap:
            raise ValueError("model request cap exceeded")
        if self.input_tokens > self.input_token_cap or self.output_tokens > self.output_token_cap:
            raise ValueError("model token component cap exceeded")
        if self.total_tokens > self.total_token_cap:
            raise ValueError("model total token cap exceeded")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("Phase 4 total token accounting does not reconcile")
        if (
            self.cached_input_tokens > self.input_tokens
            or self.reasoning_tokens > self.output_tokens
        ):
            raise ValueError("Phase 4 token breakdown is inconsistent")
        if self.proposal_items > self.proposal_item_cap:
            raise ValueError("proposal item cap exceeded")
        if self.valid_items + self.invalid_items != self.proposal_items:
            raise ValueError("proposal item validity accounting does not reconcile")
        if self.oracle_invocations > self.oracle_call_cap:
            raise ValueError("oracle cap exceeded")
        if self.evaluated_candidates != self.oracle_invocations:
            raise ValueError("every Phase 4 oracle call evaluates exactly one candidate")
        if self.physical_provider_calls + self.exact_cache_hits != self.model_request_attempts:
            raise ValueError("request attempt/cache/provider accounting does not reconcile")
        if self.retries > self.model_request_attempts:
            raise ValueError("retry count exceeds request attempts")
        if self.actual_nano_usd + self.uncertain_nano_usd > self.child_nano_usd_cap:
            raise ValueError("child model-dollar cap exceeded")

    def to_value(self) -> JsonObject:
        counters = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name
            not in {
                "model_request_cap",
                "input_token_cap",
                "output_token_cap",
                "total_token_cap",
                "proposal_item_cap",
                "oracle_call_cap",
                "child_nano_usd_cap",
            }
        }
        return {
            "budget_version": PHASE4_BUDGET_VERSION,
            "caps": {
                "model_requests": self.model_request_cap,
                "input_tokens": self.input_token_cap,
                "output_tokens": self.output_token_cap,
                "total_tokens": self.total_token_cap,
                "proposal_items": self.proposal_item_cap,
                "oracle_calls": self.oracle_call_cap,
                "child_nano_usd": self.child_nano_usd_cap,
            },
            "counters": counters,
            "remaining": {
                "model_requests": self.remaining_model_requests,
                "proposal_items": self.remaining_proposal_items,
                "oracle_calls": self.remaining_oracle_calls,
            },
        }

    @classmethod
    def from_value(cls, value: object) -> Phase4BudgetState:
        if not isinstance(value, dict) or value.get("budget_version") != PHASE4_BUDGET_VERSION:
            raise ValueError("unsupported Phase 4 budget record")
        caps = value.get("caps")
        counters = value.get("counters")
        if not isinstance(caps, dict) or not isinstance(counters, dict):
            raise ValueError("Phase 4 budget caps/counters are malformed")
        expected_caps = {
            "model_requests",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "proposal_items",
            "oracle_calls",
            "child_nano_usd",
        }
        expected_counters = {
            name
            for name in cls.__dataclass_fields__
            if name
            not in {
                "model_request_cap",
                "input_token_cap",
                "output_token_cap",
                "total_token_cap",
                "proposal_item_cap",
                "oracle_call_cap",
                "child_nano_usd_cap",
            }
        }
        if set(caps) != expected_caps or set(counters) != expected_counters:
            raise ValueError("Phase 4 budget field set is malformed")

        def integer(mapping: dict[object, object], name: str) -> int:
            item = mapping[name]
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"Phase 4 budget {name} must be an integer")
            return item

        result = cls(
            model_request_cap=integer(caps, "model_requests"),
            input_token_cap=integer(caps, "input_tokens"),
            output_token_cap=integer(caps, "output_tokens"),
            total_token_cap=integer(caps, "total_tokens"),
            proposal_item_cap=integer(caps, "proposal_items"),
            oracle_call_cap=integer(caps, "oracle_calls"),
            child_nano_usd_cap=integer(caps, "child_nano_usd"),
            **{name: integer(counters, name) for name in expected_counters},
        )
        result.validate()
        return result


def phase4_candidate(
    *,
    task_id: str,
    ast: BitExpr,
    parent_ids: tuple[str, ...],
    operator_id: str,
    context_hash: str,
    payload_hash: str,
    semantic_hash: str,
) -> Candidate:
    canonical = canonicalize(ast)
    identity: JsonObject = {
        "candidate_identity_schema": PHASE4_CANDIDATE_IDENTITY_VERSION,
        "task_id": task_id,
        "canonical_ast": ast_to_value(canonical),
        "ordered_parent_ids": list(parent_ids),
        "proposer_id": "llm" if operator_id.startswith("llm-") else "initialization",
        "operator_id": operator_id,
        "public_context_hash": context_hash,
        "payload_hash": payload_hash,
        "coding_version": "binary-ca-prefix-v1",
    }
    return Candidate(
        candidate_id=sha256_json(identity),
        task_id=task_id,
        ast=canonical,
        parent_ids=parent_ids,
        proposer_id="llm" if operator_id.startswith("llm-") else "initialization",
        operator_id=operator_id,
        context_hash=context_hash,
        payload_hash=payload_hash,
        semantic_hash=semantic_hash,
    )
