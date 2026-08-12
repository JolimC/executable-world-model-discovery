"""Single Phase 5 prompt and fully bound request identity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from world_model_search.domain.types import ProposalRole, PublicTask
from world_model_search.dsl.primitives import PrimitiveRegistry
from world_model_search.memory.retrieval import RetrievalRecord
from world_model_search.model.schema import BATCH_SCHEMA_NAME, BATCH_SCHEMA_VERSION
from world_model_search.phase5_versions import (
    PHASE5_PRIMITIVE_LANGUAGE_VERSION,
    PHASE5_PROMPT_VERSION,
    PHASE5_REQUEST_IDENTITY_VERSION,
)
from world_model_search.serialization import JsonObject, canonical_json, sha256_json

PHASE5_PROMPT_TEMPLATE = "phase5-public-task-with-explicit-memory-v1"


@dataclass(frozen=True, slots=True)
class Phase5RequestBindings:
    transfer_split_hash: str
    memory_database_export_hash: str
    memory_snapshot_hash: str
    retrieval_record_hash: str
    primitive_registry_hash: str
    prompt_schema_hash: str
    budget_policy_hash: str
    code_config_hash: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if len(value) != 64 or set(value) - set("0123456789abcdef"):
                raise ValueError(f"Phase 5 request binding {name} must be a SHA-256")

    def to_value(self) -> JsonObject:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Phase5ModelRequest:
    backend_id: str
    provider_id: str
    resolved_model: str
    endpoint: str
    service_tier: str
    reasoning_effort: str
    max_output_tokens: int
    requested_batch_size: int
    role: ProposalRole
    rendered_input: str
    structured_schema: JsonObject
    bindings: Phase5RequestBindings
    sample_identity: JsonObject | None = None

    @property
    def prompt_template(self) -> str:
        return PHASE5_PROMPT_TEMPLATE

    @property
    def prompt_version(self) -> str:
        return PHASE5_PROMPT_VERSION

    @property
    def structured_schema_name(self) -> str:
        return BATCH_SCHEMA_NAME

    @property
    def structured_schema_version(self) -> int:
        return BATCH_SCHEMA_VERSION

    @property
    def settings(self) -> JsonObject:
        value: JsonObject = {
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "truncation": "disabled",
        }
        if self.sample_identity is not None:
            value["sample_identity"] = self.sample_identity
        return value

    def identity_value(self) -> JsonObject:
        return {
            "request_identity_version": PHASE5_REQUEST_IDENTITY_VERSION,
            "backend_id": self.backend_id,
            "provider_id": self.provider_id,
            "resolved_model": self.resolved_model,
            "endpoint": self.endpoint,
            "service_tier": self.service_tier,
            "settings": self.settings,
            "prompt": {
                "template": PHASE5_PROMPT_TEMPLATE,
                "version": PHASE5_PROMPT_VERSION,
                "rendered_input_utf8": self.rendered_input,
            },
            "structured_schema": self.structured_schema,
            "requested_batch_size": self.requested_batch_size,
            "role": self.role.value,
            "bindings": self.bindings.to_value(),
        }

    @property
    def request_hash(self) -> str:
        return sha256_json(self.identity_value())

    @property
    def conservative_input_token_bound(self) -> int:
        return len(canonical_json(self.identity_value()).encode("utf-8"))


def render_phase5_prompt(
    *,
    task: PublicTask,
    role: ProposalRole,
    requested_batch_size: int,
    retrieval: RetrievalRecord,
    primitives: PrimitiveRegistry,
) -> str:
    if requested_batch_size < 1:
        raise ValueError("Phase 5 requested batch size must be positive")
    memory_value: object = json.loads(retrieval.rendered_memory)
    task_value: object = json.loads(canonical_json(task))
    if not isinstance(memory_value, dict) or not isinstance(task_value, dict):
        raise TypeError("Phase 5 prompt values must be objects")
    payload: JsonObject = cast(
        JsonObject,
        {
            "prompt_contract": {
                "template_version": PHASE5_PROMPT_VERSION,
                "stateless": True,
                "complete_candidate_documents_only": True,
                "response_order_is_evaluation_order": True,
                "role": role.value,
                "requested_batch_size": requested_batch_size,
            },
            "public_task": task_value,
            "dsl_contract": {
                "base_dsl_version": "binary-ca-radius1-dsl-v1",
                "learned_primitive_language_version": PHASE5_PRIMITIVE_LANGUAGE_VERSION,
                "primitive_calls_expand_before_evaluation": True,
                "arbitrary_code_forbidden": True,
            },
            "memory_block": memory_value,
            "enabled_primitive_registry": primitives.safe_value(),
            "instruction": (
                "Return exactly the requested strict JSON candidate batch. Use only the typed "
                "base DSL and enabled zero-arity PrimitiveCall symbols. Include no prose, source "
                "code, hidden-data guesses, or unknown fields."
            ),
        },
    )
    forbidden = {
        "generator_family",
        "family_id",
        "reference_ast",
        "reference_rule",
        "semantic_hash",
        "oracle_handle",
        "artifact_path",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return {str(key) for key in value} | set().union(
                *(keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    if keys(payload) & forbidden:
        raise ValueError("Phase 5 proposer prompt contains evaluator-only metadata")
    return canonical_json(payload)


def assert_matched_prompt_isolation(memory_off: str, memory_on: str) -> None:
    """Fail unless the two prompt payloads differ only at the declared mechanisms."""

    try:
        off: object = json.loads(memory_off)
        on: object = json.loads(memory_on)
    except json.JSONDecodeError as exc:
        raise ValueError("Phase 5 prompt is not JSON") from exc
    if not isinstance(off, dict) or not isinstance(on, dict):
        raise ValueError("Phase 5 prompt root is not an object")
    mechanisms = {"memory_block", "enabled_primitive_registry"}
    if set(off) != set(on) or any(off[key] != on[key] for key in off.keys() - mechanisms):
        raise ValueError("matched-isolation gate: prompts differ outside Phase 5 mechanisms")
