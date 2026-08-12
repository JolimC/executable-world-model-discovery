"""Immutable provider-neutral model request, response, usage, and error records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from world_model_search.domain.types import ProposalRole
from world_model_search.serialization import JsonObject, canonical_json, sha256_json


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


class ModelErrorCategory(StrEnum):
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    INVALID_REQUEST = "invalid-request"
    NOT_FOUND = "not-found"
    RATE_LIMIT = "rate-limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER = "server"
    MALFORMED_RESPONSE = "malformed-response"
    USAGE_MISSING = "usage-missing"
    CACHE_CORRUPTION = "cache-corruption"
    UNKNOWN = "unknown"


@runtime_checkable
class ModelDispatchRequest(Protocol):
    """Structural request boundary shared by frozen Phase 4 and Phase 5 identities."""

    @property
    def backend_id(self) -> str: ...

    @property
    def provider_id(self) -> str: ...

    @property
    def resolved_model(self) -> str: ...

    @property
    def endpoint(self) -> str: ...

    @property
    def service_tier(self) -> str: ...

    @property
    def rendered_input(self) -> str: ...

    @property
    def structured_schema_name(self) -> str: ...

    @property
    def structured_schema(self) -> JsonObject: ...

    @property
    def requested_batch_size(self) -> int: ...

    @property
    def settings(self) -> JsonObject: ...

    @property
    def role(self) -> ProposalRole: ...

    @property
    def request_hash(self) -> str: ...

    @property
    def conservative_input_token_bound(self) -> int: ...

    def identity_value(self) -> JsonObject: ...


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Provider-normalized token usage; reasoning is a subset of output."""

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.cached_input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.total_tokens,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("model usage values must be nonnegative integers")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input tokens cannot exceed input tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning tokens are included in and cannot exceed output tokens")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total tokens must equal input plus output tokens")

    def to_value(self) -> JsonObject:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "output_token_details": {"reasoning_tokens": self.reasoning_tokens},
            "total_tokens": self.total_tokens,
            "billing_rule": "reasoning-tokens-already-in-output-v1",
        }

    @classmethod
    def from_value(cls, value: object) -> ModelUsage:
        if not isinstance(value, dict) or set(value) != {
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "output_token_details",
            "total_tokens",
            "billing_rule",
        }:
            raise ValueError("model usage has missing or unknown fields")
        details = value.get("output_token_details")
        if not isinstance(details, dict) or set(details) != {"reasoning_tokens"}:
            raise ValueError("model output-token details are malformed")
        if value.get("billing_rule") != "reasoning-tokens-already-in-output-v1":
            raise ValueError("model usage billing rule is unsupported")
        return cls(
            _integer(value["input_tokens"], "input_tokens"),
            _integer(value["cached_input_tokens"], "cached_input_tokens"),
            _integer(value["output_tokens"], "output_tokens"),
            _integer(details["reasoning_tokens"], "reasoning_tokens"),
            _integer(value["total_tokens"], "total_tokens"),
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Complete logical request identity with no credential, path, time, or provider request ID."""

    backend_id: str
    provider_id: str
    resolved_model: str
    endpoint: str
    service_tier: str
    prompt_template: str
    prompt_version: str
    rendered_input: str
    structured_schema_name: str
    structured_schema_version: int
    structured_schema: JsonObject
    role: ProposalRole
    requested_batch_size: int
    settings: JsonObject

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.backend_id,
                self.provider_id,
                self.resolved_model,
                self.endpoint,
                self.service_tier,
                self.prompt_template,
                self.prompt_version,
                self.rendered_input,
                self.structured_schema_name,
            )
        ):
            raise ValueError("model request string fields must be nonempty")
        if self.structured_schema_version < 1 or self.requested_batch_size < 1:
            raise ValueError("model request schema version and batch size must be positive")

    def identity_value(self) -> JsonObject:
        return {
            "request_identity_version": "phase4-exact-model-request-v1",
            "backend_id": self.backend_id,
            "provider_id": self.provider_id,
            "resolved_model": self.resolved_model,
            "endpoint": self.endpoint,
            "service_tier": self.service_tier,
            "prompt": {
                "template": self.prompt_template,
                "version": self.prompt_version,
                "rendered_input_utf8": self.rendered_input,
            },
            "structured_output": {
                "name": self.structured_schema_name,
                "version": self.structured_schema_version,
                "schema": self.structured_schema,
            },
            "role": self.role.value,
            "requested_batch_size": self.requested_batch_size,
            "settings": self.settings,
        }

    @property
    def request_hash(self) -> str:
        return sha256_json(self.identity_value())

    @property
    def conservative_input_token_bound(self) -> int:
        """UTF-8 byte count is a deliberately conservative tokenizer-independent bound."""

        return len(canonical_json(self.identity_value()).encode("utf-8"))

    @classmethod
    def from_identity_value(cls, value: object) -> ModelRequest:
        expected = {
            "request_identity_version",
            "backend_id",
            "provider_id",
            "resolved_model",
            "endpoint",
            "service_tier",
            "prompt",
            "structured_output",
            "role",
            "requested_batch_size",
            "settings",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("request_identity_version") != "phase4-exact-model-request-v1"
        ):
            raise ValueError("unsupported model request identity")
        prompt = value.get("prompt")
        structured = value.get("structured_output")
        settings = value.get("settings")
        if (
            not isinstance(prompt, dict)
            or set(prompt) != {"template", "version", "rendered_input_utf8"}
            or not isinstance(structured, dict)
            or set(structured) != {"name", "version", "schema"}
            or not isinstance(settings, dict)
            or not all(isinstance(key, str) for key in settings)
        ):
            raise ValueError("model request identity sections are malformed")
        schema = structured.get("schema")
        string_fields = (
            "backend_id",
            "provider_id",
            "resolved_model",
            "endpoint",
            "service_tier",
        )
        if (
            not isinstance(schema, dict)
            or not all(isinstance(key, str) for key in schema)
            or any(not isinstance(value.get(name), str) for name in string_fields)
            or any(
                not isinstance(prompt.get(name), str)
                for name in ("template", "version", "rendered_input_utf8")
            )
            or not isinstance(structured.get("name"), str)
            or not isinstance(value.get("role"), str)
        ):
            raise ValueError("model request structured schema is malformed")
        return cls(
            backend_id=value["backend_id"],
            provider_id=value["provider_id"],
            resolved_model=value["resolved_model"],
            endpoint=value["endpoint"],
            service_tier=value["service_tier"],
            prompt_template=prompt["template"],
            prompt_version=prompt["version"],
            rendered_input=prompt["rendered_input_utf8"],
            structured_schema_name=structured["name"],
            structured_schema_version=_integer(structured["version"], "structured version"),
            structured_schema=schema,
            role=ProposalRole(value["role"]),
            requested_batch_size=_integer(value["requested_batch_size"], "batch size"),
            settings=settings,
        )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized provider response. Raw submitted text remains immutable evidence."""

    request_hash: str
    raw_text: str
    usage: ModelUsage
    provider_request_id: str | None = None
    resolved_model: str | None = None
    service_tier: str | None = None
    system_fingerprint: str | None = None
    provider_latency_ns: int | None = None

    def __post_init__(self) -> None:
        if len(self.request_hash) != 64 or not self.raw_text:
            raise ValueError("model response requires a request hash and nonempty raw text")
        if self.provider_latency_ns is not None and self.provider_latency_ns < 0:
            raise ValueError("provider latency cannot be negative")

    def deterministic_value(self) -> JsonObject:
        return {
            "response_schema_version": 1,
            "request_hash": self.request_hash,
            "raw_text": self.raw_text,
            "usage": self.usage.to_value(),
            "provider_request_id": self.provider_request_id,
            "resolved_model": self.resolved_model,
            "service_tier": self.service_tier,
            "system_fingerprint": self.system_fingerprint,
        }

    @classmethod
    def from_deterministic_value(cls, value: object) -> ModelResponse:
        if not isinstance(value, dict) or set(value) != {
            "response_schema_version",
            "request_hash",
            "raw_text",
            "usage",
            "provider_request_id",
            "resolved_model",
            "service_tier",
            "system_fingerprint",
        }:
            raise ValueError("model response has missing or unknown fields")
        if value.get("response_schema_version") != 1:
            raise ValueError("unsupported model response schema")
        request_hash = value.get("request_hash")
        raw_text = value.get("raw_text")
        if not isinstance(request_hash, str) or not isinstance(raw_text, str):
            raise ValueError("model response identity or text is malformed")
        return cls(
            request_hash=request_hash,
            raw_text=raw_text,
            usage=ModelUsage.from_value(value.get("usage")),
            provider_request_id=_optional_string(
                value.get("provider_request_id"), "provider_request_id"
            ),
            resolved_model=_optional_string(value.get("resolved_model"), "resolved_model"),
            service_tier=_optional_string(value.get("service_tier"), "service_tier"),
            system_fingerprint=_optional_string(
                value.get("system_fingerprint"), "system_fingerprint"
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelError:
    """Sanitized failure data; provider exception text and credentials never cross this boundary."""

    category: ModelErrorCategory
    retryable: bool
    usage_uncertain: bool
    http_status: int | None = None
    provider_request_id: str | None = None
    usage: ModelUsage | None = None

    def to_value(self) -> JsonObject:
        return {
            "error_schema_version": 1,
            "category": self.category.value,
            "retryable": self.retryable,
            "usage_uncertain": self.usage_uncertain,
            "http_status": self.http_status,
            "provider_request_id": self.provider_request_id,
            "usage": self.usage.to_value() if self.usage is not None else None,
        }

    @classmethod
    def from_value(cls, value: object) -> ModelError:
        if not isinstance(value, dict) or set(value) != {
            "error_schema_version",
            "category",
            "retryable",
            "usage_uncertain",
            "http_status",
            "provider_request_id",
            "usage",
        }:
            raise ValueError("model error has missing or unknown fields")
        if value.get("error_schema_version") != 1:
            raise ValueError("unsupported model error schema")
        retryable = value.get("retryable")
        usage_uncertain = value.get("usage_uncertain")
        category = value.get("category")
        status = value.get("http_status")
        if (
            not isinstance(category, str)
            or not isinstance(retryable, bool)
            or not isinstance(usage_uncertain, bool)
            or (status is not None and (isinstance(status, bool) or not isinstance(status, int)))
        ):
            raise ValueError("model error fields are malformed")
        usage_value = value.get("usage")
        return cls(
            category=ModelErrorCategory(category),
            retryable=retryable,
            usage_uncertain=usage_uncertain,
            http_status=status,
            provider_request_id=_optional_string(
                value.get("provider_request_id"), "provider_request_id"
            ),
            usage=ModelUsage.from_value(usage_value) if usage_value is not None else None,
        )


class ModelDispatchError(RuntimeError):
    """Exception wrapper carrying only a sanitized provider-neutral error."""

    def __init__(self, error: ModelError) -> None:
        super().__init__(f"model dispatch failed: {error.category.value}")
        self.error = error


@runtime_checkable
class ModelBackend(Protocol):
    backend_id: str
    provider_id: str

    def dispatch(self, request: ModelDispatchRequest) -> ModelResponse: ...
