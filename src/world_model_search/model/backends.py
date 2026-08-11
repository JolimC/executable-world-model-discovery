"""Scripted, recorded, and OpenAI Responses implementations of the model boundary."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

from world_model_search.model.types import (
    ModelBackend,
    ModelDispatchError,
    ModelError,
    ModelErrorCategory,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from world_model_search.serialization import canonical_json


def _default_scripted_text(request: ModelRequest) -> str:
    asts = (
        {"op": "Xor", "left": {"op": "At", "offset": -1}, "right": {"op": "At", "offset": 1}},
        {"op": "Parity", "mask": [-1, 0, 1]},
        {"op": "Majority", "mask": [-1, 0, 1]},
        {"op": "Not", "expr": {"op": "At", "offset": 0}},
    )
    candidates = [
        {
            "candidate_schema_version": 1,
            "dsl_version": "binary-ca-radius1-dsl-v1",
            "ast": asts[index % len(asts)],
        }
        for index in range(request.requested_batch_size)
    ]
    return canonical_json(
        {"batch_schema_version": 1, "role": request.role.value, "candidates": candidates}
    )


class ScriptedBackend:
    """Deterministic in-process backend used by all ordinary tests and fake profiles."""

    backend_id = "scripted-deterministic-v1"
    provider_id = "scripted"

    def __init__(self, script: Sequence[ModelResponse | ModelError | str] = ()) -> None:
        self._script = tuple(script)
        self.dispatch_count = 0

    def dispatch(self, request: ModelRequest) -> ModelResponse:
        index = self.dispatch_count
        self.dispatch_count += 1
        scripted: ModelResponse | ModelError | str = (
            self._script[index] if index < len(self._script) else _default_scripted_text(request)
        )
        if isinstance(scripted, ModelError):
            raise ModelDispatchError(scripted)
        if isinstance(scripted, ModelResponse):
            if scripted.request_hash != request.request_hash:
                raise ValueError("scripted response request hash mismatch")
            return scripted
        input_tokens = min(request.conservative_input_token_bound, 10_000)
        output_tokens = max(1, len(scripted.encode("utf-8")) // 4)
        return ModelResponse(
            request_hash=request.request_hash,
            raw_text=scripted,
            usage=ModelUsage(input_tokens, 0, output_tokens, 0, input_tokens + output_tokens),
            provider_request_id=f"scripted-{index}",
            resolved_model=request.resolved_model,
            service_tier=request.service_tier,
            system_fingerprint="scripted-stable-v1",
            provider_latency_ns=0,
        )


class RecordedResponseBackend:
    """Offline exact-request replay backend; it has no provider transport."""

    backend_id = "recorded-response-v1"
    provider_id = "recorded"

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = {response.request_hash: response for response in responses}
        if len(self._responses) != len(responses):
            raise ValueError("recorded responses require unique request hashes")

    def dispatch(self, request: ModelRequest) -> ModelResponse:
        try:
            return self._responses[request.request_hash]
        except KeyError as exc:
            raise ModelDispatchError(
                ModelError(ModelErrorCategory.NOT_FOUND, retryable=False, usage_uncertain=False)
            ) from exc


class OfflineResumeBackend:
    """Identity-preserving sentinel used only to finish a durable response offline."""

    def __init__(self, *, backend_id: str, provider_id: str) -> None:
        self.backend_id = backend_id
        self.provider_id = provider_id

    def dispatch(self, request: ModelRequest) -> ModelResponse:
        del request
        raise ModelDispatchError(
            ModelError(ModelErrorCategory.PERMISSION, retryable=False, usage_uncertain=False)
        )


@dataclass(frozen=True, slots=True)
class LiveOptIn:
    cli_allowed: bool
    environment_allowed: bool

    @classmethod
    def resolve(cls, cli_allowed: bool) -> LiveOptIn:
        return cls(cli_allowed, os.environ.get("WMS_ALLOW_LIVE_MODEL") == "1")

    def require(self) -> None:
        if not self.cli_allowed or not self.environment_allowed:
            raise ModelDispatchError(
                ModelError(ModelErrorCategory.PERMISSION, retryable=False, usage_uncertain=False)
            )


class OpenAIResponsesBackend:
    """Thin official-SDK transport adapter; domain types never depend on SDK classes."""

    backend_id = "openai-responses-sdk-v1"
    provider_id = "openai"

    def __init__(self, *, opt_in: LiveOptIn) -> None:
        opt_in.require()
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ModelDispatchError(ModelError(ModelErrorCategory.AUTHENTICATION, False, False))
        self._api_key = key

    def dispatch(self, request: ModelRequest) -> ModelResponse:
        if request.endpoint != "v1/responses":
            raise ModelDispatchError(ModelError(ModelErrorCategory.INVALID_REQUEST, False, False))
        started = perf_counter_ns()
        try:
            # The official dependency is intentionally imported only inside the live adapter.
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                NotFoundError,
                OpenAI,
                PermissionDeniedError,
                RateLimitError,
            )

            client = OpenAI(api_key=self._api_key, max_retries=0)
            settings = request.settings
            max_output = settings.get("max_output_tokens")
            reasoning = settings.get("reasoning")
            if not isinstance(max_output, int) or not isinstance(reasoning, dict):
                raise ModelDispatchError(
                    ModelError(ModelErrorCategory.INVALID_REQUEST, False, False)
                )
            create_arguments: dict[str, Any] = {
                "model": request.resolved_model,
                "input": request.rendered_input,
                "max_output_tokens": max_output,
                "reasoning": reasoning,
                "service_tier": request.service_tier,
                "store": False,
                "truncation": "disabled",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": request.structured_schema_name,
                        "strict": True,
                        "schema": request.structured_schema,
                    }
                },
            }
            response = client.responses.create(**create_arguments)
        except ModelDispatchError:
            raise
        except (AuthenticationError, PermissionDeniedError, BadRequestError, NotFoundError) as exc:
            category = {
                AuthenticationError: ModelErrorCategory.AUTHENTICATION,
                PermissionDeniedError: ModelErrorCategory.PERMISSION,
                BadRequestError: ModelErrorCategory.INVALID_REQUEST,
                NotFoundError: ModelErrorCategory.NOT_FOUND,
            }.get(type(exc), ModelErrorCategory.INVALID_REQUEST)
            raise ModelDispatchError(
                ModelError(category, False, False, getattr(exc, "status_code", None))
            ) from None
        except RateLimitError as exc:
            raise ModelDispatchError(
                ModelError(ModelErrorCategory.RATE_LIMIT, True, False, exc.status_code)
            ) from None
        except (APITimeoutError, APIConnectionError) as exc:
            category = (
                ModelErrorCategory.TIMEOUT
                if isinstance(exc, APITimeoutError)
                else ModelErrorCategory.CONNECTION
            )
            raise ModelDispatchError(ModelError(category, True, True)) from None
        except APIStatusError as exc:
            raise ModelDispatchError(
                ModelError(ModelErrorCategory.SERVER, exc.status_code >= 500, True, exc.status_code)
            ) from None
        usage = getattr(response, "usage", None)
        if usage is None:
            raise ModelDispatchError(ModelError(ModelErrorCategory.USAGE_MISSING, False, True))
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        try:
            normalized_usage = ModelUsage(
                input_tokens=int(getattr(usage, "input_tokens", -1)),
                cached_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", -1)),
                reasoning_tokens=int(getattr(output_details, "reasoning_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", -1)),
            )
        except (TypeError, ValueError) as exc:
            raise ModelDispatchError(
                ModelError(ModelErrorCategory.USAGE_MISSING, False, True)
            ) from exc
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text:
            # Preserve a normalized dump only for provider diagnostics; it is never accepted as AST.
            dumped = response.model_dump(mode="json")
            output_text = canonical_json(json.loads(json.dumps(dumped)))
        return ModelResponse(
            request_hash=request.request_hash,
            raw_text=output_text,
            usage=normalized_usage,
            provider_request_id=getattr(response, "id", None),
            resolved_model=getattr(response, "model", None),
            service_tier=getattr(response, "service_tier", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
            provider_latency_ns=max(0, perf_counter_ns() - started),
        )


def assert_backend_contract(backend: ModelBackend) -> None:
    if not isinstance(backend, ModelBackend):
        raise TypeError("backend does not implement the provider-neutral model protocol")
