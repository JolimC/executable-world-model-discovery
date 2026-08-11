"""Exact request-response cache with namespace and corruption checks."""

from __future__ import annotations

import json
from pathlib import Path

from world_model_search.errors import PersistenceError
from world_model_search.model.types import ModelRequest, ModelResponse
from world_model_search.persistence.artifacts import write_content_artifact
from world_model_search.serialization import JsonObject, canonical_json, sha256_text

CACHE_VERSION = "phase4-exact-response-cache-v1"


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


class ExactResponseCache:
    def __init__(self, root: Path, namespace: str) -> None:
        if not namespace or "/" in namespace or "\\" in namespace or ".." in namespace:
            raise ValueError("cache namespace must be one safe path component")
        self.root = root
        self.namespace = namespace

    def _path(self, request: ModelRequest) -> Path:
        return self.root / self.namespace / f"{request.request_hash}.json"

    def get(self, request: ModelRequest) -> ModelResponse | None:
        path = self._path(request)
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8").rstrip("\n")
            raw: object = json.loads(text, parse_constant=_reject_constant)
        except (OSError, json.JSONDecodeError) as exc:
            raise PersistenceError("model response cache entry is corrupt") from exc
        if not isinstance(raw, dict):
            raise PersistenceError("model response cache entry is not an object")
        content = raw.get("content")
        if (
            raw.get("cache_version") != CACHE_VERSION
            or raw.get("cache_key") != request.request_hash
            or not isinstance(content, dict)
            or raw.get("content_hash") != sha256_text(canonical_json(content))
        ):
            raise PersistenceError("model response cache hash or identity mismatch")
        try:
            response = ModelResponse.from_deterministic_value(content)
            if response.request_hash != request.request_hash:
                raise ValueError("cached response belongs to another exact request")
            return response
        except (KeyError, TypeError, ValueError) as exc:
            raise PersistenceError("model response cache fields are corrupt") from exc

    def put(self, request: ModelRequest, response: ModelResponse) -> str:
        if response.request_hash != request.request_hash:
            raise ValueError("cannot cache a response for another request")
        content = response.deterministic_value()
        artifact: JsonObject = {
            "cache_version": CACHE_VERSION,
            "cache_key": request.request_hash,
            "content": content,
            "content_hash": sha256_text(canonical_json(content)),
        }
        return write_content_artifact(self._path(request), canonical_json(artifact))
