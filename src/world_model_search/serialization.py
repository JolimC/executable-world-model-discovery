"""Canonical serialization and content hashing.

Only explicitly supplied values enter a hash. Callers are responsible for keeping
diagnostic timestamps, paths, timings, and platform details outside deterministic payloads.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


def to_json_value(value: object) -> JsonValue:
    """Convert supported typed values to deterministic JSON-compatible values."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats cannot be serialized deterministically")
        return value
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        result: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            result[key] = to_json_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [to_json_value(item) for item in value]
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialize with the project's version-1 deterministic JSON profile."""

    return json.dumps(
        to_json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(data).hexdigest()


def sha256_text(data: str) -> str:
    """Hash UTF-8 text."""

    return sha256_bytes(data.encode("utf-8"))


def sha256_json(value: object) -> str:
    """Hash a canonical JSON encoding."""

    return sha256_text(canonical_json(value))


def derive_seed(master_seed: int, namespace: str) -> int:
    """Derive a distinct signed-64-bit deterministic seed for one consumer."""

    material = f"phase0-seed-v1\0{namespace}\0{master_seed}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def parse_json_object(data: str) -> JsonObject:
    """Parse JSON while requiring an object at the root."""

    value: object = json.loads(data)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object")
    return value
