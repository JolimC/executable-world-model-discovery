"""Safe immutable JSON artifact operations."""

from __future__ import annotations

import os
from pathlib import Path

from world_model_search.errors import PersistenceError
from world_model_search.serialization import canonical_json, sha256_text


def write_json_exclusive(path: Path, value: object) -> None:
    """Create a canonical JSON artifact without overwriting existing data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(value) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise PersistenceError(f"artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_text_exclusive(path: Path, data: str) -> None:
    """Create an immutable UTF-8 text artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise PersistenceError(f"artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_content_artifact(path: Path, canonical_content: str) -> str:
    """Create an immutable content artifact, accepting an identical prior write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_text(canonical_content)
    if path.exists():
        existing = path.read_text(encoding="utf-8").rstrip("\n")
        if existing != canonical_content:
            raise PersistenceError(f"content collision at artifact: {path}")
        return digest
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return write_content_artifact(path, canonical_content)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(canonical_content + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def read_text_artifact(path: Path) -> str:
    if not path.is_file():
        raise PersistenceError(f"artifact does not exist: {path}")
    return path.read_text(encoding="utf-8").rstrip("\n")
