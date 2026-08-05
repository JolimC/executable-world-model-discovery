"""Resolved run-manifest construction."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from world_model_search import __version__
from world_model_search.config import AppConfig
from world_model_search.domain.types import Task
from world_model_search.serialization import JsonObject, derive_seed, sha256_bytes, to_json_value

MANIFEST_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _git_output(repository_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return b"unavailable"


def _git_state(repository_root: Path) -> JsonObject:
    commit = _git_output(repository_root, "rev-parse", "HEAD").decode().strip()
    status = _git_output(repository_root, "status", "--porcelain=v1")
    patch = _git_output(repository_root, "diff", "--binary", "HEAD")
    untracked_names = _git_output(
        repository_root, "ls-files", "--others", "--exclude-standard", "-z"
    ).split(b"\0")
    untracked = bytearray()
    for raw_name in sorted(name for name in untracked_names if name):
        relative_name = raw_name.decode("utf-8", errors="surrogateescape")
        file_path = repository_root / relative_name
        if file_path.is_file():
            untracked.extend(raw_name)
            untracked.extend(b"\0")
            untracked.extend(hashlib.sha256(file_path.read_bytes()).digest())
    dirty_material = patch + b"\0status\0" + status + b"\0untracked\0" + bytes(untracked)
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "dirty_worktree_patch_hash": sha256_bytes(dirty_material),
    }


def _lock_state(repository_root: Path) -> JsonObject:
    lock_path = repository_root / "uv.lock"
    if not lock_path.is_file():
        return {
            "manager": "uv",
            "path": "uv.lock",
            "sha256": "missing",
            "format_version": "missing",
            "revision": "missing",
            "requires_python": "missing",
        }
    content = lock_path.read_bytes()
    metadata = tomllib.loads(content.decode("utf-8"))
    return {
        "manager": "uv",
        "path": "uv.lock",
        "sha256": sha256_bytes(content),
        "format_version": metadata.get("version", "unknown"),
        "revision": metadata.get("revision", "unknown"),
        "requires_python": metadata.get("requires-python", "unknown"),
    }


def build_manifest(
    *,
    repository_root: Path,
    run_id: str,
    config: AppConfig,
    config_source: str,
    task: Task,
) -> JsonObject:
    """Build audit metadata; not all manifest fields are deterministic."""

    raw = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "package_version": __version__,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_source": config_source,
        "resolved_configuration": config.to_mapping(),
        "configuration_hash": config.content_hash,
        "git": _git_state(repository_root),
        "runtime": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "dependency_lock": _lock_state(repository_root),
            "operating_system": platform.platform(),
            "hardware": {
                "machine": platform.machine(),
                "processor": platform.processor() or "unknown",
                "logical_cpu_count": os.cpu_count(),
            },
        },
        "seeds": {
            "master_seed": config.run.seed,
            "task_seed": task.seed,
            "proposer_seed": derive_seed(config.run.seed, "proposer"),
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "internal_family_id": task.internal_family_id,
                "public_world_spec": task.public_world_spec,
                "split": task.split,
                "public_artifact_hash": task.public_artifact_hash,
                "hidden_artifact_id": task.hidden_artifact_id,
                "generator_version": task.generator_version,
            }
        ],
        "versions": {
            "oracle": config.oracle.oracle_id,
            "dsl": "phase0-opaque-stub-v1",
            "coding_scheme": "not-implemented-phase0",
            "prompt": "not-applicable-mock",
            "scheduler": "not-implemented-phase0",
        },
        "proposer": {
            "identifier": config.proposer.proposer_id,
            "model": "mock-v1",
            "decoding_settings": {},
        },
        "budget": {
            "oracle_calls": config.run.max_steps,
            "language_model_tokens": 0,
            "cpu_seconds": None,
            "elapsed_seconds": None,
        },
        "parent_run_id": None,
        "deterministic_hash_profile": "canonical-json-v1",
    }
    value = to_json_value(raw)
    if not isinstance(value, dict):  # pragma: no cover - mapping invariant
        raise AssertionError("manifest did not serialize as an object")
    return value
