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
from world_model_search.dsl.versions import (
    ANALYSIS_ARTIFACT_VERSION,
    ANALYSIS_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    CANONICALIZER_VERSION,
    DSL_VERSION,
    ENUMERATOR_VERSION,
    INTERPRETER_VERSION,
    PHASE3_ANALYSIS_VERSION,
    PHASE3_ARCHIVE_VERSION,
    PHASE3_BUDGET_VERSION,
    PHASE3_CANDIDATE_IDENTITY_VERSION,
    PHASE3_DATABASE_SCHEMA_VERSION,
    PHASE3_DESCRIPTOR_VERSION,
    PHASE3_EVENT_SCHEMA_VERSION,
    PHASE3_INITIALIZATION_VERSION,
    PHASE3_MANIFEST_SCHEMA_VERSION,
    PHASE3_OPERATOR_VERSION,
    PHASE3_PROPOSAL_ARTIFACT_VERSION,
    PHASE3_RESULTS_SCHEMA_VERSION,
    PHASE3_RNG_VERSION,
    PHASE3_SCHEDULER_VERSION,
    PREFIX_CODE_VERSION,
    RANK_VERSION,
    RESIDUAL_CODE_VERSION,
    SEMANTIC_HASH_VERSION,
    TRUTH_TABLE_BASELINE_VERSION,
)
from world_model_search.oracle.elementary import ROLLOUT_VERSION, SIMULATOR_VERSION
from world_model_search.oracle.exact import EXACT_ORACLE_VERSION
from world_model_search.serialization import JsonObject, derive_seed, sha256_bytes, to_json_value

PHASE0_MANIFEST_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 3


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

    manifest_schema = {
        1: PHASE0_MANIFEST_SCHEMA_VERSION,
        2: MANIFEST_SCHEMA_VERSION,
        3: PHASE3_MANIFEST_SCHEMA_VERSION,
    }[config.schema_version]
    versions: JsonObject = {
        "oracle": config.oracle.oracle_id,
        "dsl": "phase0-opaque-stub-v1",
        "coding_scheme": "not-implemented-phase0",
        "prompt": "not-applicable-mock",
        "scheduler": "not-implemented-phase0",
    }
    if config.schema_version == 2:
        versions = {
            "configuration_schema": 2,
            "run_manifest_schema": MANIFEST_SCHEMA_VERSION,
            "database_schema": 2,
            "event_schema": 2,
            "results_schema": 2,
            "candidate_schema": CANDIDATE_SCHEMA_VERSION,
            "dsl": DSL_VERSION,
            "canonicalizer": CANONICALIZER_VERSION,
            "interpreter": INTERPRETER_VERSION,
            "semantic_hash": SEMANTIC_HASH_VERSION,
            "coding_scheme": PREFIX_CODE_VERSION,
            "residual_code": RESIDUAL_CODE_VERSION,
            "rank": RANK_VERSION,
            "enumerator": ENUMERATOR_VERSION,
            "truth_table_baseline": TRUTH_TABLE_BASELINE_VERSION,
            "oracle": EXACT_ORACLE_VERSION,
            "simulator": SIMULATOR_VERSION,
            "rollout": ROLLOUT_VERSION,
            "analysis": ANALYSIS_VERSION,
            "analysis_artifact": ANALYSIS_ARTIFACT_VERSION,
            "artifact": "immutable-canonical-json-v1",
            "prompt": "not-applicable-enumerative",
            "scheduler": "not-implemented-phase2",
        }
    if config.schema_version == 3:
        versions = {
            "configuration_schema": 3,
            "run_manifest_schema": PHASE3_MANIFEST_SCHEMA_VERSION,
            "database_schema": PHASE3_DATABASE_SCHEMA_VERSION,
            "event_schema": PHASE3_EVENT_SCHEMA_VERSION,
            "results_schema": PHASE3_RESULTS_SCHEMA_VERSION,
            "candidate_schema": CANDIDATE_SCHEMA_VERSION,
            "candidate_identity": PHASE3_CANDIDATE_IDENTITY_VERSION,
            "dsl": DSL_VERSION,
            "canonicalizer": CANONICALIZER_VERSION,
            "interpreter": INTERPRETER_VERSION,
            "semantic_hash": SEMANTIC_HASH_VERSION,
            "coding_scheme": PREFIX_CODE_VERSION,
            "residual_code": RESIDUAL_CODE_VERSION,
            "rank": RANK_VERSION,
            "oracle": EXACT_ORACLE_VERSION,
            "simulator": SIMULATOR_VERSION,
            "rollout": ROLLOUT_VERSION,
            "operators": PHASE3_OPERATOR_VERSION,
            "rng": PHASE3_RNG_VERSION,
            "archive": PHASE3_ARCHIVE_VERSION,
            "descriptor": PHASE3_DESCRIPTOR_VERSION,
            "scheduler": PHASE3_SCHEDULER_VERSION,
            "budget": PHASE3_BUDGET_VERSION,
            "initialization": PHASE3_INITIALIZATION_VERSION,
            "analysis": PHASE3_ANALYSIS_VERSION,
            "proposal_artifact": PHASE3_PROPOSAL_ARTIFACT_VERSION,
            "artifact": "immutable-canonical-json-v1",
            "prompt": "not-applicable-no-language-model",
        }
    raw = {
        "manifest_schema_version": manifest_schema,
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
        "versions": versions,
        "proposer": {
            "identifier": config.proposer.proposer_id,
            "model": "mock-v1" if config.schema_version == 1 else "not-applicable",
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
    if config.schema_version == 2:
        if config.dsl is None or config.enumerator is None:
            raise AssertionError("Phase 2 manifest requires DSL and enumerator settings")
        raw["bounds"] = {
            "dsl": {
                "max_depth": config.dsl.max_depth,
                "max_nodes": config.dsl.max_nodes,
                "max_cases": config.dsl.max_cases,
                "allowed_macros": config.dsl.allowed_macros,
            },
            "enumerator": {
                "max_bits": config.enumerator.max_bits,
                "max_depth": config.enumerator.max_depth,
                "max_nodes": config.enumerator.max_nodes,
                "max_candidates": config.enumerator.max_candidates,
                "tie_breaker": "canonical-ast-json-utf8-lexicographic-v1",
                "duplicate_policy": "canonical-then-semantic-first-v1",
                "truth_table_literals_enumerated": False,
            },
        }
        raw["protocol"] = {
            "allowed_oracle_splits": ["training", "development"],
            "validation_consumed_by_phase2": False,
            "test_task_outcomes_accessed": False,
            "active_queries_enabled": False,
        }
    if config.schema_version == 3:
        if (
            config.dsl is None
            or config.operators is None
            or config.archive is None
            or config.scheduler is None
            or config.budget is None
            or config.initialization is None
        ):
            raise AssertionError("Phase 3 manifest requires complete mechanism settings")
        raw["bounds"] = {
            "dsl": {
                "max_depth": config.dsl.max_depth,
                "max_nodes": config.dsl.max_nodes,
                "max_cases": config.dsl.max_cases,
                "allowed_macros": config.dsl.allowed_macros,
            },
            "operator": {
                "path_ordering": "preorder-child-index-v1",
                "weights": dict(config.operators.weights),
                "retry_limit": config.operators.retry_limit,
                "fallback_policy": config.operators.fallback_policy,
                "crossover_fallback": "ordered-self-crossover-v1",
            },
            "archive": {
                "reserve_size": config.archive.reserve_size,
                "node_bin_edges": [3, 7, 15, 31, 63],
                "code_bit_bin_edges": [12, 24, 48, 96, 192],
                "classifier_precedence": [
                    "conditional",
                    "parity",
                    "threshold",
                    "count-based",
                    "position-specific",
                    "mixed-on-overlap",
                ],
                "public_probe": "first-unique-public-local-cases-up-to-16-v1",
                "tie_breaker": "rank-then-lexicographically-smallest-candidate-id-v1",
                "duplicate_policy": "canonical-cell-duplicate-semantic-diagnostic-only-v1",
            },
        }
        raw["budget"] = {
            "version": PHASE3_BUDGET_VERSION,
            "proposal_attempts": config.budget.proposal_attempt_cap,
            "oracle_calls": config.budget.oracle_call_cap,
            "primary_cost_quantum": "one-actual-oracle-invocation",
            "seed_evaluations_charged": True,
            "duplicate_evaluations_charged": True,
            "oracle_cache": False,
            "language_model_calls": 0,
            "language_model_tokens": 0,
            "cpu_seconds": None,
            "elapsed_seconds": None,
        }
        raw["protocol"] = {
            "condition_id": config.run.condition_id,
            "ordinary_allowed_oracle_splits": ["training", "development"],
            "oracle_feedback": "score-only",
            "active_queries_enabled": False,
            "cross_task_memory": False,
            "test_task_outcomes_accessed": False,
        }
    value = to_json_value(raw)
    if not isinstance(value, dict):  # pragma: no cover - mapping invariant
        raise AssertionError("manifest did not serialize as an object")
    return value
