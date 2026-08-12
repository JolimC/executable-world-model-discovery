"""Immutable typed values for Phase 5 memory and evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from world_model_search.domain.types import SplitLabel
from world_model_search.errors import PersistenceError
from world_model_search.phase5_versions import PHASE5_MEMORY_SNAPSHOT_VERSION
from world_model_search.serialization import JsonObject, sha256_json


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    HYPOTHESIS = "hypothesis"
    SEARCH_LESSON = "search-lesson"
    PRIMITIVE_PROPOSAL = "primitive-proposal"
    SELF_MODEL = "self-model"


class ValidationState(StrEnum):
    PROPOSED = "proposed"
    SCOPED = "scoped"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    INVALIDATED = "invalidated"


class EvidencePolarity(StrEnum):
    SUPPORT = "support"
    VALIDATION = "validation"
    COUNTER = "counter"


def _hash(value: str, field: str) -> None:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class EvidenceFact:
    """Evaluator-only immutable provenance for one independent task observation."""

    task_id: str
    family_id: str
    role: SplitLabel
    semantic_hash: str
    run_hash: str
    candidate_hash: str
    evaluation_hash: str
    artifact_hash: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.family_id:
            raise ValueError("evidence task and family identity must be nonempty")
        for name in (
            "semantic_hash",
            "run_hash",
            "candidate_hash",
            "evaluation_hash",
            "artifact_hash",
        ):
            _hash(getattr(self, name), name)
        if self.role is SplitLabel.TEST:
            raise ValueError("sealed test evidence cannot enter Phase 5 memory")

    def to_value(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "family_id": self.family_id,
            "role": self.role.value,
            "semantic_hash": self.semantic_hash,
            "run_hash": self.run_hash,
            "candidate_hash": self.candidate_hash,
            "evaluation_hash": self.evaluation_hash,
            "artifact_hash": self.artifact_hash,
        }

    @property
    def evidence_id(self) -> str:
        return sha256_json({"evidence_identity_version": "phase5-evidence-v1", **self.to_value()})


@dataclass(frozen=True, slots=True)
class MemoryApplicability:
    world_specification_version: str
    dsl_version: str
    requires_exact_feedback: bool = False

    def __post_init__(self) -> None:
        if not self.world_specification_version or not self.dsl_version:
            raise ValueError("memory applicability fields must be nonempty")

    def to_value(self) -> JsonObject:
        return {
            "world_specification_version": self.world_specification_version,
            "dsl_version": self.dsl_version,
            "requires_exact_feedback": self.requires_exact_feedback,
        }


@dataclass(frozen=True, slots=True)
class SafeMemoryItem:
    """The only memory record shape allowed across the proposer boundary."""

    record_id: str
    kind: MemoryKind
    proposer_text: str
    scope: str
    applicability: MemoryApplicability

    def to_value(self) -> JsonObject:
        return {
            "record_id": self.record_id,
            "kind": self.kind.value,
            "text": self.proposer_text,
            "scope": self.scope,
            "applicability": self.applicability.to_value(),
        }


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    split_registry_hash: str
    database_export_hash: str
    items: tuple[SafeMemoryItem, ...]

    def __post_init__(self) -> None:
        _hash(self.split_registry_hash, "split_registry_hash")
        _hash(self.database_export_hash, "database_export_hash")
        if tuple(sorted(self.items, key=lambda item: item.record_id)) != self.items:
            raise ValueError("memory snapshot items must be deterministically ordered")

    def to_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "snapshot_version": PHASE5_MEMORY_SNAPSHOT_VERSION,
                "split_registry_hash": self.split_registry_hash,
                "database_export_hash": self.database_export_hash,
                "items": [item.to_value() for item in self.items],
            },
        )

    @property
    def snapshot_hash(self) -> str:
        return sha256_json(self.to_value())


def load_memory_snapshot(path: Path) -> MemorySnapshot:
    """Load and verify a proposer-safe frozen memory snapshot artifact."""

    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceError("memory snapshot artifact is unavailable or corrupt") from exc
    if (
        not isinstance(value, dict)
        or value.get("snapshot_version") != PHASE5_MEMORY_SNAPSHOT_VERSION
    ):
        raise PersistenceError("memory snapshot version is unsupported")
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        raise PersistenceError("memory snapshot items are malformed")
    items: list[SafeMemoryItem] = []
    try:
        for raw_item in raw_items:
            if not isinstance(raw_item, dict) or set(raw_item) != {
                "record_id",
                "kind",
                "text",
                "scope",
                "applicability",
            }:
                raise ValueError("memory item fields are malformed")
            applicability = raw_item["applicability"]
            if not isinstance(applicability, dict) or set(applicability) != {
                "world_specification_version",
                "dsl_version",
                "requires_exact_feedback",
            }:
                raise ValueError("memory applicability is malformed")
            requires_exact = applicability["requires_exact_feedback"]
            if not isinstance(requires_exact, bool):
                raise ValueError("memory exact-feedback applicability must be boolean")
            items.append(
                SafeMemoryItem(
                    record_id=str(raw_item["record_id"]),
                    kind=MemoryKind(str(raw_item["kind"])),
                    proposer_text=str(raw_item["text"]),
                    scope=str(raw_item["scope"]),
                    applicability=MemoryApplicability(
                        str(applicability["world_specification_version"]),
                        str(applicability["dsl_version"]),
                        requires_exact,
                    ),
                )
            )
        snapshot = MemorySnapshot(
            split_registry_hash=str(value["split_registry_hash"]),
            database_export_hash=str(value["database_export_hash"]),
            items=tuple(items),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistenceError("memory snapshot content is invalid") from exc
    if snapshot.snapshot_hash != value.get("snapshot_hash"):
        raise PersistenceError("memory snapshot content identity mismatch")
    return snapshot
