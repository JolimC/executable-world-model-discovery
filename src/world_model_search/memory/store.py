"""Transactional, content-addressed SQLite store for Phase 5 memory."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from world_model_search.domain.types import SplitLabel
from world_model_search.errors import PersistenceError
from world_model_search.memory.types import (
    EvidenceFact,
    EvidencePolarity,
    MemoryApplicability,
    MemoryKind,
    MemorySnapshot,
    SafeMemoryItem,
    ValidationState,
)
from world_model_search.persistence.artifacts import write_content_artifact
from world_model_search.phase5_versions import (
    PHASE5_MEMORY_EXPORT_VERSION,
    PHASE5_MEMORY_SCHEMA_VERSION,
)
from world_model_search.serialization import JsonObject, canonical_json, sha256_json, sha256_text

_EVENTS = frozenset({"proposal", "promotion", "rejection", "scoping", "invalidation"})
_FORBIDDEN_SAFE_TERMS = frozenset(
    {
        "generator_family",
        "family_id",
        "reference_rule",
        "reference_ast",
        "semantic_hash",
        "hidden_artifact",
        "oracle_handle",
        "artifact_path",
    }
)


def _parse_object(data: str, location: str) -> JsonObject:
    try:
        value: object = json.loads(data)
    except json.JSONDecodeError as exc:
        raise PersistenceError(f"{location} JSON is corrupt") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PersistenceError(f"{location} is not an object")
    return cast(JsonObject, value)


class Phase5MemoryStore:
    """A separate schema that refuses migration/version ambiguity."""

    def __init__(
        self,
        path: Path,
        *,
        split_registry_hash: str,
        evidence_catalog: Mapping[str, EvidenceFact] | None = None,
        read_only: bool = False,
    ) -> None:
        self.path = path
        self.split_registry_hash = split_registry_hash
        self.evidence_catalog = dict(evidence_catalog or {})
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        target = f"file:{path}?mode=ro" if read_only else str(path)
        try:
            self.connection = sqlite3.connect(target, uri=read_only)
        except sqlite3.Error as exc:
            raise PersistenceError("Phase 5 memory database cannot be opened") from exc
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
        tables = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        if not tables:
            if read_only:
                raise PersistenceError("Phase 5 memory database is uninitialized")
            self._initialize()
        self._verify_metadata()

    def __enter__(self) -> Phase5MemoryStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE evidence(
            evidence_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            family_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('training','development')),
            content_json TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE
        );
        CREATE TABLE memory_record(
            record_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            proposer_text TEXT NOT NULL,
            scope TEXT NOT NULL,
            applicability_json TEXT NOT NULL,
            definition_cost_bits INTEGER,
            provenance_json TEXT NOT NULL,
            content_json TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE
        );
        CREATE TABLE evidence_link(
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT NOT NULL REFERENCES memory_record(record_id),
            evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
            polarity TEXT NOT NULL CHECK(polarity IN ('support','validation','counter')),
            link_hash TEXT NOT NULL UNIQUE,
            UNIQUE(record_id,evidence_id,polarity)
        );
        CREATE TABLE lifecycle_event(
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT NOT NULL REFERENCES memory_record(record_id),
            event_type TEXT NOT NULL CHECK(
                event_type IN ('proposal','promotion','rejection','scoping','invalidation')
            ),
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL UNIQUE
        );
        """
        catalog_hash = sha256_json(
            [self.evidence_catalog[key].to_value() for key in sorted(self.evidence_catalog)]
        )
        try:
            with self.connection:
                self.connection.executescript(schema)
                self.connection.executemany(
                    "INSERT INTO metadata VALUES (?,?)",
                    (
                        ("database_schema_version", str(PHASE5_MEMORY_SCHEMA_VERSION)),
                        ("split_registry_hash", self.split_registry_hash),
                        ("evidence_catalog_hash", catalog_hash),
                    ),
                )
        except sqlite3.Error as exc:
            raise PersistenceError("Phase 5 memory initialization failed") from exc

    def _verify_metadata(self) -> None:
        try:
            rows = self.connection.execute("SELECT key,value FROM metadata").fetchall()
        except sqlite3.Error as exc:
            raise PersistenceError("Phase 5 memory schema is unsupported") from exc
        metadata = {str(row["key"]): str(row["value"]) for row in rows}
        if metadata.get("database_schema_version") != str(PHASE5_MEMORY_SCHEMA_VERSION):
            raise PersistenceError("Phase 5 memory schema version mismatch; migration refused")
        if metadata.get("split_registry_hash") != self.split_registry_hash:
            raise PersistenceError("Phase 5 memory split registry identity mismatch")
        if set(metadata) != {
            "database_schema_version",
            "split_registry_hash",
            "evidence_catalog_hash",
        }:
            raise PersistenceError("Phase 5 memory metadata is malformed")

    def admit_evidence(self, fact: EvidenceFact) -> str:
        expected = self.evidence_catalog.get(fact.evidence_id)
        if expected != fact:
            raise PersistenceError("memory evidence is absent from the frozen eligible catalog")
        content = {
            "evidence_identity_version": "phase5-evidence-v1",
            **fact.to_value(),
        }
        digest = sha256_json(content)
        if digest != fact.evidence_id:
            raise PersistenceError("memory evidence identity does not reconcile")
        with self.connection:
            prior = self.connection.execute(
                "SELECT content_json FROM evidence WHERE evidence_id=?", (digest,)
            ).fetchone()
            if prior is not None:
                if prior["content_json"] != canonical_json(content):
                    raise PersistenceError("memory evidence identity collision")
                return digest
            self.connection.execute(
                "INSERT INTO evidence VALUES (?,?,?,?,?,?)",
                (
                    digest,
                    fact.task_id,
                    fact.family_id,
                    fact.role.value,
                    canonical_json(content),
                    digest,
                ),
            )
        return digest

    @staticmethod
    def _safe_text(value: str) -> None:
        if not value or len(value.encode("utf-8")) > 4096:
            raise ValueError("proposer-safe memory text must contain 1-4096 UTF-8 bytes")
        lowered = value.casefold()
        if any(term in lowered for term in _FORBIDDEN_SAFE_TERMS):
            raise ValueError("proposer-safe memory text names an evaluator-only field")

    def propose_record(
        self,
        *,
        kind: MemoryKind,
        proposer_text: str,
        scope: str,
        applicability: MemoryApplicability,
        support_evidence_ids: Sequence[str],
        provenance_hashes: Sequence[str],
        definition_cost_bits: int | None = None,
    ) -> str:
        self._safe_text(proposer_text)
        if not scope or len(scope) > 200:
            raise ValueError("memory scope is invalid")
        supports = tuple(sorted(set(support_evidence_ids)))
        provenance = tuple(sorted(set(provenance_hashes)))
        if not supports or len(supports) != len(support_evidence_ids):
            raise ValueError("memory proposal needs unique supporting evidence")
        if not provenance or any(len(item) != 64 for item in provenance):
            raise ValueError("memory proposal needs complete immutable provenance hashes")
        if definition_cost_bits is not None and (
            isinstance(definition_cost_bits, bool) or definition_cost_bits < 1
        ):
            raise ValueError("memory definition cost must be positive")
        placeholders = ",".join("?" for _ in supports)
        rows = self.connection.execute(
            f"SELECT evidence_id,role FROM evidence WHERE evidence_id IN ({placeholders})",
            supports,
        ).fetchall()
        if len(rows) != len(supports) or any(
            row["role"] != SplitLabel.TRAINING.value for row in rows
        ):
            raise PersistenceError("memory proposals require admitted training-family evidence")
        content: JsonObject = {
            "record_identity_version": "phase5-memory-record-v1",
            "kind": kind.value,
            "proposer_text": proposer_text,
            "scope": scope,
            "applicability": applicability.to_value(),
            "initial_support_evidence_ids": list(supports),
            "provenance_hashes": list(provenance),
            "definition_cost_bits": definition_cost_bits,
        }
        record_id = sha256_json(content)
        event_payload: JsonObject = {
            "record_id": record_id,
            "initial_state": ValidationState.PROPOSED.value,
            "support_evidence_ids": list(supports),
        }
        with self.connection:
            prior = self.connection.execute(
                "SELECT content_json FROM memory_record WHERE record_id=?", (record_id,)
            ).fetchone()
            if prior is not None:
                if prior["content_json"] != canonical_json(content):
                    raise PersistenceError("memory record identity collision")
                return record_id
            self.connection.execute(
                "INSERT INTO memory_record VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record_id,
                    kind.value,
                    proposer_text,
                    scope,
                    canonical_json(applicability.to_value()),
                    definition_cost_bits,
                    canonical_json(list(provenance)),
                    canonical_json(content),
                    record_id,
                ),
            )
            for evidence_id in supports:
                self._insert_link(record_id, evidence_id, EvidencePolarity.SUPPORT)
            self._insert_event(record_id, "proposal", event_payload)
        return record_id

    def _insert_link(self, record_id: str, evidence_id: str, polarity: EvidencePolarity) -> None:
        payload = {
            "link_version": "phase5-evidence-link-v1",
            "record_id": record_id,
            "evidence_id": evidence_id,
            "polarity": polarity.value,
        }
        self.connection.execute(
            "INSERT INTO evidence_link(record_id,evidence_id,polarity,link_hash) VALUES (?,?,?,?)",
            (record_id, evidence_id, polarity.value, sha256_json(payload)),
        )

    def _insert_event(self, record_id: str, event_type: str, payload: JsonObject) -> None:
        if event_type not in _EVENTS:
            raise ValueError("unknown Phase 5 lifecycle event")
        envelope: JsonObject = {
            "event_version": "phase5-memory-lifecycle-event-v1",
            "record_id": record_id,
            "event_type": event_type,
            "payload": payload,
        }
        self.connection.execute(
            """INSERT INTO lifecycle_event(
                   record_id,event_type,payload_json,payload_hash
               ) VALUES (?,?,?,?)""",
            (record_id, event_type, canonical_json(envelope), sha256_json(envelope)),
        )

    def _state(self, record_id: str) -> ValidationState:
        rows = self.connection.execute(
            "SELECT event_type FROM lifecycle_event WHERE record_id=? ORDER BY sequence",
            (record_id,),
        ).fetchall()
        if not rows or rows[0]["event_type"] != "proposal":
            raise PersistenceError("memory record has no proposal event")
        state = ValidationState.PROPOSED
        transitions = {
            "scoping": ValidationState.SCOPED,
            "promotion": ValidationState.PROMOTED,
            "rejection": ValidationState.REJECTED,
            "invalidation": ValidationState.INVALIDATED,
        }
        for row in rows[1:]:
            event = str(row["event_type"])
            if event == "proposal" or state in {
                ValidationState.REJECTED,
                ValidationState.INVALIDATED,
            }:
                raise PersistenceError("memory lifecycle transition is invalid")
            if state is ValidationState.PROMOTED and event != "invalidation":
                raise PersistenceError("promoted memory may only be invalidated")
            state = transitions[event]
        return state

    def link_evidence(
        self,
        record_id: str,
        evidence_id: str,
        polarity: EvidencePolarity,
    ) -> None:
        row = self.connection.execute(
            "SELECT role FROM evidence WHERE evidence_id=?", (evidence_id,)
        ).fetchone()
        if row is None:
            raise PersistenceError("linked memory evidence was not admitted")
        if polarity is EvidencePolarity.VALIDATION and row["role"] != SplitLabel.DEVELOPMENT.value:
            raise PersistenceError("primitive validation evidence must be development-family data")
        if polarity is EvidencePolarity.SUPPORT and row["role"] != SplitLabel.TRAINING.value:
            raise PersistenceError("memory support evidence must be training-family data")
        state = self._state(record_id)
        if state in {ValidationState.REJECTED, ValidationState.INVALIDATED}:
            raise PersistenceError("terminal memory records cannot receive evidence")
        prior = self.connection.execute(
            """SELECT 1 FROM evidence_link
               WHERE record_id=? AND evidence_id=? AND polarity=?""",
            (record_id, evidence_id, polarity.value),
        ).fetchone()
        if prior is not None:
            return
        try:
            with self.connection:
                self._insert_link(record_id, evidence_id, polarity)
                if polarity is EvidencePolarity.COUNTER and state is ValidationState.PROMOTED:
                    self._insert_event(
                        record_id,
                        "invalidation",
                        {"reason": "counterevidence-after-promotion", "evidence_id": evidence_id},
                    )
        except sqlite3.IntegrityError as exc:
            raise PersistenceError("duplicate or inconsistent memory evidence link") from exc

    def independent_support(self, record_id: str, polarity: EvidencePolarity) -> tuple[int, int]:
        rows = self.connection.execute(
            """SELECT DISTINCT e.task_id,e.family_id FROM evidence_link l
               JOIN evidence e ON e.evidence_id=l.evidence_id
               WHERE l.record_id=? AND l.polarity=?""",
            (record_id, polarity.value),
        ).fetchall()
        return len({str(row["task_id"]) for row in rows}), len(
            {str(row["family_id"]) for row in rows}
        )

    def transition(
        self,
        record_id: str,
        state: ValidationState,
        *,
        reason: str,
        gate_payload: JsonObject | None = None,
    ) -> None:
        current = self._state(record_id)
        if current is state:
            return
        if current in {ValidationState.REJECTED, ValidationState.INVALIDATED}:
            raise PersistenceError("terminal memory lifecycle cannot transition")
        events = {
            ValidationState.SCOPED: "scoping",
            ValidationState.PROMOTED: "promotion",
            ValidationState.REJECTED: "rejection",
            ValidationState.INVALIDATED: "invalidation",
        }
        if state not in events or current is ValidationState.PROMOTED:
            raise PersistenceError("requested memory lifecycle transition is invalid")
        if not reason:
            raise ValueError("memory lifecycle transition requires a reason")
        if state is ValidationState.PROMOTED:
            support_tasks, support_families = self.independent_support(
                record_id, EvidencePolarity.SUPPORT
            )
            counters = self.connection.execute(
                "SELECT COUNT(*) FROM evidence_link WHERE record_id=? AND polarity='counter'",
                (record_id,),
            ).fetchone()[0]
            kind = self.connection.execute(
                "SELECT kind,scope FROM memory_record WHERE record_id=?", (record_id,)
            ).fetchone()
            if kind is None:
                raise PersistenceError("memory record is unavailable")
            if support_tasks < 2 or support_families < 2 or counters:
                raise PersistenceError(
                    "global promotion requires two independent training tasks/families "
                    "and no counterevidence"
                )
            if kind["scope"] != "global-f0":
                raise PersistenceError("only global-f0 memory may be globally promoted")
            if kind["kind"] == MemoryKind.PRIMITIVE_PROPOSAL.value:
                validation_tasks, validation_families = self.independent_support(
                    record_id, EvidencePolarity.VALIDATION
                )
                if validation_tasks < 2 or validation_families < 2:
                    raise PersistenceError(
                        "primitive promotion requires two independent development tasks/families"
                    )
                if (
                    gate_payload is None
                    or gate_payload.get("strictly_positive_net_gain") is not True
                ):
                    raise PersistenceError("primitive promotion requires the positive net-MDL gate")
        payload: JsonObject = {"reason": reason, "target_state": state.value}
        if gate_payload is not None:
            payload["gate"] = gate_payload
        with self.connection:
            self._insert_event(record_id, events[state], payload)

    def deterministic_export(self) -> JsonObject:
        self.audit()

        def rows(table: str, order: str) -> list[JsonObject]:
            result: list[JsonObject] = []
            for row in self.connection.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                result.append(dict(zip(row.keys(), tuple(row), strict=True)))
            return result

        return cast(
            JsonObject,
            {
                "export_version": PHASE5_MEMORY_EXPORT_VERSION,
                "metadata": rows("metadata", "key"),
                "evidence": rows("evidence", "evidence_id"),
                "records": rows("memory_record", "record_id"),
                "evidence_links": rows("evidence_link", "sequence"),
                "lifecycle_events": rows("lifecycle_event", "sequence"),
            },
        )

    def audit(self) -> None:
        integrity = self.connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise PersistenceError("Phase 5 memory SQLite integrity check failed")
        self._verify_metadata()
        for row in self.connection.execute("SELECT * FROM evidence"):
            content = _parse_object(row["content_json"], "memory evidence")
            if (
                sha256_json(content) != row["evidence_id"]
                or row["content_hash"] != row["evidence_id"]
            ):
                raise PersistenceError("memory evidence content identity mismatch")
        for row in self.connection.execute("SELECT * FROM memory_record"):
            content = _parse_object(row["content_json"], "memory record")
            if sha256_json(content) != row["record_id"] or row["content_hash"] != row["record_id"]:
                raise PersistenceError("memory record content identity mismatch")
            self._safe_text(str(row["proposer_text"]))
            self._state(str(row["record_id"]))
        for row in self.connection.execute("SELECT payload_json,payload_hash FROM lifecycle_event"):
            if sha256_text(row["payload_json"]) != row["payload_hash"]:
                raise PersistenceError("memory lifecycle event hash mismatch")
        for row in self.connection.execute(
            "SELECT record_id,evidence_id,polarity,link_hash FROM evidence_link"
        ):
            payload = {
                "link_version": "phase5-evidence-link-v1",
                "record_id": row["record_id"],
                "evidence_id": row["evidence_id"],
                "polarity": row["polarity"],
            }
            if sha256_json(payload) != row["link_hash"]:
                raise PersistenceError("memory evidence link hash mismatch")
        missing = self.connection.execute(
            """SELECT COUNT(*) FROM memory_record r
               LEFT JOIN evidence_link l ON l.record_id=r.record_id AND l.polarity='support'
               WHERE l.record_id IS NULL"""
        ).fetchone()[0]
        if missing:
            raise PersistenceError("memory record has missing provenance evidence")

    def freeze_snapshot(self, path: Path) -> MemorySnapshot:
        export = self.deterministic_export()
        items: list[SafeMemoryItem] = []
        for row in self.connection.execute("SELECT * FROM memory_record ORDER BY record_id"):
            if self._state(str(row["record_id"])) is not ValidationState.PROMOTED:
                continue
            applicability_raw = _parse_object(row["applicability_json"], "memory applicability")
            applicability = MemoryApplicability(
                str(applicability_raw["world_specification_version"]),
                str(applicability_raw["dsl_version"]),
                bool(applicability_raw["requires_exact_feedback"]),
            )
            items.append(
                SafeMemoryItem(
                    str(row["record_id"]),
                    MemoryKind(str(row["kind"])),
                    str(row["proposer_text"]),
                    str(row["scope"]),
                    applicability,
                )
            )
        snapshot = MemorySnapshot(
            self.split_registry_hash,
            sha256_json(export),
            tuple(items),
        )
        artifact: JsonObject = {**snapshot.to_value(), "snapshot_hash": snapshot.snapshot_hash}
        write_content_artifact(path, canonical_json(artifact))
        return snapshot
