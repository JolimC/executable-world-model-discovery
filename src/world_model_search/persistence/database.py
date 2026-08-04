"""SQLite run ledger for resumable Phase 0 execution."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from world_model_search.domain.types import Candidate, OracleResult, SearchEvent, Task
from world_model_search.errors import PersistenceError
from world_model_search.serialization import canonical_json


@dataclass(frozen=True, slots=True)
class RunState:
    run_id: str
    status: str
    next_step: int


class RunDatabase:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        target = f"file:{path}?mode=ro" if read_only else str(path)
        self.connection = sqlite3.connect(target, uri=read_only)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        if not read_only:
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = FULL")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> RunDatabase:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def initialize(self, run_id: str, task: Task, timestamp: str) -> None:
        schema = """
        CREATE TABLE run_state (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            run_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK (status IN ('running', 'interrupted', 'completed')),
            next_step INTEGER NOT NULL CHECK (next_step >= 0),
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE task (
            task_id TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            split TEXT NOT NULL,
            public_artifact_hash TEXT NOT NULL,
            hidden_artifact_id TEXT NOT NULL,
            generator_version TEXT NOT NULL,
            seed INTEGER NOT NULL
        );
        CREATE TABLE candidate (
            candidate_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES task(task_id),
            ast_json TEXT NOT NULL,
            parent_ids_json TEXT NOT NULL,
            proposer_id TEXT NOT NULL,
            operator_id TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            artifact_name TEXT NOT NULL
        );
        CREATE TABLE evaluation (
            candidate_id TEXT PRIMARY KEY REFERENCES candidate(candidate_id),
            oracle_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            runtime_ns INTEGER NOT NULL
        );
        CREATE TABLE event (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence INTEGER NOT NULL UNIQUE,
            run_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            logical_cost INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            audit_timestamp TEXT NOT NULL
        );
        """
        with self.connection:
            self.connection.executescript(schema)
            self.connection.execute(
                "INSERT INTO run_state VALUES (1, ?, 'running', 0, ?, ?)",
                (run_id, timestamp, timestamp),
            )
            self.connection.execute(
                """INSERT INTO task (
                    task_id, family, split, public_artifact_hash, hidden_artifact_id,
                    generator_version, seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.task_id,
                    task.family,
                    task.split.value,
                    task.public_artifact_hash,
                    task.hidden_artifact_id,
                    task.generator_version,
                    task.seed,
                ),
            )

    def state(self) -> RunState:
        row = self.connection.execute(
            "SELECT run_id, status, next_step FROM run_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise PersistenceError(f"run database has no state: {self.path}")
        return RunState(run_id=row["run_id"], status=row["status"], next_step=row["next_step"])

    def set_status(self, status: str, timestamp: str) -> None:
        if status not in {"running", "interrupted", "completed"}:
            raise ValueError(f"invalid status: {status}")
        with self.connection:
            self.connection.execute(
                "UPDATE run_state SET status = ?, updated_at = ? WHERE singleton = 1",
                (status, timestamp),
            )

    def append_evaluation(
        self,
        *,
        candidate: Candidate,
        artifact_name: str,
        result: OracleResult,
        oracle_version: str,
        event: SearchEvent,
        next_step: int,
    ) -> None:
        result_json = canonical_json(result)
        with self.connection:
            self.connection.execute(
                """INSERT INTO candidate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.candidate_id,
                    candidate.task_id,
                    canonical_json(candidate.ast),
                    canonical_json(candidate.parent_ids),
                    candidate.proposer_id,
                    candidate.operator_id,
                    candidate.context_hash,
                    candidate.payload_hash,
                    artifact_name,
                ),
            )
            self.connection.execute(
                "INSERT INTO evaluation VALUES (?, ?, ?, ?)",
                (candidate.candidate_id, oracle_version, result_json, result.runtime_ns),
            )
            self.connection.execute(
                """INSERT INTO event (
                    sequence, run_id, event_type, logical_cost, payload_json,
                    payload_hash, audit_timestamp
                ) SELECT ?, run_id, ?, ?, ?, ?, ? FROM run_state WHERE singleton = 1""",
                (
                    event.sequence,
                    event.event_type,
                    event.logical_cost,
                    event.payload_json,
                    event.payload_hash,
                    event.audit_timestamp,
                ),
            )
            self.connection.execute(
                "UPDATE run_state SET next_step = ?, updated_at = ? WHERE singleton = 1",
                (next_step, event.audit_timestamp),
            )

    def events(self) -> tuple[SearchEvent, ...]:
        rows = self.connection.execute(
            """SELECT sequence, event_type, logical_cost, payload_json,
                      payload_hash, audit_timestamp
               FROM event ORDER BY sequence"""
        ).fetchall()
        return tuple(
            SearchEvent(
                sequence=row["sequence"],
                event_type=row["event_type"],
                logical_cost=row["logical_cost"],
                payload_json=row["payload_json"],
                payload_hash=row["payload_hash"],
                audit_timestamp=row["audit_timestamp"],
            )
            for row in rows
        )

    def candidate_record(self, candidate_id: str) -> sqlite3.Row:
        row = cast(
            sqlite3.Row | None,
            self.connection.execute(
                "SELECT * FROM candidate WHERE candidate_id = ?", (candidate_id,)
            ).fetchone(),
        )
        if row is None:
            raise PersistenceError(f"missing candidate record: {candidate_id}")
        return row
