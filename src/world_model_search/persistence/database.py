"""SQLite run ledger for resumable Phase 0 execution."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from world_model_search.domain.types import Candidate, OracleResult, SearchEvent, Task
from world_model_search.dsl.ast import BitExpr
from world_model_search.dsl.json_schema import ast_canonical_json
from world_model_search.dsl.versions import PHASE3_DATABASE_SCHEMA_VERSION
from world_model_search.errors import PersistenceError
from world_model_search.serialization import JsonObject, canonical_json

if TYPE_CHECKING:
    from world_model_search.scheduler.uniform import SchedulerDecision
    from world_model_search.search.archive import ArchiveDecision
    from world_model_search.search.phase3_types import BudgetState


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
            internal_family_id TEXT NOT NULL,
            public_world_spec_json TEXT NOT NULL,
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
                    task_id, internal_family_id, public_world_spec_json, split,
                    public_artifact_hash, hidden_artifact_id, generator_version, seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.task_id,
                    task.internal_family_id,
                    canonical_json(task.public_world_spec),
                    task.split.value,
                    task.public_artifact_hash,
                    task.hidden_artifact_id,
                    task.generator_version,
                    task.seed,
                ),
            )

    def initialize_phase2(self, run_id: str, task: Task, timestamp: str) -> None:
        """Create the deliberately versioned Phase 2 ledger schema."""

        schema = """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
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
            internal_family_id TEXT NOT NULL,
            public_world_spec_json TEXT NOT NULL,
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
            artifact_name TEXT NOT NULL,
            candidate_schema_version INTEGER NOT NULL,
            source_ast_json TEXT NOT NULL,
            canonical_ast_json TEXT NOT NULL,
            semantic_hash TEXT NOT NULL,
            ast_bits INTEGER NOT NULL CHECK (ast_bits > 0)
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
            self.connection.execute("INSERT INTO metadata VALUES ('database_schema_version', '2')")
            self.connection.execute(
                "INSERT INTO run_state VALUES (1, ?, 'running', 0, ?, ?)",
                (run_id, timestamp, timestamp),
            )
            self.connection.execute(
                """INSERT INTO task (
                    task_id, internal_family_id, public_world_spec_json, split,
                    public_artifact_hash, hidden_artifact_id, generator_version, seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.task_id,
                    task.internal_family_id,
                    canonical_json(task.public_world_spec),
                    task.split.value,
                    task.public_artifact_hash,
                    task.hidden_artifact_id,
                    task.generator_version,
                    task.seed,
                ),
            )

    def initialize_phase3(
        self,
        run_id: str,
        task: Task,
        timestamp: str,
        *,
        proposal_attempt_cap: int,
        oracle_call_cap: int,
    ) -> None:
        """Create the deliberately separate Phase 3 schema."""

        schema = """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
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
            internal_family_id TEXT NOT NULL,
            public_world_spec_json TEXT NOT NULL,
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
            artifact_name TEXT NOT NULL,
            candidate_schema_version INTEGER NOT NULL,
            identity_schema TEXT NOT NULL,
            source_ast_json TEXT NOT NULL,
            canonical_ast_json TEXT NOT NULL,
            semantic_hash TEXT NOT NULL,
            ast_bits INTEGER NOT NULL CHECK (ast_bits > 0),
            first_attempt_index INTEGER NOT NULL UNIQUE
        );
        CREATE TABLE proposal_attempt (
            attempt_index INTEGER PRIMARY KEY,
            scheduler_json TEXT NOT NULL,
            operator_json TEXT NOT NULL,
            artifact_name TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            outcome TEXT NOT NULL,
            candidate_id TEXT REFERENCES candidate(candidate_id),
            canonical_duplicate INTEGER NOT NULL CHECK (canonical_duplicate IN (0, 1)),
            semantic_duplicate INTEGER NOT NULL CHECK (semantic_duplicate IN (0, 1))
        );
        CREATE TABLE evaluation (
            attempt_index INTEGER PRIMARY KEY REFERENCES proposal_attempt(attempt_index),
            candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
            oracle_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            runtime_ns INTEGER NOT NULL
        );
        CREATE TABLE archive_transition (
            attempt_index INTEGER PRIMARY KEY REFERENCES proposal_attempt(attempt_index),
            decision_json TEXT NOT NULL,
            decision_hash TEXT NOT NULL,
            coordinate_json TEXT NOT NULL,
            outcome TEXT NOT NULL,
            role TEXT,
            inserted_candidate_id TEXT,
            replaced_candidate_id TEXT,
            evicted_candidate_id TEXT
        );
        CREATE TABLE lineage_edge (
            child_candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
            parent_candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
            parent_order INTEGER NOT NULL CHECK (parent_order >= 0),
            PRIMARY KEY (child_candidate_id, parent_order)
        );
        CREATE TABLE budget_state (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            state_json TEXT NOT NULL,
            proposal_attempts INTEGER NOT NULL,
            oracle_invocations INTEGER NOT NULL,
            proposal_attempt_cap INTEGER NOT NULL,
            oracle_call_cap INTEGER NOT NULL
        );
        CREATE TABLE attempt_diagnostic (
            attempt_index INTEGER PRIMARY KEY REFERENCES proposal_attempt(attempt_index),
            attempt_cpu_ns INTEGER NOT NULL CHECK (attempt_cpu_ns >= 0),
            oracle_cpu_ns INTEGER NOT NULL CHECK (oracle_cpu_ns >= 0),
            attempt_elapsed_ns INTEGER NOT NULL CHECK (attempt_elapsed_ns >= 0),
            oracle_elapsed_ns INTEGER NOT NULL CHECK (oracle_elapsed_ns >= 0),
            language_model_calls INTEGER NOT NULL CHECK (language_model_calls >= 0),
            language_model_tokens INTEGER NOT NULL CHECK (language_model_tokens >= 0)
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
        from world_model_search.search.phase3_types import BudgetState

        initial_budget = BudgetState(proposal_attempt_cap, oracle_call_cap)
        with self.connection:
            self.connection.executescript(schema)
            self.connection.execute(
                "INSERT INTO metadata VALUES ('database_schema_version', ?)",
                (str(PHASE3_DATABASE_SCHEMA_VERSION),),
            )
            self.connection.execute(
                "INSERT INTO run_state VALUES (1, ?, 'running', 0, ?, ?)",
                (run_id, timestamp, timestamp),
            )
            self.connection.execute(
                """INSERT INTO task VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.task_id,
                    task.internal_family_id,
                    canonical_json(task.public_world_spec),
                    task.split.value,
                    task.public_artifact_hash,
                    task.hidden_artifact_id,
                    task.generator_version,
                    task.seed,
                ),
            )
            self.connection.execute(
                "INSERT INTO budget_state VALUES (1, ?, 0, 0, ?, ?)",
                (
                    canonical_json(initial_budget.to_value()),
                    proposal_attempt_cap,
                    oracle_call_cap,
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

    def append_phase2_evaluation(
        self,
        *,
        candidate: Candidate,
        source_ast: BitExpr,
        artifact_name: str,
        result: OracleResult,
        oracle_version: str,
        event: SearchEvent,
        next_step: int,
    ) -> None:
        if not isinstance(candidate.ast, BitExpr) or candidate.semantic_hash is None:
            raise TypeError("Phase 2 persistence requires a typed semantic candidate")
        result_json = canonical_json(result)
        with self.connection:
            self.connection.execute(
                """INSERT INTO candidate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.candidate_id,
                    candidate.task_id,
                    ast_canonical_json(candidate.ast),
                    canonical_json(candidate.parent_ids),
                    candidate.proposer_id,
                    candidate.operator_id,
                    candidate.context_hash,
                    candidate.payload_hash,
                    artifact_name,
                    1,
                    ast_canonical_json(source_ast),
                    ast_canonical_json(candidate.ast),
                    candidate.semantic_hash,
                    result.ast_bits,
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

    def phase3_budget(self) -> BudgetState:
        import json

        from world_model_search.search.phase3_types import BudgetState

        row = self.connection.execute(
            "SELECT state_json FROM budget_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise PersistenceError("Phase 3 database has no budget state")
        try:
            raw: object = json.loads(row["state_json"])
            return BudgetState.from_value(raw)
        except (ValueError, TypeError) as exc:
            raise PersistenceError("Phase 3 budget state is invalid") from exc

    def append_phase3_step(
        self,
        *,
        attempt_index: int,
        candidate: Candidate | None,
        source_ast: BitExpr | None,
        artifact_name: str,
        artifact_hash: str,
        scheduler: SchedulerDecision,
        operator_json: JsonObject,
        attempt_outcome: str,
        canonical_duplicate: bool,
        semantic_duplicate: bool,
        result: OracleResult | None,
        oracle_version: str,
        decision: ArchiveDecision | None,
        budget: BudgetState,
        event: SearchEvent,
        next_step: int,
        attempt_cpu_ns: int = 0,
        oracle_cpu_ns: int = 0,
        attempt_elapsed_ns: int = 0,
        oracle_elapsed_ns: int = 0,
    ) -> None:
        """Atomically commit attempt, evaluation, transition, budget, event, and resume state."""

        from world_model_search.dsl.codec import encoded_length
        from world_model_search.dsl.versions import (
            CANDIDATE_SCHEMA_VERSION,
            PHASE3_CANDIDATE_IDENTITY_VERSION,
        )

        if (candidate is None) != (result is None) or (result is None) != (decision is None):
            raise ValueError("candidate, evaluation, and transition must be present together")
        if event.sequence != attempt_index or next_step != attempt_index + 1:
            raise ValueError("Phase 3 step sequence is inconsistent")
        diagnostic_values = (
            attempt_cpu_ns,
            oracle_cpu_ns,
            attempt_elapsed_ns,
            oracle_elapsed_ns,
        )
        if any(value < 0 for value in diagnostic_values) or oracle_cpu_ns > attempt_cpu_ns:
            raise ValueError("Phase 3 timing diagnostics are inconsistent")
        budget.validate()
        with self.connection:
            if candidate is not None and result is not None and source_ast is not None:
                if not isinstance(candidate.ast, BitExpr) or candidate.semantic_hash is None:
                    raise TypeError("Phase 3 candidate must be a typed semantic AST")
                existing = self.connection.execute(
                    "SELECT * FROM candidate WHERE candidate_id = ?", (candidate.candidate_id,)
                ).fetchone()
                if existing is None:
                    for parent_id in candidate.parent_ids:
                        parent = self.connection.execute(
                            """SELECT task_id, first_attempt_index FROM candidate
                               WHERE candidate_id = ?""",
                            (parent_id,),
                        ).fetchone()
                        if (
                            parent is None
                            or parent["task_id"] != candidate.task_id
                            or parent["first_attempt_index"] >= attempt_index
                        ):
                            raise PersistenceError(
                                "Phase 3 parent must exist earlier in the same task/run"
                            )
                        if parent_id == candidate.candidate_id:
                            raise PersistenceError("Phase 3 lineage cycle is forbidden")
                    self.connection.execute(
                        """INSERT INTO candidate VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )""",
                        (
                            candidate.candidate_id,
                            candidate.task_id,
                            ast_canonical_json(candidate.ast),
                            canonical_json(candidate.parent_ids),
                            candidate.proposer_id,
                            candidate.operator_id,
                            candidate.context_hash,
                            candidate.payload_hash,
                            artifact_name,
                            CANDIDATE_SCHEMA_VERSION,
                            PHASE3_CANDIDATE_IDENTITY_VERSION,
                            ast_canonical_json(source_ast),
                            ast_canonical_json(candidate.ast),
                            candidate.semantic_hash,
                            encoded_length(candidate.ast),
                            attempt_index,
                        ),
                    )
                    self.connection.executemany(
                        "INSERT INTO lineage_edge VALUES (?, ?, ?)",
                        tuple(
                            (candidate.candidate_id, parent_id, parent_order)
                            for parent_order, parent_id in enumerate(candidate.parent_ids)
                        ),
                    )
                elif (
                    existing["task_id"] != candidate.task_id
                    or existing["payload_hash"] != candidate.payload_hash
                    or existing["parent_ids_json"] != canonical_json(candidate.parent_ids)
                ):
                    raise PersistenceError("candidate identity collision")
            self.connection.execute(
                """INSERT INTO proposal_attempt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    attempt_index,
                    canonical_json(scheduler.to_value()),
                    canonical_json(operator_json),
                    artifact_name,
                    artifact_hash,
                    attempt_outcome,
                    candidate.candidate_id if candidate is not None else None,
                    int(canonical_duplicate),
                    int(semantic_duplicate),
                ),
            )
            if candidate is not None and result is not None and decision is not None:
                self.connection.execute(
                    "INSERT INTO evaluation VALUES (?, ?, ?, ?, ?)",
                    (
                        attempt_index,
                        candidate.candidate_id,
                        oracle_version,
                        canonical_json(result),
                        result.runtime_ns,
                    ),
                )
                self.connection.execute(
                    """INSERT INTO archive_transition VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        attempt_index,
                        canonical_json(decision.to_value()),
                        decision.decision_hash,
                        canonical_json(decision.coordinate.to_value()),
                        decision.outcome.value,
                        decision.role,
                        decision.inserted_candidate_id,
                        decision.replaced_candidate_id,
                        decision.evicted_candidate_id,
                    ),
                )
            self.connection.execute(
                """UPDATE budget_state SET
                    state_json = ?, proposal_attempts = ?, oracle_invocations = ?
                    WHERE singleton = 1""",
                (
                    canonical_json(budget.to_value()),
                    budget.proposal_attempts,
                    budget.oracle_invocations,
                ),
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
            if self._has_table("attempt_diagnostic"):
                self.connection.execute(
                    "INSERT INTO attempt_diagnostic VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        attempt_index,
                        attempt_cpu_ns,
                        oracle_cpu_ns,
                        attempt_elapsed_ns,
                        oracle_elapsed_ns,
                        budget.language_model_calls,
                        budget.language_model_tokens,
                    ),
                )

    def _has_table(self, table: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        return row is not None

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

    def evaluation_record(self, candidate_id: str) -> sqlite3.Row:
        row = cast(
            sqlite3.Row | None,
            self.connection.execute(
                "SELECT * FROM evaluation WHERE candidate_id = ?", (candidate_id,)
            ).fetchone(),
        )
        if row is None:
            raise PersistenceError(f"missing evaluation record: {candidate_id}")
        return row

    def candidate_records(self) -> tuple[sqlite3.Row, ...]:
        return tuple(self.connection.execute("SELECT * FROM candidate ORDER BY rowid").fetchall())

    def evaluation_records(self) -> tuple[sqlite3.Row, ...]:
        return tuple(self.connection.execute("SELECT * FROM evaluation ORDER BY rowid").fetchall())

    def phase3_attempt_records(self) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self.connection.execute(
                "SELECT * FROM proposal_attempt ORDER BY attempt_index"
            ).fetchall()
        )

    def phase3_transition_records(self) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self.connection.execute(
                "SELECT * FROM archive_transition ORDER BY attempt_index"
            ).fetchall()
        )

    def phase3_lineage_records(self) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self.connection.execute(
                "SELECT * FROM lineage_edge ORDER BY child_candidate_id, parent_order"
            ).fetchall()
        )

    def phase3_diagnostic_records(self) -> tuple[sqlite3.Row, ...]:
        """Return optional non-replay-stable timing data from hardened Phase 3 runs."""

        if not self._has_table("attempt_diagnostic"):
            return ()
        return tuple(
            self.connection.execute(
                "SELECT * FROM attempt_diagnostic ORDER BY attempt_index"
            ).fetchall()
        )

    def table_count(self, table: str) -> int:
        if table not in {
            "candidate",
            "evaluation",
            "event",
            "task",
            "proposal_attempt",
            "archive_transition",
            "lineage_edge",
        }:
            raise ValueError("unsupported table count")
        row = self.connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        if row is None:
            raise PersistenceError(f"cannot count table: {table}")
        return int(row["count"])
