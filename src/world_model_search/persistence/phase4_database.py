"""SQLite schema 5 for crash-safe Phase 4 requests, batches, and evaluations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from world_model_search.domain.types import Candidate, OracleResult, SearchEvent, Task
from world_model_search.dsl.ast import BitExpr
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.json_schema import ast_canonical_json
from world_model_search.errors import PersistenceError
from world_model_search.phase4_versions import (
    PHASE4_CANDIDATE_IDENTITY_VERSION,
    PHASE4_DATABASE_SCHEMA_VERSION,
)
from world_model_search.search.archive import ArchiveDecision
from world_model_search.search.phase4_types import Phase4BudgetState, RequestState
from world_model_search.serialization import JsonObject, canonical_json


@dataclass(frozen=True, slots=True)
class Phase4RunState:
    run_id: str
    status: str
    next_evaluation: int
    next_request: int
    next_event: int


class Phase4Database:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        target = f"file:{path}?mode=ro" if read_only else str(path)
        self.connection = sqlite3.connect(target, uri=read_only)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")

    def __enter__(self) -> Phase4Database:
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    def initialize(
        self,
        *,
        run_id: str,
        task: Task,
        timestamp: str,
        budget: Phase4BudgetState,
    ) -> None:
        schema = """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE run_state (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            run_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK(
                status IN (
                    'running','interrupted','completed','cost-cap-exhausted',
                    'usage-uncertain','failed'
                )
            ),
            next_evaluation INTEGER NOT NULL,
            next_request INTEGER NOT NULL,
            next_event INTEGER NOT NULL,
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
            parent_ids_json TEXT NOT NULL,
            proposer_id TEXT NOT NULL,
            operator_id TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            source_ast_json TEXT NOT NULL,
            canonical_ast_json TEXT NOT NULL,
            semantic_hash TEXT NOT NULL,
            ast_bits INTEGER NOT NULL,
            identity_schema TEXT NOT NULL,
            first_evaluation_index INTEGER NOT NULL
        );
        CREATE TABLE model_request (
            request_index INTEGER PRIMARY KEY,
            logical_call_index INTEGER NOT NULL,
            retry_index INTEGER NOT NULL,
            condition_id TEXT NOT NULL,
            role TEXT NOT NULL,
            selected_branch_id TEXT,
            ordered_parent_ids_json TEXT NOT NULL,
            scheduler_json TEXT NOT NULL,
            prompt_artifact TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            request_artifact TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            backend_id TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            resolved_model TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            service_tier TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            batch_size INTEGER NOT NULL,
            state TEXT NOT NULL,
            cache_namespace TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            cache_hit INTEGER NOT NULL,
            reservation_id TEXT,
            reserved_nano_usd INTEGER NOT NULL,
            provider_request_id TEXT,
            response_artifact TEXT,
            response_hash TEXT,
            error_json TEXT,
            usage_json TEXT,
            actual_nano_usd INTEGER NOT NULL DEFAULT 0,
            released_nano_usd INTEGER NOT NULL DEFAULT 0,
            uncertain_nano_usd INTEGER NOT NULL DEFAULT 0,
            price_entry_json TEXT NOT NULL,
            next_item_ordinal INTEGER NOT NULL DEFAULT 0,
            item_count INTEGER
        );
        CREATE TABLE proposal_item (
            request_index INTEGER NOT NULL REFERENCES model_request(request_index),
            ordinal INTEGER NOT NULL,
            submitted_document_json TEXT,
            outcome TEXT NOT NULL,
            rejection_reason TEXT,
            candidate_id TEXT REFERENCES candidate(candidate_id),
            canonical_duplicate INTEGER NOT NULL,
            semantic_duplicate INTEGER NOT NULL,
            PRIMARY KEY(request_index, ordinal)
        );
        CREATE TABLE evaluation (
            evaluation_index INTEGER PRIMARY KEY,
            candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
            request_index INTEGER REFERENCES model_request(request_index),
            item_ordinal INTEGER,
            initialization_index INTEGER,
            oracle_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            runtime_ns INTEGER NOT NULL
        );
        CREATE TABLE archive_transition (
            evaluation_index INTEGER PRIMARY KEY REFERENCES evaluation(evaluation_index),
            decision_json TEXT NOT NULL,
            decision_hash TEXT NOT NULL,
            outcome TEXT NOT NULL,
            role TEXT,
            coordinate_json TEXT NOT NULL
        );
        CREATE TABLE lineage_edge (
            child_candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
            parent_candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
            parent_order INTEGER NOT NULL,
            PRIMARY KEY(child_candidate_id,parent_order)
        );
        CREATE TABLE budget_state (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            state_json TEXT NOT NULL
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
                "INSERT INTO metadata VALUES ('database_schema_version',?)",
                (str(PHASE4_DATABASE_SCHEMA_VERSION),),
            )
            self.connection.execute(
                "INSERT INTO run_state VALUES (1,?,'running',0,0,0,?,?)",
                (run_id, timestamp, timestamp),
            )
            self.connection.execute(
                "INSERT INTO task VALUES (?,?,?,?,?,?,?,?)",
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
                "INSERT INTO budget_state VALUES (1,?)", (canonical_json(budget.to_value()),)
            )

    def state(self) -> Phase4RunState:
        row = self.connection.execute("SELECT * FROM run_state WHERE singleton=1").fetchone()
        if row is None:
            raise PersistenceError("Phase 4 run has no state")
        return Phase4RunState(
            row["run_id"],
            row["status"],
            row["next_evaluation"],
            row["next_request"],
            row["next_event"],
        )

    def set_status(self, status: str, timestamp: str) -> None:
        if status not in {
            "running",
            "interrupted",
            "completed",
            "cost-cap-exhausted",
            "usage-uncertain",
            "failed",
        }:
            raise ValueError("invalid Phase 4 run status")
        with self.connection:
            self.connection.execute(
                "UPDATE run_state SET status=?,updated_at=? WHERE singleton=1",
                (status, timestamp),
            )

    def budget(self) -> Phase4BudgetState:
        import json

        row: sqlite3.Row | None = self.connection.execute(
            "SELECT state_json FROM budget_state WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise PersistenceError("Phase 4 run has no budget")
        try:
            return Phase4BudgetState.from_value(json.loads(row["state_json"]))
        except (ValueError, TypeError, KeyError) as exc:
            raise PersistenceError("Phase 4 budget record is corrupt") from exc

    def _write_budget(self, budget: Phase4BudgetState) -> None:
        budget.validate()
        self.connection.execute(
            "UPDATE budget_state SET state_json=? WHERE singleton=1",
            (canonical_json(budget.to_value()),),
        )

    def _insert_candidate(
        self,
        *,
        candidate: Candidate,
        source_ast: BitExpr,
        evaluation_index: int,
    ) -> None:
        if not isinstance(candidate.ast, BitExpr) or candidate.semantic_hash is None:
            raise TypeError("Phase 4 candidate must be a typed semantic AST")
        existing = self.connection.execute(
            "SELECT * FROM candidate WHERE candidate_id=?", (candidate.candidate_id,)
        ).fetchone()
        if existing is not None:
            if existing["payload_hash"] != candidate.payload_hash or existing[
                "parent_ids_json"
            ] != canonical_json(candidate.parent_ids):
                raise PersistenceError("Phase 4 candidate identity collision")
            return
        for parent_id in candidate.parent_ids:
            parent = self.connection.execute(
                "SELECT first_evaluation_index FROM candidate WHERE candidate_id=?", (parent_id,)
            ).fetchone()
            if parent is None or parent["first_evaluation_index"] >= evaluation_index:
                raise PersistenceError("Phase 4 parent must be an earlier recorded candidate")
        self.connection.execute(
            "INSERT INTO candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                candidate.candidate_id,
                candidate.task_id,
                canonical_json(candidate.parent_ids),
                candidate.proposer_id,
                candidate.operator_id,
                candidate.context_hash,
                candidate.payload_hash,
                ast_canonical_json(source_ast),
                ast_canonical_json(candidate.ast),
                candidate.semantic_hash,
                encoded_length(candidate.ast),
                PHASE4_CANDIDATE_IDENTITY_VERSION,
                evaluation_index,
            ),
        )
        self.connection.executemany(
            "INSERT INTO lineage_edge VALUES (?,?,?)",
            tuple(
                (candidate.candidate_id, parent_id, order)
                for order, parent_id in enumerate(candidate.parent_ids)
            ),
        )

    def append_evaluation(
        self,
        *,
        candidate: Candidate,
        source_ast: BitExpr,
        result: OracleResult,
        decision: ArchiveDecision,
        oracle_version: str,
        event: SearchEvent,
        budget: Phase4BudgetState,
        request_index: int | None = None,
        item_ordinal: int | None = None,
        initialization_index: int | None = None,
        submitted_document: JsonObject | None = None,
        canonical_duplicate: bool = False,
        semantic_duplicate: bool = False,
    ) -> None:
        state = self.state()
        evaluation_index = state.next_evaluation
        if event.sequence != state.next_event:
            raise ValueError("Phase 4 event sequence is inconsistent")
        with self.connection:
            self._insert_candidate(
                candidate=candidate, source_ast=source_ast, evaluation_index=evaluation_index
            )
            if request_index is not None and item_ordinal is not None:
                self.connection.execute(
                    "INSERT INTO proposal_item VALUES (?,?,?,?,?,?,?,?)",
                    (
                        request_index,
                        item_ordinal,
                        canonical_json(submitted_document)
                        if submitted_document is not None
                        else None,
                        "accepted",
                        None,
                        candidate.candidate_id,
                        int(canonical_duplicate),
                        int(semantic_duplicate),
                    ),
                )
            self.connection.execute(
                "INSERT INTO evaluation VALUES (?,?,?,?,?,?,?,?)",
                (
                    evaluation_index,
                    candidate.candidate_id,
                    request_index,
                    item_ordinal,
                    initialization_index,
                    oracle_version,
                    canonical_json(result),
                    result.runtime_ns,
                ),
            )
            self.connection.execute(
                "INSERT INTO archive_transition VALUES (?,?,?,?,?,?)",
                (
                    evaluation_index,
                    canonical_json(decision.to_value()),
                    decision.decision_hash,
                    decision.outcome.value,
                    decision.role,
                    canonical_json(decision.coordinate.to_value()),
                ),
            )
            self._write_budget(budget)
            self._insert_event(event)
            if request_index is not None and item_ordinal is not None:
                self.connection.execute(
                    "UPDATE model_request SET next_item_ordinal=? WHERE request_index=?",
                    (item_ordinal + 1, request_index),
                )
            self.connection.execute(
                """UPDATE run_state SET next_evaluation=?,next_event=?,updated_at=?
                   WHERE singleton=1""",
                (evaluation_index + 1, event.sequence + 1, event.audit_timestamp),
            )

    def append_rejected_item(
        self,
        *,
        request_index: int,
        ordinal: int,
        submitted_document: JsonObject | None,
        rejection_reason: str,
        event: SearchEvent,
        budget: Phase4BudgetState,
    ) -> None:
        state = self.state()
        if event.sequence != state.next_event:
            raise ValueError("Phase 4 rejected-item event sequence is inconsistent")
        with self.connection:
            self.connection.execute(
                "INSERT INTO proposal_item VALUES (?,?,?,?,?,?,0,0)",
                (
                    request_index,
                    ordinal,
                    canonical_json(submitted_document) if submitted_document is not None else None,
                    "rejected",
                    rejection_reason,
                    None,
                ),
            )
            self._write_budget(budget)
            self._insert_event(event)
            self.connection.execute(
                "UPDATE model_request SET next_item_ordinal=? WHERE request_index=?",
                (ordinal + 1, request_index),
            )
            self.connection.execute(
                "UPDATE run_state SET next_event=?,updated_at=? WHERE singleton=1",
                (event.sequence + 1, event.audit_timestamp),
            )

    def prepare_request(
        self, record: JsonObject, *, budget: Phase4BudgetState, timestamp: str
    ) -> int:
        state = self.state()
        index = state.next_request
        expected = {
            "logical_call_index",
            "retry_index",
            "condition_id",
            "role",
            "selected_branch_id",
            "ordered_parent_ids",
            "scheduler",
            "prompt_artifact",
            "prompt_hash",
            "request_artifact",
            "request_hash",
            "backend_id",
            "provider_id",
            "resolved_model",
            "endpoint",
            "service_tier",
            "settings",
            "batch_size",
            "cache_namespace",
            "cache_key",
            "cache_hit",
            "reservation_id",
            "reserved_nano_usd",
            "price_entry",
        }
        if set(record) != expected:
            raise ValueError("Phase 4 pending request record has missing or unknown fields")
        with self.connection:
            self.connection.execute(
                """INSERT INTO model_request (
                    request_index,logical_call_index,retry_index,condition_id,role,
                    selected_branch_id,ordered_parent_ids_json,scheduler_json,prompt_artifact,
                    prompt_hash,request_artifact,request_hash,backend_id,provider_id,
                    resolved_model,endpoint,service_tier,settings_json,batch_size,state,
                    cache_namespace,cache_key,cache_hit,reservation_id,reserved_nano_usd,
                    price_entry_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?,?,?,?,?,?)""",
                (
                    index,
                    record["logical_call_index"],
                    record["retry_index"],
                    record["condition_id"],
                    record["role"],
                    record["selected_branch_id"],
                    canonical_json(record["ordered_parent_ids"]),
                    canonical_json(record["scheduler"]),
                    record["prompt_artifact"],
                    record["prompt_hash"],
                    record["request_artifact"],
                    record["request_hash"],
                    record["backend_id"],
                    record["provider_id"],
                    record["resolved_model"],
                    record["endpoint"],
                    record["service_tier"],
                    canonical_json(record["settings"]),
                    record["batch_size"],
                    record["cache_namespace"],
                    record["cache_key"],
                    int(bool(record["cache_hit"])),
                    record["reservation_id"],
                    record["reserved_nano_usd"],
                    canonical_json(record["price_entry"]),
                ),
            )
            self._write_budget(budget)
            self.connection.execute(
                "UPDATE run_state SET next_request=?,updated_at=? WHERE singleton=1",
                (index + 1, timestamp),
            )
        return index

    def mark_dispatched(self, request_index: int) -> None:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE model_request SET state='dispatched' "
                "WHERE request_index=? AND state='pending'",
                (request_index,),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("model request is not pending")

    def finalize_request(
        self,
        *,
        request_index: int,
        state: RequestState,
        provider_request_id: str | None,
        response_artifact: str | None,
        response_hash: str | None,
        error: JsonObject | None,
        usage: JsonObject | None,
        actual_nano_usd: int,
        released_nano_usd: int,
        uncertain_nano_usd: int,
        item_count: int | None,
        budget: Phase4BudgetState,
    ) -> None:
        if state not in {
            RequestState.RESPONDED,
            RequestState.SCHEMA_FAILURE,
            RequestState.FAILED,
            RequestState.USAGE_UNCERTAIN,
        }:
            raise ValueError("invalid finalized request state")
        with self.connection:
            cursor = self.connection.execute(
                """UPDATE model_request SET state=?,provider_request_id=?,response_artifact=?,
                   response_hash=?,error_json=?,usage_json=?,actual_nano_usd=?,
                   released_nano_usd=?,uncertain_nano_usd=?,item_count=?
                   WHERE request_index=? AND state IN ('pending','dispatched')""",
                (
                    state.value,
                    provider_request_id,
                    response_artifact,
                    response_hash,
                    canonical_json(error) if error is not None else None,
                    canonical_json(usage) if usage is not None else None,
                    actual_nano_usd,
                    released_nano_usd,
                    uncertain_nano_usd,
                    item_count,
                    request_index,
                ),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("model request cannot be finalized from its current state")
            self._write_budget(budget)

    def complete_request(self, request_index: int) -> None:
        with self.connection:
            row = self.connection.execute(
                "SELECT next_item_ordinal,item_count,state "
                "FROM model_request WHERE request_index=?",
                (request_index,),
            ).fetchone()
            if (
                row is None
                or row["state"] != "responded"
                or row["next_item_ordinal"] != row["item_count"]
            ):
                raise PersistenceError("model response items are not fully committed")
            self.connection.execute(
                "UPDATE model_request SET state='completed' WHERE request_index=?",
                (request_index,),
            )

    def _insert_event(self, event: SearchEvent) -> None:
        self.connection.execute(
            """INSERT INTO event (
                sequence,run_id,event_type,logical_cost,payload_json,payload_hash,audit_timestamp
            ) SELECT ?,run_id,?,?,?,?,? FROM run_state WHERE singleton=1""",
            (
                event.sequence,
                event.event_type,
                event.logical_cost,
                event.payload_json,
                event.payload_hash,
                event.audit_timestamp,
            ),
        )

    def events(self) -> tuple[SearchEvent, ...]:
        rows = self.connection.execute("SELECT * FROM event ORDER BY sequence").fetchall()
        return tuple(
            SearchEvent(
                row["sequence"],
                row["event_type"],
                row["logical_cost"],
                row["payload_json"],
                row["payload_hash"],
                row["audit_timestamp"],
            )
            for row in rows
        )

    def requests(self) -> tuple[sqlite3.Row, ...]:
        return tuple(self.connection.execute("SELECT * FROM model_request ORDER BY request_index"))

    def items(self) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self.connection.execute("SELECT * FROM proposal_item ORDER BY request_index,ordinal")
        )

    def candidates(self) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self.connection.execute("SELECT * FROM candidate ORDER BY first_evaluation_index")
        )

    def evaluations(self) -> tuple[sqlite3.Row, ...]:
        return tuple(self.connection.execute("SELECT * FROM evaluation ORDER BY evaluation_index"))

    def transitions(self) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self.connection.execute("SELECT * FROM archive_transition ORDER BY evaluation_index")
        )

    def lineage(self) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self.connection.execute(
                "SELECT * FROM lineage_edge ORDER BY child_candidate_id,parent_order"
            )
        )

    def request(self, request_index: int) -> sqlite3.Row:
        row: sqlite3.Row | None = self.connection.execute(
            "SELECT * FROM model_request WHERE request_index=?", (request_index,)
        ).fetchone()
        if row is None:
            raise PersistenceError("missing Phase 4 model request")
        return row

    def candidate_result(self, candidate_id: str) -> sqlite3.Row:
        row: sqlite3.Row | None = self.connection.execute(
            """SELECT evaluation.* FROM evaluation
               WHERE candidate_id=? ORDER BY evaluation_index LIMIT 1""",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise PersistenceError("missing parent evaluation")
        return row

    def table_count(self, table: str) -> int:
        if table not in {
            "candidate",
            "evaluation",
            "event",
            "model_request",
            "proposal_item",
            "archive_transition",
            "lineage_edge",
        }:
            raise ValueError("unsupported Phase 4 table count")
        row = self.connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        return int(row["count"] if row is not None else 0)
