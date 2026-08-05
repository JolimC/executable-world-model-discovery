"""Deterministic Phase 0 run lifecycle: start, interrupt, and resume."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from world_model_search.config import AppConfig, config_from_mapping
from world_model_search.domain.types import (
    Candidate,
    CandidatePayload,
    ProposalBudget,
    ProposalContext,
    SearchEvent,
    Task,
)
from world_model_search.errors import ConfigurationError, PersistenceError
from world_model_search.logging import structured_log
from world_model_search.oracle.mock import MockOracle
from world_model_search.persistence.artifacts import write_content_artifact, write_json_exclusive
from world_model_search.persistence.database import RunDatabase
from world_model_search.persistence.manifest import MANIFEST_SCHEMA_VERSION, build_manifest, utc_now
from world_model_search.proposer.mock import MockProposer
from world_model_search.search.fixture import make_fixture_task
from world_model_search.serialization import JsonObject, derive_seed, sha256_json, sha256_text

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    run_id: str
    status: str
    completed_steps: int
    event_payload_hashes: tuple[str, ...]
    run_directory: Path


def validate_run_id(run_id: str) -> str:
    """Reject traversal and ambiguous filesystem identifiers before any write."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ConfigurationError(
            "run id must be 1-80 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return run_id


def generate_run_id(config: AppConfig) -> str:
    """Generate an audit identifier; it never enters deterministic event payloads."""

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run-{timestamp}-s{config.run.seed}-p{os.getpid()}"


def _manifest_config(manifest: object) -> AppConfig:
    if not isinstance(manifest, dict):
        raise PersistenceError("run manifest must contain a JSON object")
    raw = manifest.get("resolved_configuration")
    try:
        return config_from_mapping(raw)
    except ConfigurationError as exc:
        raise PersistenceError(f"recorded configuration is invalid: {exc}") from exc


def load_manifest(run_directory: Path) -> JsonObject:
    path = run_directory / "manifest.json"
    if not path.is_file():
        raise PersistenceError(f"run manifest does not exist: {path}")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceError(f"cannot read run manifest: {exc}") from exc
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise PersistenceError("run manifest must contain a JSON object")
    recorded_schema = raw.get("manifest_schema_version")
    if recorded_schema != MANIFEST_SCHEMA_VERSION:
        raise PersistenceError(
            f"unsupported run manifest schema {recorded_schema!r}; "
            f"this build requires schema {MANIFEST_SCHEMA_VERSION}"
        )
    return raw


def _candidate(
    *,
    payload: CandidatePayload,
    task: Task,
    context: ProposalContext,
    proposer_id: str,
) -> Candidate:
    payload_hash = sha256_text(payload.to_json())
    identity = {
        "task_id": task.task_id,
        "ast": payload.ast,
        "parent_ids": [],
        "proposer_id": proposer_id,
        "operator_id": "mock-generate-v1",
        "context_hash": context.content_hash,
        "payload_hash": payload_hash,
    }
    candidate_id = sha256_json(identity)
    return Candidate(
        candidate_id=candidate_id,
        task_id=task.task_id,
        ast=payload.ast,
        parent_ids=(),
        proposer_id=proposer_id,
        operator_id="mock-generate-v1",
        context_hash=context.content_hash,
        payload_hash=payload_hash,
    )


def deterministic_event_payload(
    *, candidate: Candidate, result_payload: JsonObject, step_index: int
) -> JsonObject:
    """Define the complete Phase 0 deterministic event hashing boundary."""

    return {
        "schema_version": 1,
        "step_index": step_index,
        "task_id": candidate.task_id,
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "payload_hash": candidate.payload_hash,
            "parent_ids": list(candidate.parent_ids),
            "proposer_id": candidate.proposer_id,
            "operator_id": candidate.operator_id,
            "context_hash": candidate.context_hash,
        },
        "oracle_result": result_payload,
    }


def deterministic_results(events: tuple[SearchEvent, ...]) -> JsonObject:
    payloads = [event.payload for event in events]

    def is_exact(payload: JsonObject) -> bool:
        oracle_result = payload.get("oracle_result")
        return isinstance(oracle_result, dict) and oracle_result.get("exact") is True

    exact_count = sum(1 for payload in payloads if is_exact(payload))
    summary: JsonObject = {
        "schema_version": 1,
        "status": "completed",
        "metrics": {
            "candidate_evaluations": len(events),
            "exact_candidates": exact_count,
            "first_exact_logical_cost": next(
                (
                    event.logical_cost
                    for event, payload in zip(events, payloads, strict=True)
                    if is_exact(payload)
                ),
                None,
            ),
        },
        "event_payload_hashes": [event.payload_hash for event in events],
    }
    summary["deterministic_summary_hash"] = sha256_json(summary)
    return summary


class RunEngine:
    def __init__(
        self,
        *,
        repository_root: Path,
        run_directory: Path,
        config: AppConfig,
        task: Task,
    ) -> None:
        self.repository_root = repository_root
        self.run_directory = run_directory
        self.config = config
        self.task = task
        self.proposer = MockProposer()
        self.oracle = MockOracle(exact_index=config.run.max_steps - 1)

    def execute(self, *, interrupt_after: int | None = None) -> RunOutcome:
        if interrupt_after is not None and interrupt_after < 1:
            raise ConfigurationError("interrupt_after must be >= 1")
        database_path = self.run_directory / "run.sqlite3"
        with RunDatabase(database_path) as database:
            state = database.state()
            if state.status == "completed":
                events = database.events()
                return self._outcome(state.run_id, state.status, events)
            database.set_status("running", utc_now())
            try:
                for step in range(state.next_step, self.config.run.max_steps):
                    context = ProposalContext(task=self.task.public_view())
                    budget = ProposalBudget(
                        max_candidates=1,
                        start_index=step,
                        proposer_seed=derive_seed(self.config.run.seed, "proposer"),
                    )
                    proposals = self.proposer.propose(context, budget)
                    if len(proposals) != 1:
                        raise RuntimeError(
                            "Phase 0 mock proposer must return exactly one candidate"
                        )
                    payload = proposals[0]
                    candidate = _candidate(
                        payload=payload,
                        task=self.task,
                        context=context,
                        proposer_id=self.proposer.proposer_id,
                    )
                    artifact_name = f"proposals/{candidate.candidate_id}.json"
                    artifact_path = self.run_directory / artifact_name
                    artifact_hash = write_content_artifact(artifact_path, payload.to_json())
                    if artifact_hash != candidate.payload_hash:
                        raise PersistenceError("proposal artifact hash does not match candidate")
                    result = self.oracle.evaluate(candidate)
                    event = SearchEvent.create(
                        sequence=step,
                        event_type="candidate_evaluated",
                        logical_cost=step + 1,
                        payload=deterministic_event_payload(
                            candidate=candidate,
                            result_payload=result.deterministic_payload(),
                            step_index=step,
                        ),
                        audit_timestamp=utc_now(),
                    )
                    database.append_evaluation(
                        candidate=candidate,
                        artifact_name=artifact_name,
                        result=result,
                        oracle_version=self.config.oracle.oracle_id,
                        event=event,
                        next_step=step + 1,
                    )
                    structured_log(
                        LOGGER,
                        logging.INFO,
                        "candidate evaluated",
                        run_id=state.run_id,
                        step=step,
                        payload_hash=event.payload_hash,
                        exact=result.exact,
                        runtime_ns=result.runtime_ns,
                    )
                    if interrupt_after is not None and step + 1 >= interrupt_after:
                        database.set_status("interrupted", utc_now())
                        events = database.events()
                        structured_log(
                            LOGGER,
                            logging.WARNING,
                            "run deliberately interrupted",
                            run_id=state.run_id,
                            completed_steps=len(events),
                        )
                        return self._outcome(state.run_id, "interrupted", events)
            except KeyboardInterrupt:
                database.set_status("interrupted", utc_now())
                structured_log(LOGGER, logging.WARNING, "run interrupted", run_id=state.run_id)
                raise
            database.set_status("completed", utc_now())
            events = database.events()
            write_json_exclusive(self.run_directory / "results.json", deterministic_results(events))
            structured_log(
                LOGGER,
                logging.INFO,
                "run completed",
                run_id=state.run_id,
                completed_steps=len(events),
            )
            return self._outcome(state.run_id, "completed", events)

    def _outcome(self, run_id: str, status: str, events: tuple[SearchEvent, ...]) -> RunOutcome:
        return RunOutcome(
            run_id=run_id,
            status=status,
            completed_steps=len(events),
            event_payload_hashes=tuple(event.payload_hash for event in events),
            run_directory=self.run_directory,
        )


def start_run(
    *,
    repository_root: Path,
    config: AppConfig,
    config_source: str,
    run_id: str | None = None,
    interrupt_after: int | None = None,
) -> RunOutcome:
    """Create a fully validated run and execute it."""

    selected_run_id = validate_run_id(run_id or generate_run_id(config))
    if interrupt_after is not None and interrupt_after < 1:
        raise ConfigurationError("interrupt_after must be >= 1")
    run_directory = repository_root / config.run.root / selected_run_id
    if run_directory.exists():
        raise PersistenceError(f"run already exists: {selected_run_id}")
    task = make_fixture_task(config)
    manifest = build_manifest(
        repository_root=repository_root,
        run_id=selected_run_id,
        config=config,
        config_source=config_source,
        task=task,
    )

    # No persistent operation occurs above this line. Configuration and CLI values are
    # fully validated before the run root or any database/artifact is created.
    run_directory.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(run_directory / "manifest.json", manifest)
    with RunDatabase(run_directory / "run.sqlite3") as database:
        database.initialize(selected_run_id, task, utc_now())
    return RunEngine(
        repository_root=repository_root,
        run_directory=run_directory,
        config=config,
        task=task,
    ).execute(interrupt_after=interrupt_after)


def resume_run(
    *,
    repository_root: Path,
    runs_root: Path,
    run_id: str,
    interrupt_after: int | None = None,
) -> RunOutcome:
    validate_run_id(run_id)
    if runs_root.is_absolute() or ".." in runs_root.parts:
        raise ConfigurationError("runs root must be repository-relative without '..'")
    run_directory = repository_root / runs_root / run_id
    manifest = load_manifest(run_directory)
    config = _manifest_config(manifest)
    if config.run.root != runs_root:
        raise PersistenceError("recorded run root does not match --runs-root")
    task = make_fixture_task(config)
    return RunEngine(
        repository_root=repository_root,
        run_directory=run_directory,
        config=config,
        task=task,
    ).execute(interrupt_after=interrupt_after)
