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
    SplitLabel,
    Task,
)
from world_model_search.dsl.ast import AstLimits
from world_model_search.dsl.json_schema import DslCandidateDocument, ast_to_value
from world_model_search.dsl.versions import (
    CANDIDATE_SCHEMA_VERSION,
    PHASE3_MANIFEST_SCHEMA_VERSION,
    PREFIX_CODE_VERSION,
)
from world_model_search.errors import ConfigurationError, PersistenceError
from world_model_search.evaluation.phase2_analysis import write_phase2_analysis
from world_model_search.logging import structured_log
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.oracle.mock import MockOracle
from world_model_search.persistence.artifacts import write_content_artifact, write_json_exclusive
from world_model_search.persistence.database import RunDatabase
from world_model_search.persistence.manifest import (
    MANIFEST_SCHEMA_VERSION,
    PHASE0_MANIFEST_SCHEMA_VERSION,
    build_manifest,
    utc_now,
)
from world_model_search.proposer.enumerative import (
    EnumeratedProgram,
    EnumerationBounds,
    EnumerationResult,
    enumerate_programs,
)
from world_model_search.proposer.mock import MockProposer
from world_model_search.search.fixture import make_fixture_task
from world_model_search.serialization import (
    JsonObject,
    canonical_json,
    derive_seed,
    sha256_json,
    sha256_text,
)
from world_model_search.tasks import (
    HiddenTaskBundle,
    HiddenTaskStore,
    benchmark_root_for_config,
    load_public_task,
)

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
    if recorded_schema not in {
        PHASE0_MANIFEST_SCHEMA_VERSION,
        MANIFEST_SCHEMA_VERSION,
        PHASE3_MANIFEST_SCHEMA_VERSION,
    }:
        raise PersistenceError(
            f"unsupported run manifest schema {recorded_schema!r}; "
            f"this build requires schema {PHASE0_MANIFEST_SCHEMA_VERSION} or "
            f"{MANIFEST_SCHEMA_VERSION}, or {PHASE3_MANIFEST_SCHEMA_VERSION}"
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


def phase2_deterministic_event_payload(
    *, candidate: Candidate, result_payload: JsonObject, step_index: int, ast_bits: int
) -> JsonObject:
    if candidate.semantic_hash is None:
        raise ValueError("Phase 2 event requires an internal semantic hash")
    return {
        "schema_version": 2,
        "step_index": step_index,
        "task_id": candidate.task_id,
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "payload_hash": candidate.payload_hash,
            "parent_ids": list(candidate.parent_ids),
            "proposer_id": candidate.proposer_id,
            "operator_id": candidate.operator_id,
            "context_hash": candidate.context_hash,
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "semantic_hash": candidate.semantic_hash,
            "ast_bits": ast_bits,
            "coding_version": PREFIX_CODE_VERSION,
            "discovery_index": step_index,
        },
        "oracle_result": result_payload,
    }


def phase2_deterministic_results(
    events: tuple[SearchEvent, ...], *, analysis_manifest_hash: str
) -> JsonObject:
    payloads = [event.payload for event in events]
    exact_payloads: list[tuple[SearchEvent, JsonObject]] = []
    semantic_hashes: set[str] = set()
    for event, payload in zip(events, payloads, strict=True):
        oracle_result = payload.get("oracle_result")
        candidate = payload.get("candidate")
        if not isinstance(oracle_result, dict) or not isinstance(candidate, dict):
            raise PersistenceError("Phase 2 event payload is malformed")
        semantic = candidate.get("semantic_hash")
        if isinstance(semantic, str):
            semantic_hashes.add(semantic)
        if oracle_result.get("exact") is True:
            exact_payloads.append((event, payload))

    def ast_bits(payload: JsonObject) -> int:
        candidate = payload.get("candidate")
        if not isinstance(candidate, dict):
            raise PersistenceError("Phase 2 event has no AST bit length")
        value = candidate.get("ast_bits")
        if isinstance(value, bool) or not isinstance(value, int):
            raise PersistenceError("Phase 2 event has no AST bit length")
        return value

    summary: JsonObject = {
        "schema_version": 2,
        "status": "completed",
        "metrics": {
            "candidate_evaluations": len(events),
            "distinct_candidate_semantics": len(semantic_hashes),
            "exact_candidates": len(exact_payloads),
            "first_exact_logical_cost": exact_payloads[0][0].logical_cost
            if exact_payloads
            else None,
            "best_exact_ast_bits": (
                min(ast_bits(payload) for _, payload in exact_payloads) if exact_payloads else None
            ),
        },
        "event_payload_hashes": [event.payload_hash for event in events],
        "analysis_manifest_hash": analysis_manifest_hash,
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


@dataclass(frozen=True, slots=True)
class Phase2PreparedRun:
    task: Task
    hidden: HiddenTaskBundle
    task_store: HiddenTaskStore
    enumeration: EnumerationResult


def _prepare_phase2_run(
    *, repository_root: Path, config: AppConfig, purpose: str
) -> Phase2PreparedRun:
    if config.dsl is None or config.enumerator is None:
        raise ConfigurationError("Phase 2 configuration is missing DSL or enumerator bounds")
    benchmark_root = benchmark_root_for_config(repository_root, config)
    task = load_public_task(benchmark_root, config.run.task_id)
    if task.split != config.run.split:
        raise ConfigurationError("configured split does not match the public task artifact")
    allowed = frozenset({SplitLabel.TRAINING, SplitLabel.DEVELOPMENT})
    if task.split not in allowed:
        raise ConfigurationError("Phase 2 runs may use only training or development tasks")
    task_store = HiddenTaskStore(benchmark_root)
    hidden = task_store.load(task.task_id, allowed_splits=allowed, purpose=purpose)
    bounds = EnumerationBounds(
        max_bits=config.enumerator.max_bits,
        max_depth=config.enumerator.max_depth,
        max_nodes=config.enumerator.max_nodes,
        max_candidates=config.enumerator.max_candidates,
    )
    enumeration = enumerate_programs(bounds)
    if len(enumeration.programs) < config.run.max_steps:
        raise ConfigurationError(
            "run.max_steps exceeds the programs produced within enumerator bounds"
        )
    return Phase2PreparedRun(
        task=task,
        hidden=hidden,
        task_store=task_store,
        enumeration=enumeration,
    )


def _phase2_candidate(
    *, program: EnumeratedProgram, task: Task, context: ProposalContext
) -> tuple[Candidate, DslCandidateDocument]:
    document = DslCandidateDocument(ast=program.ast)
    payload_hash = sha256_text(document.to_json())
    identity = {
        "candidate_identity_schema": 2,
        "task_id": task.task_id,
        "canonical_ast": ast_to_value(program.ast),
        "parent_ids": [],
        "proposer_id": "enumerative",
        "operator_id": "cost-ordered-enumeration-v1",
        "context_hash": context.content_hash,
        "payload_hash": payload_hash,
        "coding_version": PREFIX_CODE_VERSION,
    }
    candidate = Candidate(
        candidate_id=sha256_json(identity),
        task_id=task.task_id,
        ast=program.ast,
        parent_ids=(),
        proposer_id="enumerative",
        operator_id="cost-ordered-enumeration-v1",
        context_hash=context.content_hash,
        payload_hash=payload_hash,
        semantic_hash=program.semantic_hash,
    )
    return candidate, document


class Phase2RunEngine:
    """Recorded enumerative evidence using the existing run/event lifecycle."""

    def __init__(
        self,
        *,
        repository_root: Path,
        run_directory: Path,
        config: AppConfig,
        prepared: Phase2PreparedRun,
    ) -> None:
        if config.dsl is None:
            raise ConfigurationError("Phase 2 configuration is missing DSL limits")
        self.repository_root = repository_root
        self.run_directory = run_directory
        self.config = config
        self.prepared = prepared
        self.limits = AstLimits(
            max_depth=config.dsl.max_depth,
            max_nodes=config.dsl.max_nodes,
            max_cases=config.dsl.max_cases,
        )
        self.oracle = ExactDslOracle(
            prepared.hidden,
            limits=self.limits,
            response_mode=config.oracle.response_mode,
        )

    def execute(self, *, interrupt_after: int | None = None) -> RunOutcome:
        if interrupt_after is not None and interrupt_after < 1:
            raise ConfigurationError("interrupt_after must be >= 1")
        with RunDatabase(self.run_directory / "run.sqlite3") as database:
            state = database.state()
            if state.status == "completed":
                return self._outcome(state.run_id, state.status, database.events())
            database.set_status("running", utc_now())
            try:
                for step in range(state.next_step, self.config.run.max_steps):
                    program = self.prepared.enumeration.programs[step]
                    context = ProposalContext(task=self.prepared.task.public_view())
                    candidate, document = _phase2_candidate(
                        program=program,
                        task=self.prepared.task,
                        context=context,
                    )
                    artifact_name = f"proposals/{candidate.candidate_id}.json"
                    artifact_hash = write_content_artifact(
                        self.run_directory / artifact_name, document.to_json()
                    )
                    if artifact_hash != candidate.payload_hash:
                        raise PersistenceError("proposal artifact hash does not match candidate")
                    evaluated = self.oracle.evaluate(program.ast)
                    if (
                        evaluated.canonical_ast != program.ast
                        or evaluated.semantic_hash != program.semantic_hash
                        or evaluated.result.ast_bits != program.ast_bits
                    ):
                        raise PersistenceError(
                            "enumerator and exact oracle candidate metadata diverged"
                        )
                    event = SearchEvent.create(
                        sequence=step,
                        event_type="candidate_evaluated",
                        logical_cost=step + 1,
                        payload=phase2_deterministic_event_payload(
                            candidate=candidate,
                            result_payload=evaluated.result.deterministic_payload(),
                            step_index=step,
                            ast_bits=program.ast_bits,
                        ),
                        audit_timestamp=utc_now(),
                    )
                    database.append_phase2_evaluation(
                        candidate=candidate,
                        source_ast=program.ast,
                        artifact_name=artifact_name,
                        result=evaluated.result,
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
                        exact=evaluated.result.exact,
                        runtime_ns=evaluated.result.runtime_ns,
                    )
                    if interrupt_after is not None and step + 1 >= interrupt_after:
                        database.set_status("interrupted", utc_now())
                        return self._outcome(state.run_id, "interrupted", database.events())
            except KeyboardInterrupt:
                database.set_status("interrupted", utc_now())
                raise
            analysis = write_phase2_analysis(
                run_directory=self.run_directory,
                enumeration=self.prepared.enumeration,
                task=self.prepared.task,
                hidden=self.prepared.hidden,
                accesses=self.prepared.task_store.accesses,
            )
            analysis_hash = sha256_text(analysis.manifest.read_text(encoding="utf-8").rstrip("\n"))
            events = database.events()
            results = phase2_deterministic_results(events, analysis_manifest_hash=analysis_hash)
            write_content_artifact(self.run_directory / "results.json", canonical_json(results))
            database.set_status("completed", utc_now())
            structured_log(
                LOGGER,
                logging.INFO,
                "Phase 2 run completed",
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
    if config.schema_version == 3:
        from world_model_search.search.phase3 import start_phase3_run

        return start_phase3_run(
            repository_root=repository_root,
            config=config,
            config_source=config_source,
            run_id=selected_run_id,
            interrupt_after=interrupt_after,
        )  # type: ignore[return-value]
    if interrupt_after is not None and interrupt_after < 1:
        raise ConfigurationError("interrupt_after must be >= 1")
    run_directory = repository_root / config.run.root / selected_run_id
    if run_directory.exists():
        raise PersistenceError(f"run already exists: {selected_run_id}")
    prepared = (
        _prepare_phase2_run(
            repository_root=repository_root,
            config=config,
            purpose="phase2-recorded-enumerative-run",
        )
        if config.schema_version == 2
        else None
    )
    task = prepared.task if prepared is not None else make_fixture_task(config)
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
        if prepared is None:
            database.initialize(selected_run_id, task, utc_now())
        else:
            database.initialize_phase2(selected_run_id, task, utc_now())
    if prepared is not None:
        return Phase2RunEngine(
            repository_root=repository_root,
            run_directory=run_directory,
            config=config,
            prepared=prepared,
        ).execute(interrupt_after=interrupt_after)
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
    if config.schema_version == 3:
        from world_model_search.search.phase3 import resume_phase3_run

        return resume_phase3_run(
            repository_root=repository_root,
            run_directory=run_directory,
            run_id=run_id,
            config=config,
            manifest=manifest,
            interrupt_after=interrupt_after,
        )  # type: ignore[return-value]
    if config.schema_version == 2:
        if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
            raise PersistenceError("Phase 2 configuration requires run manifest schema 3")
        prepared = _prepare_phase2_run(
            repository_root=repository_root,
            config=config,
            purpose="phase2-recorded-enumerative-run",
        )
        return Phase2RunEngine(
            repository_root=repository_root,
            run_directory=run_directory,
            config=config,
            prepared=prepared,
        ).execute(interrupt_after=interrupt_after)
    if manifest.get("manifest_schema_version") != PHASE0_MANIFEST_SCHEMA_VERSION:
        raise PersistenceError("Phase 0 configuration requires run manifest schema 2")
    task = make_fixture_task(config)
    return RunEngine(
        repository_root=repository_root,
        run_directory=run_directory,
        config=config,
        task=task,
    ).execute(interrupt_after=interrupt_after)
