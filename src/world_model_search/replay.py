"""Deterministic replay from recorded proposer artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from world_model_search.config import AppConfig, config_from_mapping
from world_model_search.domain.types import (
    Candidate,
    CandidatePayload,
    CandidateSummary,
    OracleResult,
    ProposalContext,
    SearchEvent,
    SplitLabel,
)
from world_model_search.dsl.ast import AstLimits
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.interpreter import semantic_hash
from world_model_search.dsl.json_schema import (
    DslCandidateDocument,
    ast_canonical_json,
    ast_to_value,
)
from world_model_search.dsl.versions import PREFIX_CODE_VERSION
from world_model_search.errors import ConfigurationError, ReplayError
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.oracle.mock import MockOracle
from world_model_search.persistence.artifacts import read_text_artifact
from world_model_search.persistence.database import RunDatabase
from world_model_search.search.fixture import make_fixture_task
from world_model_search.search.loop import (
    deterministic_event_payload,
    deterministic_results,
    load_manifest,
    phase2_deterministic_event_payload,
    phase2_deterministic_results,
    validate_run_id,
)
from world_model_search.serialization import (
    JsonObject,
    canonical_json,
    parse_json_object,
    sha256_json,
    sha256_text,
)
from world_model_search.tasks import HiddenTaskStore, benchmark_root_for_config, load_public_task


@dataclass(frozen=True, slots=True)
class ReplayReport:
    run_id: str
    event_count: int
    event_payload_hashes: tuple[str, ...]
    deterministic_summary_hash: str
    proposer_invocations: int = 0


def _config_from_manifest(manifest: JsonObject) -> AppConfig:
    raw = manifest.get("resolved_configuration")
    try:
        return config_from_mapping(raw)
    except ConfigurationError as exc:
        raise ReplayError(f"manifest configuration failed validation: {exc}") from exc


def _string_list(raw: object, location: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ReplayError(f"{location} must be a list of strings")
    return tuple(raw)


def replay_run(*, repository_root: Path, runs_root: Path, run_id: str) -> ReplayReport:
    """Re-evaluate recorded payloads; proposal generation is never called."""

    validate_run_id(run_id)
    if runs_root.is_absolute() or ".." in runs_root.parts:
        raise ConfigurationError("runs root must be repository-relative without '..'")
    run_directory = repository_root / runs_root / run_id
    manifest = load_manifest(run_directory)
    config = _config_from_manifest(manifest)
    if config.run.root != runs_root:
        raise ReplayError("manifest run root does not match --runs-root")
    if config.schema_version == 2:
        return _replay_phase2(
            repository_root=repository_root,
            run_directory=run_directory,
            run_id=run_id,
            config=config,
        )
    if config.schema_version == 3:
        return _replay_phase3(
            repository_root=repository_root,
            run_directory=run_directory,
            run_id=run_id,
            config=config,
            manifest=manifest,
        )
    task = make_fixture_task(config)
    context = ProposalContext(task=task.public_view())
    oracle = MockOracle(exact_index=config.run.max_steps - 1)

    with RunDatabase(run_directory / "run.sqlite3", read_only=True) as database:
        state = database.state()
        if state.status != "completed":
            raise ReplayError("only completed runs can be replayed")
        events = database.events()
        for expected_sequence, recorded_event in enumerate(events):
            if recorded_event.sequence != expected_sequence:
                raise ReplayError("event sequence is not contiguous")
            if sha256_text(recorded_event.payload_json) != recorded_event.payload_hash:
                raise ReplayError(f"event {expected_sequence} payload hash mismatch")
            if recorded_event.event_type != "candidate_evaluated":
                raise ReplayError(f"unsupported recorded event type: {recorded_event.event_type}")
            try:
                event_payload = recorded_event.payload
            except (ValueError, json.JSONDecodeError) as exc:
                raise ReplayError(f"event {expected_sequence} payload is invalid JSON") from exc
            candidate_raw = event_payload.get("candidate")
            if not isinstance(candidate_raw, dict):
                raise ReplayError(f"event {expected_sequence} has no candidate object")
            candidate_id = candidate_raw.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise ReplayError(f"event {expected_sequence} has no candidate id")
            row = database.candidate_record(candidate_id)
            artifact_name: str = row["artifact_name"]
            artifact_relative = Path(artifact_name)
            if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
                raise ReplayError("candidate artifact path escapes the run directory")
            proposal_json = read_text_artifact(run_directory / artifact_relative)
            try:
                proposal = CandidatePayload.from_json(proposal_json)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ReplayError(f"candidate {candidate_id} artifact is invalid") from exc
            payload_hash = sha256_text(proposal_json)
            if payload_hash != row["payload_hash"] or payload_hash != candidate_raw.get(
                "payload_hash"
            ):
                raise ReplayError(f"candidate {candidate_id} proposal artifact hash mismatch")
            if canonical_json(proposal.ast) != row["ast_json"]:
                raise ReplayError(f"candidate {candidate_id} AST differs from its artifact")
            parent_ids_raw: object = json.loads(row["parent_ids_json"])
            parent_ids = _string_list(parent_ids_raw, "candidate parent_ids")
            candidate = Candidate(
                candidate_id=candidate_id,
                task_id=row["task_id"],
                ast=proposal.ast,
                parent_ids=parent_ids,
                proposer_id=row["proposer_id"],
                operator_id=row["operator_id"],
                context_hash=row["context_hash"],
                payload_hash=payload_hash,
            )
            identity = {
                "task_id": candidate.task_id,
                "ast": candidate.ast,
                "parent_ids": list(candidate.parent_ids),
                "proposer_id": candidate.proposer_id,
                "operator_id": candidate.operator_id,
                "context_hash": candidate.context_hash,
                "payload_hash": candidate.payload_hash,
            }
            if sha256_json(identity) != candidate.candidate_id:
                raise ReplayError(f"candidate {candidate_id} identity hash mismatch")
            if candidate.context_hash != context.content_hash:
                raise ReplayError(f"candidate {candidate_id} context hash mismatch")
            result = oracle.evaluate(candidate)
            rebuilt_payload = deterministic_event_payload(
                candidate=candidate,
                result_payload=result.deterministic_payload(),
                step_index=expected_sequence,
            )
            rebuilt_event = SearchEvent.create(
                sequence=expected_sequence,
                event_type="candidate_evaluated",
                logical_cost=expected_sequence + 1,
                payload=rebuilt_payload,
                audit_timestamp=recorded_event.audit_timestamp,
            )
            if rebuilt_event.payload_json != recorded_event.payload_json:
                raise ReplayError(f"event {expected_sequence} deterministic payload diverged")
            if rebuilt_event.payload_hash != recorded_event.payload_hash:
                raise ReplayError(f"event {expected_sequence} deterministic hash diverged")

    rebuilt_results = deterministic_results(events)
    results_path = run_directory / "results.json"
    recorded_results = parse_json_object(read_text_artifact(results_path))
    if canonical_json(rebuilt_results) != canonical_json(recorded_results):
        raise ReplayError("recomputed results differ from the frozen results artifact")
    summary_hash = rebuilt_results.get("deterministic_summary_hash")
    if not isinstance(summary_hash, str):
        raise ReplayError("results artifact has no deterministic summary hash")
    return ReplayReport(
        run_id=run_id,
        event_count=len(events),
        event_payload_hashes=tuple(event.payload_hash for event in events),
        deterministic_summary_hash=summary_hash,
    )


def _replay_phase2(
    *, repository_root: Path, run_directory: Path, run_id: str, config: AppConfig
) -> ReplayReport:
    """Replay Phase 2 only from recorded candidate documents and frozen analysis."""

    if config.dsl is None:
        raise ReplayError("Phase 2 manifest has no DSL bounds")
    benchmark_root = benchmark_root_for_config(repository_root, config)
    task = load_public_task(benchmark_root, config.run.task_id)
    allowed = frozenset({SplitLabel.TRAINING, SplitLabel.DEVELOPMENT})
    store = HiddenTaskStore(benchmark_root)
    hidden = store.load(task.task_id, allowed_splits=allowed, purpose="phase2-replay")
    limits = AstLimits(
        max_depth=config.dsl.max_depth,
        max_nodes=config.dsl.max_nodes,
        max_cases=config.dsl.max_cases,
    )
    oracle = ExactDslOracle(hidden, limits=limits, response_mode=config.oracle.response_mode)
    context = ProposalContext(task=task.public_view())
    with RunDatabase(run_directory / "run.sqlite3", read_only=True) as database:
        state = database.state()
        if state.status != "completed":
            raise ReplayError("only completed runs can be replayed")
        events = database.events()
        for expected_sequence, recorded_event in enumerate(events):
            if recorded_event.sequence != expected_sequence:
                raise ReplayError("event sequence is not contiguous")
            if sha256_text(recorded_event.payload_json) != recorded_event.payload_hash:
                raise ReplayError(f"event {expected_sequence} payload hash mismatch")
            event_payload = recorded_event.payload
            candidate_raw = event_payload.get("candidate")
            if not isinstance(candidate_raw, dict):
                raise ReplayError(f"event {expected_sequence} has no candidate object")
            candidate_id = candidate_raw.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise ReplayError(f"event {expected_sequence} has no candidate id")
            row = database.candidate_record(candidate_id)
            artifact_relative = Path(row["artifact_name"])
            if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
                raise ReplayError("candidate artifact path escapes the run directory")
            proposal_json = read_text_artifact(run_directory / artifact_relative)
            try:
                document = DslCandidateDocument.from_json(
                    proposal_json,
                    limits=limits,
                    allowed_macros=frozenset(config.dsl.allowed_macros),
                )
            except ValueError as exc:
                raise ReplayError(f"candidate {candidate_id} artifact is invalid") from exc
            canonical = canonicalize(document.ast)
            if canonical != document.ast:
                raise ReplayError(f"candidate {candidate_id} artifact is noncanonical")
            digest = sha256_text(proposal_json)
            candidate_semantic_hash = semantic_hash(canonical, limits=limits)
            bits = encoded_length(canonical)
            if (
                digest != row["payload_hash"]
                or digest != candidate_raw.get("payload_hash")
                or ast_canonical_json(canonical) != row["canonical_ast_json"]
                or ast_canonical_json(canonical) != row["ast_json"]
                or candidate_semantic_hash != row["semantic_hash"]
                or bits != row["ast_bits"]
            ):
                raise ReplayError(f"candidate {candidate_id} persisted metadata diverged")
            parent_ids_raw: object = json.loads(row["parent_ids_json"])
            parent_ids = _string_list(parent_ids_raw, "candidate parent_ids")
            candidate = Candidate(
                candidate_id=candidate_id,
                task_id=row["task_id"],
                ast=canonical,
                parent_ids=parent_ids,
                proposer_id=row["proposer_id"],
                operator_id=row["operator_id"],
                context_hash=row["context_hash"],
                payload_hash=digest,
                semantic_hash=candidate_semantic_hash,
            )
            identity = {
                "candidate_identity_schema": 2,
                "task_id": candidate.task_id,
                "canonical_ast": ast_to_value(canonical),
                "parent_ids": list(parent_ids),
                "proposer_id": candidate.proposer_id,
                "operator_id": candidate.operator_id,
                "context_hash": candidate.context_hash,
                "payload_hash": candidate.payload_hash,
                "coding_version": PREFIX_CODE_VERSION,
            }
            if sha256_json(identity) != candidate.candidate_id:
                raise ReplayError(f"candidate {candidate_id} identity hash mismatch")
            if candidate.context_hash != context.content_hash:
                raise ReplayError(f"candidate {candidate_id} context hash mismatch")
            evaluated = oracle.evaluate(canonical)
            evaluation_row = database.evaluation_record(candidate_id)
            try:
                recorded_result: object = json.loads(evaluation_row["result_json"])
                rebuilt_result: object = json.loads(canonical_json(evaluated.result))
            except json.JSONDecodeError as exc:
                raise ReplayError(f"candidate {candidate_id} evaluation JSON is invalid") from exc
            if not isinstance(recorded_result, dict) or not isinstance(rebuilt_result, dict):
                raise ReplayError(f"candidate {candidate_id} evaluation is not an object")
            recorded_runtime = recorded_result.pop("runtime_ns", None)
            rebuilt_result.pop("runtime_ns", None)
            if (
                evaluation_row["oracle_version"] != config.oracle.oracle_id
                or recorded_runtime != evaluation_row["runtime_ns"]
                or canonical_json(recorded_result) != canonical_json(rebuilt_result)
            ):
                raise ReplayError(f"candidate {candidate_id} frozen evaluation diverged")
            rebuilt_payload = phase2_deterministic_event_payload(
                candidate=candidate,
                result_payload=evaluated.result.deterministic_payload(),
                step_index=expected_sequence,
                ast_bits=bits,
            )
            rebuilt_event = SearchEvent.create(
                sequence=expected_sequence,
                event_type="candidate_evaluated",
                logical_cost=expected_sequence + 1,
                payload=rebuilt_payload,
                audit_timestamp=recorded_event.audit_timestamp,
            )
            if rebuilt_event.payload_json != recorded_event.payload_json:
                raise ReplayError(f"event {expected_sequence} deterministic payload diverged")
            if rebuilt_event.payload_hash != recorded_event.payload_hash:
                raise ReplayError(f"event {expected_sequence} deterministic hash diverged")
    analysis_manifest = read_text_artifact(run_directory / "analysis" / "manifest.json")
    rebuilt_results = phase2_deterministic_results(
        events, analysis_manifest_hash=sha256_text(analysis_manifest)
    )
    recorded_results = parse_json_object(read_text_artifact(run_directory / "results.json"))
    if canonical_json(rebuilt_results) != canonical_json(recorded_results):
        raise ReplayError("recomputed Phase 2 results differ from frozen results")
    summary_hash = rebuilt_results.get("deterministic_summary_hash")
    if not isinstance(summary_hash, str):
        raise ReplayError("Phase 2 results have no deterministic summary hash")
    return ReplayReport(
        run_id=run_id,
        event_count=len(events),
        event_payload_hashes=tuple(event.payload_hash for event in events),
        deterministic_summary_hash=summary_hash,
    )


def _replay_phase3(
    *,
    repository_root: Path,
    run_directory: Path,
    run_id: str,
    config: AppConfig,
    manifest: JsonObject,
) -> ReplayReport:
    """Replay recorded Phase 3 proposals without operators or live scheduler selection."""

    from world_model_search.dsl.interpreter import semantic_hash as phase3_semantic_hash
    from world_model_search.dsl.versions import PHASE3_MANIFEST_SCHEMA_VERSION
    from world_model_search.scheduler.uniform import SchedulerDecision
    from world_model_search.search.operators import AttemptOutcome
    from world_model_search.search.phase3 import (
        _attempt_event_payload,
        _budget_after_evaluation,
        _mechanism,
        _public_context_value,
        authority_from_manifest,
        phase3_results,
        prepare_phase3,
    )
    from world_model_search.search.phase3_types import BudgetState, phase3_candidate

    if manifest.get("manifest_schema_version") != PHASE3_MANIFEST_SCHEMA_VERSION:
        raise ReplayError("Phase 3 manifest schema mismatch")
    if config.budget is None or config.dsl is None:
        raise ReplayError("Phase 3 manifest has no budget/DSL settings")
    authority = authority_from_manifest(manifest)
    prepared = prepare_phase3(
        repository_root=repository_root,
        config=config,
        authority=authority,
        purpose=(
            "phase3-locked-validation-replay"
            if authority.mode == "phase3-locked-validation-once-v1"
            else "phase3-replay"
        ),
    )
    mechanism = _mechanism(config, prepared.task)
    budget = BudgetState(
        proposal_attempt_cap=config.budget.proposal_attempt_cap,
        oracle_call_cap=config.budget.oracle_call_cap,
    )
    canonical_seen: set[str] = set()
    semantic_seen: set[str] = set()
    candidate_by_id: dict[str, Candidate] = {}
    rebuilt_events: list[SearchEvent] = []
    with RunDatabase(run_directory / "run.sqlite3", read_only=True) as database:
        state = database.state()
        if state.status != "completed":
            raise ReplayError("only completed runs can be replayed")
        events = database.events()
        attempts = database.phase3_attempt_records()
        evaluations = {row["attempt_index"]: row for row in database.evaluation_records()}
        transitions = {row["attempt_index"]: row for row in database.phase3_transition_records()}
        if len(events) != len(attempts):
            raise ReplayError("Phase 3 event/attempt counts diverge")
        for attempt_index, (event, attempt_row) in enumerate(zip(events, attempts, strict=True)):
            if event.sequence != attempt_index or attempt_row["attempt_index"] != attempt_index:
                raise ReplayError("Phase 3 attempt sequence is not contiguous")
            artifact_name = attempt_row["artifact_name"]
            relative = Path(artifact_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ReplayError("Phase 3 proposal path escapes the run directory")
            artifact_text = read_text_artifact(run_directory / relative)
            if (
                sha256_text(artifact_text) != attempt_row["artifact_hash"]
                or sha256_text(event.payload_json) != event.payload_hash
            ):
                raise ReplayError("Phase 3 proposal/event content hash mismatch")
            artifact = parse_json_object(artifact_text)
            if set(artifact) != {
                "artifact_version",
                "attempt_index",
                "task_id",
                "public_context",
                "public_context_hash",
                "ordered_parent_ids",
                "operator",
                "candidate_document",
                "submitted_source_ast",
            }:
                raise ReplayError("Phase 3 proposal artifact fields are malformed")
            scheduler_raw = event.payload.get("scheduler")
            try:
                scheduler = SchedulerDecision.from_value(scheduler_raw)
            except ValueError as exc:
                raise ReplayError("recorded scheduler decision is invalid") from exc
            if (
                scheduler.remaining_proposal_attempts != budget.remaining_proposal_attempts
                or scheduler.remaining_oracle_calls != budget.remaining_oracle_calls
            ):
                raise ReplayError("recorded scheduler remaining budget diverged")
            operator = artifact.get("operator")
            if (
                not isinstance(operator, dict)
                or canonical_json(operator) != attempt_row["operator_json"]
            ):
                raise ReplayError("recorded operator decision diverged")
            parent_ids = artifact.get("ordered_parent_ids")
            if not isinstance(parent_ids, list) or not all(
                isinstance(item, str) for item in parent_ids
            ):
                raise ReplayError("recorded parent IDs are malformed")
            parent_id_values = cast(list[str], parent_ids)
            try:
                parents = tuple(
                    CandidateSummary(parent_id, candidate_by_id[parent_id].ast)
                    for parent_id in parent_id_values
                )
            except KeyError as exc:
                raise ReplayError("Phase 3 parent does not exist earlier in replay") from exc
            context = ProposalContext(
                task=prepared.task.public_view(), parents=parents, feedback=()
            )
            if context.content_hash != artifact.get("public_context_hash") or canonical_json(
                _public_context_value(context)
            ) != canonical_json(artifact.get("public_context")):
                raise ReplayError("Phase 3 public proposal context diverged")
            candidate_value = artifact.get("candidate_document")
            candidate: Candidate | None = None
            result: OracleResult | None = None
            decision = None
            canonical_duplicate = False
            semantic_duplicate = False
            if isinstance(candidate_value, dict):
                try:
                    document = DslCandidateDocument.from_json(
                        canonical_json(candidate_value), limits=prepared.limits
                    )
                except ValueError as exc:
                    raise ReplayError("Phase 3 recorded candidate is invalid") from exc
                if document.ast != canonicalize(document.ast):
                    raise ReplayError("Phase 3 recorded candidate is noncanonical")
                candidate_semantic = phase3_semantic_hash(document.ast, limits=prepared.limits)
                canonical_key = ast_canonical_json(document.ast)
                canonical_duplicate = canonical_key in canonical_seen
                semantic_duplicate = candidate_semantic in semantic_seen
                operator_id = operator.get("operator_id")
                if not isinstance(operator_id, str):
                    raise ReplayError("Phase 3 operator identifier is missing")
                candidate = phase3_candidate(
                    task_id=prepared.task.task_id,
                    ast=document.ast,
                    parent_ids=tuple(parent_id_values),
                    proposer_id="mutation",
                    operator_id=operator_id,
                    context_hash=context.content_hash,
                    payload_hash=sha256_text(canonical_json(candidate_value)),
                    coding_version=PREFIX_CODE_VERSION,
                    semantic_hash=candidate_semantic,
                )
                row = database.candidate_record(candidate.candidate_id)
                if (
                    row["canonical_ast_json"] != ast_canonical_json(document.ast)
                    or row["context_hash"] != context.content_hash
                    or row["semantic_hash"] != candidate_semantic
                ):
                    raise ReplayError("Phase 3 candidate persistence diverged")
                evaluated = ExactDslOracle(
                    prepared.hidden,
                    limits=prepared.limits,
                    response_mode=config.oracle.response_mode,
                ).evaluate(document.ast)
                result = evaluated.result
                decision = mechanism.insert(candidate, result)
                budget = _budget_after_evaluation(
                    budget,
                    operator_attempt=operator_id != "public-baseline-initialization",
                    canonical_duplicate=canonical_duplicate,
                    semantic_duplicate=semantic_duplicate,
                    decision=decision,
                )
                evaluation = evaluations.get(attempt_index)
                transition = transitions.get(attempt_index)
                if evaluation is None or transition is None:
                    raise ReplayError("Phase 3 evaluation/transition is missing")
                recorded_result = parse_json_object(evaluation["result_json"])
                recorded_result.pop("runtime_ns", None)
                rebuilt_result = parse_json_object(canonical_json(result))
                rebuilt_result.pop("runtime_ns", None)
                if canonical_json(recorded_result) != canonical_json(rebuilt_result) or transition[
                    "decision_json"
                ] != canonical_json(decision.to_value()):
                    raise ReplayError("Phase 3 oracle/archive replay diverged")
                candidate_by_id[candidate.candidate_id] = candidate
                canonical_seen.add(canonical_key)
                semantic_seen.add(candidate_semantic)
            else:
                outcome = operator.get("outcome")
                if outcome not in {AttemptOutcome.NO_OP.value, AttemptOutcome.REJECTED.value}:
                    raise ReplayError("candidate-free attempt has invalid operator outcome")
                budget = budget.updated(
                    proposal_attempts=1,
                    operator_attempts=1,
                    invalid_outputs=int(outcome == AttemptOutcome.REJECTED.value),
                    noop_outputs=int(outcome == AttemptOutcome.NO_OP.value),
                    scheduler_selections=1,
                )
            rebuilt = SearchEvent.create(
                sequence=attempt_index,
                event_type="phase3_proposal_attempt",
                logical_cost=budget.oracle_invocations,
                payload=_attempt_event_payload(
                    attempt_index=attempt_index,
                    task_id=prepared.task.task_id,
                    scheduler=scheduler,
                    operator=operator,
                    artifact_hash=attempt_row["artifact_hash"],
                    artifact_name=artifact_name,
                    candidate=candidate,
                    canonical_duplicate=canonical_duplicate,
                    semantic_duplicate=semantic_duplicate,
                    result=result,
                    decision=decision,
                    budget=budget,
                ),
                audit_timestamp=event.audit_timestamp,
            )
            if (
                rebuilt.payload_json != event.payload_json
                or rebuilt.payload_hash != event.payload_hash
            ):
                raise ReplayError("Phase 3 deterministic event replay diverged")
            rebuilt_events.append(rebuilt)
        if canonical_json(budget.to_value()) != canonical_json(database.phase3_budget().to_value()):
            raise ReplayError("Phase 3 final budget replay diverged")
    rebuilt_results = phase3_results(
        tuple(rebuilt_events), budget=budget, mechanism=mechanism, config=config
    )
    analysis_manifest = read_text_artifact(run_directory / "analysis" / "manifest.json")
    rebuilt_results["analysis_manifest_hash"] = sha256_text(analysis_manifest)
    rebuilt_results["deterministic_summary_hash"] = sha256_json(rebuilt_results)
    recorded_results = parse_json_object(read_text_artifact(run_directory / "results.json"))
    if canonical_json(rebuilt_results) != canonical_json(recorded_results):
        raise ReplayError("Phase 3 frozen results replay diverged")
    summary_hash = rebuilt_results["deterministic_summary_hash"]
    if not isinstance(summary_hash, str):
        raise ReplayError("Phase 3 result hash is malformed")
    return ReplayReport(
        run_id=run_id,
        event_count=len(rebuilt_events),
        event_payload_hashes=tuple(event.payload_hash for event in rebuilt_events),
        deterministic_summary_hash=summary_hash,
    )
