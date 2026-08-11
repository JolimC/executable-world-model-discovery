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
from world_model_search.dsl.ast import AstLimits, BitExpr
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.interpreter import semantic_hash
from world_model_search.dsl.json_schema import (
    DslCandidateDocument,
    ast_canonical_json,
    ast_to_value,
)
from world_model_search.dsl.versions import PREFIX_CODE_VERSION
from world_model_search.errors import ConfigurationError, PersistenceError, ReplayError
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
    if config.schema_version == 4:
        return _replay_phase4(
            repository_root=repository_root,
            run_directory=run_directory,
            run_id=run_id,
            config=config,
            manifest=manifest,
        )
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


def _replay_phase4(
    *,
    repository_root: Path,
    run_directory: Path,
    run_id: str,
    config: AppConfig,
    manifest: JsonObject,
) -> ReplayReport:
    """Replay Phase 4 from immutable request/response records with no provider or cache."""

    from world_model_search.domain.types import ProposalRole
    from world_model_search.model.prompts import ParentScoreFeedback, render_prompt
    from world_model_search.model.schema import (
        BATCH_SCHEMA_NAME,
        BATCH_SCHEMA_VERSION,
        BatchEnvelopeError,
        candidate_batch_json_schema,
        parse_candidate_batch,
    )
    from world_model_search.model.types import ModelError, ModelRequest
    from world_model_search.persistence.phase4_database import Phase4Database
    from world_model_search.phase4_versions import PHASE4_MANIFEST_SCHEMA_VERSION
    from world_model_search.search.phase4 import (
        _candidate_from_row,
        _event_payload,
        _feedback,
        _mechanism,
        _request_from_artifact,
        _response_from_artifact,
        _result_from_json,
        authority_from_manifest,
        phase4_results,
        prepare_phase4,
    )
    from world_model_search.search.phase4_types import Phase4BudgetState, phase4_candidate

    if manifest.get("manifest_schema_version") != PHASE4_MANIFEST_SCHEMA_VERSION:
        raise ReplayError("Phase 4 manifest schema mismatch")
    if config.dsl is None or config.model is None:
        raise ReplayError("Phase 4 replay requires frozen DSL/model settings")
    try:
        prepared = prepare_phase4(
            repository_root=repository_root,
            config=config,
            authority=authority_from_manifest(manifest),
            purpose="phase4-recorded-replay",
        )
    except (ConfigurationError, PersistenceError, ValueError) as exc:
        raise ReplayError(f"Phase 4 replay authority failed: {exc}") from exc
    mechanism = _mechanism(config, prepared.task)
    oracle = ExactDslOracle(
        prepared.hidden,
        limits=prepared.limits,
        response_mode=config.oracle.response_mode,
    )
    with Phase4Database(run_directory / "run.sqlite3", read_only=True) as database:
        recorded_status = database.state().status
        if recorded_status not in {
            "completed",
            "cost-cap-exhausted",
            "usage-uncertain",
            "failed",
        }:
            raise ReplayError("Phase 4 replay requires a terminal recorded run")
        request_rows = tuple(dict(row) for row in database.requests())
        item_rows = {
            (int(row["request_index"]), int(row["ordinal"])): dict(row) for row in database.items()
        }
        candidate_rows = {str(row["candidate_id"]): row for row in database.candidates()}
        evaluation_rows = {int(row["evaluation_index"]): row for row in database.evaluations()}
        transition_rows = {int(row["evaluation_index"]): row for row in database.transitions()}
        candidates: dict[str, Candidate] = {}

        # Recreate every fixed prompt/request and validate every submitted response item.
        for row in request_rows:
            request_index = int(row["request_index"])
            prompt_name = Path(str(row["prompt_artifact"]))
            request_name = Path(str(row["request_artifact"]))
            if any(
                path.is_absolute() or ".." in path.parts for path in (prompt_name, request_name)
            ):
                raise ReplayError("Phase 4 request path escapes its run")
            prompt_text = read_text_artifact(run_directory / prompt_name)
            if sha256_text(prompt_text) != row["prompt_hash"]:
                raise ReplayError(f"Phase 4 prompt {request_index} hash mismatch")
            request_artifact = parse_json_object(read_text_artifact(run_directory / request_name))
            try:
                recorded_request = _request_from_artifact(
                    request_artifact, expected_hash=row["request_hash"]
                )
            except PersistenceError as exc:
                raise ReplayError(f"Phase 4 request {request_index} is corrupt: {exc}") from exc
            if recorded_request.rendered_input != prompt_text:
                raise ReplayError(f"Phase 4 request {request_index} identity mismatch")
            parent_ids_raw: object = json.loads(str(row["ordered_parent_ids_json"]))
            parent_ids = _string_list(parent_ids_raw, "Phase 4 ordered parent IDs")
            parent = None
            feedback: ParentScoreFeedback | None = None
            if parent_ids:
                if len(parent_ids) != 1:
                    raise ReplayError("Phase 4 iterative request must have one primary parent")
                parent_candidate = _candidate_from_row(
                    candidate_rows[parent_ids[0]], prepared.limits
                )
                parent = CandidateSummary(parent_candidate.candidate_id, parent_candidate.ast)
                parent_result = _result_from_json(
                    str(database.candidate_result(parent_candidate.candidate_id)["result_json"])
                )
                feedback = _feedback(parent_candidate.candidate_id, parent_result)
            role = ProposalRole(str(row["role"]))
            template, version, rendered = render_prompt(
                task=prepared.task.public_view(),
                role=role,
                requested_batch_size=int(row["batch_size"]),
                parent=parent,
                feedback=feedback,
            )
            settings = config.model.request_settings()
            settings["independent_sample_index"] = int(row["logical_call_index"])
            rebuilt_request = ModelRequest(
                backend_id=str(row["backend_id"]),
                provider_id=str(row["provider_id"]),
                resolved_model=config.model.resolved_model,
                endpoint=config.model.endpoint,
                service_tier=config.model.service_tier,
                prompt_template=template,
                prompt_version=version,
                rendered_input=rendered,
                structured_schema_name=BATCH_SCHEMA_NAME,
                structured_schema_version=BATCH_SCHEMA_VERSION,
                structured_schema=candidate_batch_json_schema(
                    role=role, batch_size=int(row["batch_size"])
                ),
                role=role,
                requested_batch_size=int(row["batch_size"]),
                settings=settings,
            )
            if rebuilt_request.request_hash != recorded_request.request_hash:
                raise ReplayError(f"Phase 4 request {request_index} prompt reconstruction diverged")
            response_name_value = row.get("response_artifact")
            if response_name_value is None:
                if row["state"] not in {"failed", "usage-uncertain"}:
                    raise ReplayError("Phase 4 response-free request has an invalid state")
                continue
            response_name = Path(str(response_name_value))
            if response_name.is_absolute() or ".." in response_name.parts:
                raise ReplayError("Phase 4 response path escapes its run")
            response_text = read_text_artifact(run_directory / response_name)
            if sha256_text(response_text) != row["response_hash"]:
                raise ReplayError(f"Phase 4 response {request_index} hash mismatch")
            response_artifact = parse_json_object(response_text)
            if row["state"] in {"failed", "usage-uncertain"}:
                try:
                    failure_error = ModelError.from_value(response_artifact.get("error"))
                except ValueError as exc:
                    raise ReplayError("Phase 4 failure artifact is malformed") from exc
                if (
                    set(response_artifact) != {"artifact_version", "request_index", "error"}
                    or response_artifact.get("artifact_version") != "phase4-model-failure-v1"
                    or response_artifact.get("request_index") != request_index
                    or canonical_json(failure_error.to_value()) != row["error_json"]
                ):
                    raise ReplayError("Phase 4 failure artifact diverged")
                continue
            try:
                response = _response_from_artifact(response_artifact, request=recorded_request)
            except PersistenceError as exc:
                raise ReplayError(f"Phase 4 response {request_index} is corrupt: {exc}") from exc
            if response.request_hash != rebuilt_request.request_hash:
                raise ReplayError("Phase 4 response/request identity mismatch")
            try:
                batch = parse_candidate_batch(
                    response.raw_text,
                    expected_role=role,
                    requested_batch_size=int(row["batch_size"]),
                    limits=prepared.limits,
                    allowed_macros=frozenset(config.dsl.allowed_macros),
                )
            except BatchEnvelopeError:
                if row["state"] != "schema-failure":
                    raise ReplayError("unrecorded Phase 4 response schema failure") from None
                continue
            if row["state"] not in {"responded", "completed"}:
                raise ReplayError("valid Phase 4 response has an invalid lifecycle state")
            for item in batch.items:
                persisted = item_rows.get((request_index, item.ordinal))
                if persisted is None:
                    raise ReplayError("Phase 4 response item is missing")
                expected_status = "accepted" if item.accepted else "rejected"
                if persisted["outcome"] != expected_status:
                    raise ReplayError("Phase 4 item validation outcome diverged")
                submitted = (
                    canonical_json(item.submitted_document)
                    if item.submitted_document is not None
                    else None
                )
                if persisted["submitted_document_json"] != submitted:
                    raise ReplayError("Phase 4 submitted item artifact diverged")

        rebuilt_events: list[SearchEvent] = []
        for sequence, event in enumerate(database.events()):
            if event.sequence != sequence or sha256_text(event.payload_json) != event.payload_hash:
                raise ReplayError("Phase 4 event sequence/hash diverged")
            payload = event.payload
            try:
                event_budget = Phase4BudgetState.from_value(payload.get("budget"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayError("Phase 4 event budget is invalid") from exc
            evaluation_index = payload.get("evaluation_index")
            event_request_index = payload.get("request_index")
            item_ordinal = payload.get("item_ordinal")
            candidate_value = payload.get("candidate")
            rebuilt_candidate: Candidate | None = None
            rebuilt_result: OracleResult | None = None
            rebuilt_decision = None
            rejection = payload.get("rejection_reason")
            if isinstance(evaluation_index, int) and isinstance(candidate_value, dict):
                evaluation = evaluation_rows.get(evaluation_index)
                if evaluation is None:
                    raise ReplayError("Phase 4 event evaluation is missing")
                candidate_id = candidate_value.get("candidate_id")
                if not isinstance(candidate_id, str) or candidate_id not in candidate_rows:
                    raise ReplayError("Phase 4 event candidate is missing")
                stored = _candidate_from_row(candidate_rows[candidate_id], prepared.limits)
                if not isinstance(stored.ast, BitExpr) or stored.semantic_hash is None:
                    raise ReplayError("Phase 4 candidate is not a typed DSL candidate")
                rebuilt_candidate = phase4_candidate(
                    task_id=stored.task_id,
                    ast=stored.ast,
                    parent_ids=stored.parent_ids,
                    operator_id=stored.operator_id,
                    context_hash=stored.context_hash,
                    payload_hash=stored.payload_hash,
                    semantic_hash=semantic_hash(stored.ast, limits=prepared.limits),
                )
                if rebuilt_candidate.candidate_id != stored.candidate_id:
                    raise ReplayError("Phase 4 candidate identity diverged")
                rebuilt_result = oracle.evaluate(stored.ast).result
                if rebuilt_result.deterministic_payload() != payload.get("oracle_result"):
                    raise ReplayError("Phase 4 exact oracle replay diverged")
                rebuilt_decision = mechanism.insert(rebuilt_candidate, rebuilt_result)
                transition = transition_rows.get(evaluation_index)
                if transition is None or transition["decision_json"] != canonical_json(
                    rebuilt_decision.to_value()
                ):
                    raise ReplayError("Phase 4 archive/incumbent transition diverged")
                candidates[stored.candidate_id] = stored
            elif candidate_value is not None or not isinstance(rejection, str):
                raise ReplayError("Phase 4 candidate-free event is malformed")
            rebuilt_payload = _event_payload(
                evaluation_index=evaluation_index if isinstance(evaluation_index, int) else None,
                request_index=(
                    event_request_index if isinstance(event_request_index, int) else None
                ),
                item_ordinal=item_ordinal if isinstance(item_ordinal, int) else None,
                candidate=rebuilt_candidate,
                result=rebuilt_result,
                decision=rebuilt_decision,
                rejection_reason=rejection if isinstance(rejection, str) else None,
                budget=event_budget,
            )
            rebuilt = SearchEvent.create(
                sequence=sequence,
                event_type=event.event_type,
                logical_cost=event.logical_cost,
                payload=rebuilt_payload,
                audit_timestamp=event.audit_timestamp,
            )
            if (
                rebuilt.payload_json != event.payload_json
                or rebuilt.payload_hash != event.payload_hash
            ):
                raise ReplayError("Phase 4 deterministic event replay diverged")
            rebuilt_events.append(rebuilt)

        final_budget = database.budget()
        rebuilt_results = phase4_results(
            database=database,
            budget=final_budget,
            mechanism=mechanism,
            config=config,
            status=recorded_status,
        )
    analysis_manifest = read_text_artifact(run_directory / "analysis" / "manifest.json")
    rebuilt_results["analysis_manifest_hash"] = sha256_text(analysis_manifest)
    rebuilt_results["deterministic_summary_hash"] = sha256_json(rebuilt_results)
    recorded_results = parse_json_object(read_text_artifact(run_directory / "results.json"))
    if canonical_json(rebuilt_results) != canonical_json(recorded_results):
        raise ReplayError("Phase 4 frozen results replay diverged")
    summary_hash = rebuilt_results.get("deterministic_summary_hash")
    if not isinstance(summary_hash, str):
        raise ReplayError("Phase 4 deterministic summary hash is malformed")
    return ReplayReport(
        run_id=run_id,
        event_count=len(rebuilt_events),
        event_payload_hashes=tuple(event.payload_hash for event in rebuilt_events),
        deterministic_summary_hash=summary_hash,
        proposer_invocations=0,
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
