"""Deterministic replay from recorded proposer artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from world_model_search.config import AppConfig, config_from_mapping
from world_model_search.domain.types import (
    Candidate,
    CandidatePayload,
    ProposalContext,
    SearchEvent,
)
from world_model_search.errors import ConfigurationError, ReplayError
from world_model_search.oracle.mock import MockOracle
from world_model_search.persistence.artifacts import read_text_artifact
from world_model_search.persistence.database import RunDatabase
from world_model_search.search.fixture import make_fixture_task
from world_model_search.search.loop import (
    deterministic_event_payload,
    deterministic_results,
    load_manifest,
    validate_run_id,
)
from world_model_search.serialization import (
    JsonObject,
    canonical_json,
    parse_json_object,
    sha256_json,
    sha256_text,
)


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
