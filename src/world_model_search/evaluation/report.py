"""Generate a Phase 0 report using frozen data only."""

from __future__ import annotations

import json
from pathlib import Path

from world_model_search.dsl.versions import PHASE3_MANIFEST_SCHEMA_VERSION
from world_model_search.errors import PersistenceError
from world_model_search.persistence.artifacts import (
    read_text_artifact,
    write_json_exclusive,
    write_text_exclusive,
)
from world_model_search.persistence.database import RunDatabase
from world_model_search.persistence.manifest import MANIFEST_SCHEMA_VERSION
from world_model_search.persistence.phase4_database import Phase4Database
from world_model_search.phase4_versions import PHASE4_MANIFEST_SCHEMA_VERSION
from world_model_search.search.loop import load_manifest, validate_run_id
from world_model_search.serialization import JsonObject, JsonValue, canonical_json, sha256_text


def create_recorded_report(
    *, repository_root: Path, runs_root: Path, run_id: str, output_directory: Path
) -> tuple[Path, Path]:
    """Read manifest/results/ledger without invoking search, proposer, or oracle."""

    validate_run_id(run_id)
    if runs_root.is_absolute() or ".." in runs_root.parts:
        raise PersistenceError("runs root must be repository-relative without '..'")
    run_directory = repository_root / runs_root / run_id
    manifest = load_manifest(run_directory)
    results_path = run_directory / "results.json"
    if not results_path.is_file():
        raise PersistenceError("a completed results artifact is required for reporting")
    try:
        raw_results: object = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceError(f"cannot read results artifact: {exc}") from exc
    if not isinstance(raw_results, dict):
        raise PersistenceError("results artifact must be a JSON object")
    results: JsonObject = raw_results
    if manifest.get("manifest_schema_version") == PHASE4_MANIFEST_SCHEMA_VERSION:
        return _create_phase4_report(
            run_id=run_id,
            run_directory=run_directory,
            output_directory=output_directory,
            manifest=manifest,
            results=results,
        )
    with RunDatabase(run_directory / "run.sqlite3", read_only=True) as database:
        state = database.state()
        event_count = len(database.events())
        candidate_count = database.table_count("candidate")
        evaluation_count = database.table_count("evaluation")
        candidate_records = database.candidate_records()
        evaluation_records = database.evaluation_records()
        phase3_attempt_records = (
            database.phase3_attempt_records()
            if manifest.get("manifest_schema_version") == PHASE3_MANIFEST_SCHEMA_VERSION
            else ()
        )
        phase3_transition_records = (
            database.phase3_transition_records()
            if manifest.get("manifest_schema_version") == PHASE3_MANIFEST_SCHEMA_VERSION
            else ()
        )
        phase3_lineage_records = (
            database.phase3_lineage_records()
            if manifest.get("manifest_schema_version") == PHASE3_MANIFEST_SCHEMA_VERSION
            else ()
        )
    if state.status != "completed":
        raise PersistenceError("report command requires a completed run")

    configuration_hash = manifest.get("configuration_hash")
    if not isinstance(configuration_hash, str):
        raise PersistenceError("manifest has no configuration hash")
    if manifest.get("manifest_schema_version") == PHASE3_MANIFEST_SCHEMA_VERSION:
        return _create_phase3_report(
            run_id=run_id,
            run_directory=run_directory,
            output_directory=output_directory,
            configuration_hash=configuration_hash,
            results=results,
            event_count=event_count,
            candidate_records=tuple(dict(record) for record in candidate_records),
            evaluation_records=tuple(dict(record) for record in evaluation_records),
            attempt_records=tuple(dict(record) for record in phase3_attempt_records),
            transition_records=tuple(dict(record) for record in phase3_transition_records),
            lineage_records=tuple(dict(record) for record in phase3_lineage_records),
        )
    if manifest.get("manifest_schema_version") == MANIFEST_SCHEMA_VERSION:
        return _create_phase2_report(
            run_id=run_id,
            run_directory=run_directory,
            output_directory=output_directory,
            configuration_hash=configuration_hash,
            results=results,
            event_count=event_count,
            candidate_count=candidate_count,
            evaluation_count=evaluation_count,
            candidate_record_hashes=tuple(
                sha256_text(canonical_json(dict(record))) for record in candidate_records
            ),
            evaluation_record_hashes=tuple(
                sha256_text(canonical_json(dict(record))) for record in evaluation_records
            ),
        )
    report: JsonObject = {
        "schema_version": 1,
        "run_id": run_id,
        "source": "frozen-run-artifacts-only",
        "configuration_hash": configuration_hash,
        "recorded_event_count": event_count,
        "results": results,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "summary.json"
    markdown_path = output_directory / "summary.md"
    write_json_exclusive(json_path, report)
    metrics = results.get("metrics")
    markdown = (
        "# Phase 0 recorded run report\n\n"
        f"- Run: `{run_id}`\n"
        "- Source: frozen manifest, SQLite event ledger, and results artifact\n"
        f"- Recorded events: {event_count}\n"
        f"- Metrics: `{json.dumps(metrics, sort_keys=True)}`\n"
        f"- Deterministic summary hash: `{results.get('deterministic_summary_hash')}`\n"
    )
    write_text_exclusive(markdown_path, markdown)
    return json_path, markdown_path


def _create_phase4_report(
    *,
    run_id: str,
    run_directory: Path,
    output_directory: Path,
    manifest: JsonObject,
    results: JsonObject,
) -> tuple[Path, Path]:
    """Verify and copy Phase 4 records without model, cache, search, or oracle access."""

    with Phase4Database(run_directory / "run.sqlite3", read_only=True) as database:
        state = database.state()
        if state.status not in {
            "completed",
            "cost-cap-exhausted",
            "usage-uncertain",
            "failed",
        }:
            raise PersistenceError("report command requires a terminal Phase 4 run")
        requests = tuple(dict(row) for row in database.requests())
        counts: JsonObject = {
            "events": database.table_count("event"),
            "candidates": database.table_count("candidate"),
            "evaluations": database.table_count("evaluation"),
            "model_requests": database.table_count("model_request"),
            "proposal_items": database.table_count("proposal_item"),
            "archive_transitions": database.table_count("archive_transition"),
            "lineage_edges": database.table_count("lineage_edge"),
        }
    artifact_hashes: JsonObject = {}
    for request in requests:
        for name_field, hash_field in (
            ("prompt_artifact", "prompt_hash"),
            ("response_artifact", "response_hash"),
        ):
            name = request.get(name_field)
            digest = request.get(hash_field)
            if name is None and digest is None:
                continue
            if not isinstance(name, str) or not isinstance(digest, str):
                raise PersistenceError("Phase 4 request artifact metadata is incomplete")
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise PersistenceError("Phase 4 request artifact path escapes its run")
            if sha256_text(read_text_artifact(run_directory / relative)) != digest:
                raise PersistenceError(f"Phase 4 request artifact hash mismatch: {name}")
            artifact_hashes[name] = digest
        request_name = request.get("request_artifact")
        request_hash = request.get("request_hash")
        if not isinstance(request_name, str) or not isinstance(request_hash, str):
            raise PersistenceError("Phase 4 request identity artifact metadata is incomplete")
        request_text = read_text_artifact(run_directory / request_name)
        try:
            request_value: object = json.loads(request_text)
        except json.JSONDecodeError as exc:
            raise PersistenceError("Phase 4 request identity artifact is invalid") from exc
        if not isinstance(request_value, dict) or request_value.get("request_hash") != request_hash:
            raise PersistenceError("Phase 4 request identity artifact hash diverged")
        artifact_hashes[request_name] = sha256_text(request_text)

    analysis_root = run_directory / "analysis"
    manifest_text = read_text_artifact(analysis_root / "manifest.json")
    try:
        analysis_manifest: object = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise PersistenceError("Phase 4 analysis manifest is invalid") from exc
    if not isinstance(analysis_manifest, dict):
        raise PersistenceError("Phase 4 analysis manifest must be an object")
    files = analysis_manifest.get("files")
    diagnostic_names = analysis_manifest.get("non_deterministic_diagnostic_files", [])
    if not isinstance(files, dict) or not all(
        isinstance(name, str) and isinstance(digest, str) for name, digest in files.items()
    ):
        raise PersistenceError("Phase 4 analysis file map is invalid")
    if diagnostic_names != ["runtime-diagnostics.json"]:
        raise PersistenceError("Phase 4 runtime diagnostic declaration is invalid")
    frozen: dict[str, str] = {}
    for name, digest in files.items():
        content = read_text_artifact(analysis_root / name)
        if sha256_text(content) != digest:
            raise PersistenceError(f"frozen Phase 4 analysis hash mismatch: {name}")
        frozen[name] = content
    runtime_diagnostics = read_text_artifact(analysis_root / "runtime-diagnostics.json")
    if results.get("analysis_manifest_hash") != sha256_text(manifest_text):
        raise PersistenceError("Phase 4 results/analysis manifest hash mismatch")
    configuration_hash = manifest.get("configuration_hash")
    if not isinstance(configuration_hash, str):
        raise PersistenceError("Phase 4 manifest has no configuration hash")
    report: JsonObject = {
        "schema_version": 4,
        "run_id": run_id,
        "source": "frozen-run-artifacts-only",
        "evidence_class": results.get("evidence_class"),
        "scientific_conclusion": "not-established-by-fake-evidence"
        if results.get("evidence_class") == "fake"
        else "individual-live-run-only",
        "configuration_hash": configuration_hash,
        "recorded_counts": counts,
        "request_artifact_hashes": artifact_hashes,
        "analysis_manifest_hash": sha256_text(manifest_text),
        "analysis_file_hashes": files,
        "diagnostic_file_hashes": {"runtime-diagnostics.json": sha256_text(runtime_diagnostics)},
        "runtime_diagnostics": json.loads(runtime_diagnostics),
        "results": results,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, content in frozen.items():
        write_text_exclusive(output_directory / name, content + "\n")
    write_text_exclusive(output_directory / "runtime-diagnostics.json", runtime_diagnostics + "\n")
    json_path = output_directory / "summary.json"
    markdown_path = output_directory / "summary.md"
    write_json_exclusive(json_path, report)
    markdown = (
        "# Phase 4 frozen LLM-search report\n\n"
        f"- Run: `{run_id}`\n"
        f"- Evidence: `{results.get('evidence_class')}`; fake evidence supports lifecycle "
        "validation only, not H1/H2 conclusions\n"
        "- Source: immutable model request/response, SQLite, results, and analysis records only\n"
        f"- Recorded requests/items/evaluations: "
        f"{counts['model_requests']}/{counts['proposal_items']}/{counts['evaluations']}\n"
        f"- Metrics: `{json.dumps(results.get('metrics'), sort_keys=True)}`\n"
        f"- Deterministic summary hash: `{results.get('deterministic_summary_hash')}`\n"
    )
    write_text_exclusive(markdown_path, markdown)
    return json_path, markdown_path


def _create_phase2_report(
    *,
    run_id: str,
    run_directory: Path,
    output_directory: Path,
    configuration_hash: str,
    results: JsonObject,
    event_count: int,
    candidate_count: int,
    evaluation_count: int,
    candidate_record_hashes: tuple[str, ...],
    evaluation_record_hashes: tuple[str, ...],
) -> tuple[Path, Path]:
    """Copy and summarize frozen Phase 2 data without importing live analysis code."""

    analysis_root = run_directory / "analysis"
    try:
        analysis_manifest_raw: object = json.loads(
            (analysis_root / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceError("Phase 2 analysis manifest is missing or invalid") from exc
    if not isinstance(analysis_manifest_raw, dict):
        raise PersistenceError("Phase 2 analysis manifest must be an object")
    files = analysis_manifest_raw.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(name, str) and isinstance(digest, str) for name, digest in files.items()
    ):
        raise PersistenceError("Phase 2 analysis manifest file map is invalid")
    expected_names = {
        "elementary-complexity.json",
        "elementary-complexity.csv",
        "elementary-complexity.svg",
        "collapse-examples.json",
        "gate-report.json",
        "access-ledger.json",
    }
    if set(files) != expected_names:
        raise PersistenceError("Phase 2 analysis manifest has missing or unknown files")
    frozen_contents: dict[str, str] = {}
    for name, expected_hash in files.items():
        content = read_text_artifact(analysis_root / name)
        if sha256_text(content) != expected_hash:
            raise PersistenceError(f"frozen analysis hash mismatch: {name}")
        frozen_contents[name] = content
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, content in frozen_contents.items():
        write_text_exclusive(output_directory / name, content + "\n")
    candidate_hash_values: list[JsonValue] = list(candidate_record_hashes)
    evaluation_hash_values: list[JsonValue] = list(evaluation_record_hashes)
    report: JsonObject = {
        "schema_version": 2,
        "run_id": run_id,
        "source": "frozen-run-artifacts-only",
        "configuration_hash": configuration_hash,
        "recorded_counts": {
            "events": event_count,
            "candidates": candidate_count,
            "evaluations": evaluation_count,
        },
        "analysis_manifest_hash": sha256_text(read_text_artifact(analysis_root / "manifest.json")),
        "analysis_file_hashes": files,
        "frozen_candidate_record_hashes": candidate_hash_values,
        "frozen_evaluation_record_hashes": evaluation_hash_values,
        "results": results,
    }
    json_path = output_directory / "summary.json"
    markdown_path = output_directory / "summary.md"
    write_json_exclusive(json_path, report)
    markdown = (
        "# Phase 2 frozen enumerative report\n\n"
        f"- Run: `{run_id}`\n"
        "- Source: frozen manifest, SQLite candidates/evaluations/events, results, and analysis\n"
        "- Recorded candidates/evaluations/events: "
        f"{candidate_count}/{evaluation_count}/{event_count}\n"
        f"- Metrics: `{json.dumps(results.get('metrics'), sort_keys=True)}`\n"
        "- Complexity data/plot, collapse examples, gate evidence, and access ledger are copied "
        "byte-for-byte from the frozen run analysis.\n"
        f"- Deterministic summary hash: `{results.get('deterministic_summary_hash')}`\n"
    )
    write_text_exclusive(markdown_path, markdown)
    return json_path, markdown_path


def _create_phase3_report(
    *,
    run_id: str,
    run_directory: Path,
    output_directory: Path,
    configuration_hash: str,
    results: JsonObject,
    event_count: int,
    candidate_records: tuple[dict[str, object], ...],
    evaluation_records: tuple[dict[str, object], ...],
    attempt_records: tuple[dict[str, object], ...],
    transition_records: tuple[dict[str, object], ...],
    lineage_records: tuple[dict[str, object], ...],
) -> tuple[Path, Path]:
    """Verify and copy frozen Phase 3 data without live search or evaluation."""

    analysis_root = run_directory / "analysis"
    try:
        manifest_text = read_text_artifact(analysis_root / "manifest.json")
        analysis_manifest: object = json.loads(manifest_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistenceError("Phase 3 analysis manifest is missing or invalid") from exc
    if not isinstance(analysis_manifest, dict):
        raise PersistenceError("Phase 3 analysis manifest must be an object")
    files = analysis_manifest.get("files")
    diagnostic_names = analysis_manifest.get("non_deterministic_diagnostic_files", [])
    expected = {
        "lineage.json",
        "lineage.dot",
        "exact-curve.csv",
        "archive-coverage.csv",
        "operator-diagnostics.json",
        "budget-reconciliation.json",
        "access-ledger.json",
    }
    if (
        not isinstance(files, dict)
        or set(files) != expected
        or not all(
            isinstance(name, str) and isinstance(digest, str) for name, digest in files.items()
        )
    ):
        raise PersistenceError("Phase 3 analysis file map is invalid")
    if diagnostic_names not in ([], ["runtime-diagnostics.json"]):
        raise PersistenceError("Phase 3 diagnostic file declaration is invalid")
    frozen: dict[str, str] = {}
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise PersistenceError("Phase 3 analysis hash map is malformed")
        content = read_text_artifact(analysis_root / name)
        if sha256_text(content) != digest:
            raise PersistenceError(f"frozen Phase 3 analysis hash mismatch: {name}")
        frozen[name] = content
    diagnostics: dict[str, str] = {}
    for name in diagnostic_names:
        if not isinstance(name, str):
            raise PersistenceError("Phase 3 diagnostic file name is invalid")
        diagnostics[name] = read_text_artifact(analysis_root / name)
    diagnostic_values: dict[str, JsonValue] = {
        name: json.loads(content) for name, content in diagnostics.items()
    }
    proposal_hashes: dict[str, JsonValue] = {}
    for attempt in attempt_records:
        name = attempt.get("artifact_name")
        digest = attempt.get("artifact_hash")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise PersistenceError("Phase 3 proposal attempt artifact metadata is invalid")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PersistenceError("Phase 3 proposal artifact path escapes its run")
        if sha256_text(read_text_artifact(run_directory / relative)) != digest:
            raise PersistenceError(f"Phase 3 proposal artifact hash mismatch: {name}")
        proposal_hashes[name] = digest
    record_hashes: JsonObject = {
        "candidates": [sha256_text(canonical_json(record)) for record in candidate_records],
        "evaluations": [sha256_text(canonical_json(record)) for record in evaluation_records],
        "proposal_attempts": [sha256_text(canonical_json(record)) for record in attempt_records],
        "archive_transitions": [
            sha256_text(canonical_json(record)) for record in transition_records
        ],
        "lineage_edges": [sha256_text(canonical_json(record)) for record in lineage_records],
    }
    report: JsonObject = {
        "schema_version": 3,
        "run_id": run_id,
        "source": "frozen-run-artifacts-only",
        "configuration_hash": configuration_hash,
        "recorded_counts": {
            "events": event_count,
            "candidates": len(candidate_records),
            "evaluations": len(evaluation_records),
            "proposal_attempts": len(attempt_records),
            "archive_transitions": len(transition_records),
            "lineage_edges": len(lineage_records),
        },
        "analysis_manifest_hash": sha256_text(manifest_text),
        "analysis_file_hashes": files,
        "diagnostic_file_hashes": {
            name: sha256_text(content) for name, content in diagnostics.items()
        },
        "diagnostics": diagnostic_values,
        "proposal_artifact_hashes": proposal_hashes,
        "frozen_record_hashes": record_hashes,
        "results": results,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, content in frozen.items():
        write_text_exclusive(output_directory / name, content + "\n")
    for name, content in diagnostics.items():
        write_text_exclusive(output_directory / name, content + "\n")
    json_path = output_directory / "summary.json"
    markdown_path = output_directory / "summary.md"
    write_json_exclusive(json_path, report)
    markdown = (
        "# Phase 3 frozen mutation-search report\n\n"
        f"- Run: `{run_id}`\n"
        f"- Condition: `{results.get('condition_id')}`\n"
        "- Source: frozen manifest, SQLite attempts/evaluations/transitions/lineage, results, "
        "and analysis artifacts\n"
        f"- Recorded attempts/evaluations/events: "
        f"{len(attempt_records)}/{len(evaluation_records)}/{event_count}\n"
        f"- Metrics: `{json.dumps(results.get('metrics'), sort_keys=True)}`\n"
        f"- Runtime diagnostics: `{json.dumps(diagnostic_values, sort_keys=True)}`\n"
        f"- Deterministic summary hash: `{results.get('deterministic_summary_hash')}`\n"
    )
    write_text_exclusive(markdown_path, markdown)
    return json_path, markdown_path
