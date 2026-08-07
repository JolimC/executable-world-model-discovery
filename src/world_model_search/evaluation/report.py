"""Generate a Phase 0 report using frozen data only."""

from __future__ import annotations

import json
from pathlib import Path

from world_model_search.errors import PersistenceError
from world_model_search.persistence.artifacts import (
    read_text_artifact,
    write_json_exclusive,
    write_text_exclusive,
)
from world_model_search.persistence.database import RunDatabase
from world_model_search.persistence.manifest import MANIFEST_SCHEMA_VERSION
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
    with RunDatabase(run_directory / "run.sqlite3", read_only=True) as database:
        state = database.state()
        event_count = len(database.events())
        candidate_count = database.table_count("candidate")
        evaluation_count = database.table_count("evaluation")
        candidate_records = database.candidate_records()
        evaluation_records = database.evaluation_records()
    if state.status != "completed":
        raise PersistenceError("report command requires a completed run")

    configuration_hash = manifest.get("configuration_hash")
    if not isinstance(configuration_hash, str):
        raise PersistenceError("manifest has no configuration hash")
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
