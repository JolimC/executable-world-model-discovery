"""Generate a Phase 0 report using frozen data only."""

from __future__ import annotations

import json
from pathlib import Path

from world_model_search.errors import PersistenceError
from world_model_search.persistence.artifacts import write_json_exclusive, write_text_exclusive
from world_model_search.persistence.database import RunDatabase
from world_model_search.search.loop import load_manifest, validate_run_id
from world_model_search.serialization import JsonObject


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
    if state.status != "completed":
        raise PersistenceError("report command requires a completed run")

    configuration_hash = manifest.get("configuration_hash")
    if not isinstance(configuration_hash, str):
        raise PersistenceError("manifest has no configuration hash")
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
