"""Deterministic finalization and analysis artifacts for contextual experience."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import cast

from world_model_search.memory.contextual import (
    DownstreamOutcomeAnnotator,
    ExperienceRecord,
)
from world_model_search.persistence.artifacts import read_text_artifact, write_content_artifact
from world_model_search.persistence.phase4_database import Phase4Database
from world_model_search.serialization import JsonObject, canonical_json, parse_json_object

EXPERIENCE_ANALYSIS_SCHEMA = "contextual-experience-analysis-v1"


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("experience analysis expected an integer")
    return value


def _records(run_directory: Path) -> tuple[ExperienceRecord, ...]:
    raw_directory = run_directory / "experience_v3" / "raw"
    if not raw_directory.is_dir():
        return ()
    return tuple(
        ExperienceRecord.from_value(parse_json_object(read_text_artifact(path)))
        for path in sorted(raw_directory.glob("evaluation-*.json"))
    )


def _csv(records: tuple[ExperienceRecord, ...]) -> str:
    output = io.StringIO(newline="")
    fields = (
        "schema_version",
        "record_id",
        "task_id",
        "source_split",
        "search_seed",
        "sequence_index",
        "parent_candidate_id",
        "child_candidate_id",
        "representation_family",
        "parent_score",
        "child_score",
        "score_delta",
        "edit_classes",
        "archive_outcome",
        "exact_solution",
        "canonical_duplicate",
        "semantic_duplicate",
        "eventually_had_exact_descendant",
        "short_horizon_best_score_gain",
        "sealed_test",
        "training_eligible",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "schema_version": EXPERIENCE_ANALYSIS_SCHEMA,
                "record_id": record.record_id,
                "task_id": record.provenance.task_id,
                "source_split": record.provenance.source_split,
                "search_seed": record.provenance.search_seed,
                "sequence_index": record.provenance.sequence_index,
                "parent_candidate_id": record.provenance.parent_candidate_id,
                "child_candidate_id": record.provenance.child_candidate_id,
                "representation_family": record.context.representation_family,
                "parent_score": record.context.parent_score,
                "child_score": record.immediate_outcome.child_score,
                "score_delta": record.immediate_outcome.score_delta,
                "edit_classes": "|".join(item.value for item in record.action.edit_classes),
                "archive_outcome": record.immediate_outcome.archive_outcome,
                "exact_solution": int(record.immediate_outcome.exact_solution),
                "canonical_duplicate": int(record.immediate_outcome.canonical_duplicate),
                "semantic_duplicate": int(record.immediate_outcome.semantic_duplicate),
                "eventually_had_exact_descendant": int(
                    record.downstream_outcome.eventually_had_exact_descendant
                ),
                "short_horizon_best_score_gain": (
                    record.downstream_outcome.short_horizon_best_score_gain
                ),
                "sealed_test": int(record.memory_metadata.sealed_test),
                "training_eligible": int(record.memory_metadata.training_eligible),
            }
        )
    return output.getvalue().rstrip("\n")


def finalize_experience_artifacts(
    *,
    run_directory: Path,
    database: Phase4Database,
    phase4_results: JsonObject,
    short_horizon_steps: int = 8,
) -> JsonObject:
    """Annotate completed lineages and emit stable JSONL, CSV, and human summaries."""

    raw_records = _records(run_directory)
    candidate_rows = database.candidates()
    candidate_ids = {str(row["candidate_id"]) for row in candidate_rows}
    committed = tuple(
        record for record in raw_records if record.provenance.child_candidate_id in candidate_ids
    )
    parent_ids = {
        str(row["candidate_id"]): tuple(
            str(item) for item in cast(list[object], json.loads(row["parent_ids_json"]))
        )
        for row in candidate_rows
    }
    score_by_candidate: dict[str, int] = {}
    exact_by_candidate: dict[str, bool] = {}
    evaluation_by_candidate: dict[str, int] = {}
    local_accuracy_numerator = 0
    for row in database.evaluations():
        result = parse_json_object(str(row["result_json"]))
        candidate_id = str(row["candidate_id"])
        errors = _integer(result["local_errors"])
        cases = _integer(result["local_cases"])
        score_by_candidate[candidate_id] = cases - errors
        exact_by_candidate[candidate_id] = bool(result["exact"])
        evaluation_by_candidate.setdefault(candidate_id, int(row["evaluation_index"]))
        local_accuracy_numerator += cases - errors
    annotated = DownstreamOutcomeAnnotator().annotate(
        committed,
        parent_ids=parent_ids,
        score_by_candidate=score_by_candidate,
        exact_by_candidate=exact_by_candidate,
        evaluation_by_candidate=evaluation_by_candidate,
        short_horizon_steps=short_horizon_steps,
    )
    root = run_directory / "experience_v3"
    for record in annotated:
        write_content_artifact(
            root / "annotated" / f"evaluation-{record.provenance.sequence_index:05d}.json",
            canonical_json(record.to_value()),
        )
    jsonl = "\n".join(canonical_json(record.to_value()) for record in annotated)
    records_jsonl_hash = write_content_artifact(root / "records.jsonl", jsonl)
    records_csv_hash = write_content_artifact(root / "records.csv", _csv(annotated))
    items = database.items()
    duplicate_count = sum(
        int(row["canonical_duplicate"] or row["semantic_duplicate"])
        for row in items
        if row["outcome"] == "accepted"
    )
    valid_count = sum(row["outcome"] == "accepted" for row in items)
    metrics_value = phase4_results.get("metrics")
    if not isinstance(metrics_value, dict):
        raise ValueError("Phase 4 results have no metrics object")
    metrics = metrics_value
    run_outcomes: JsonObject = {
        "exact_solve": bool(metrics["final_exact_solved"]),
        "exact_solve_auc": metrics["normalized_exact_auc"],
        "local_accuracy_auc": (
            local_accuracy_numerator / (8 * len(database.evaluations()))
            if database.evaluations()
            else 0.0
        ),
        "evaluations_to_first_exact": metrics["calls_to_first_exact"],
        "archive_coverage": metrics["archive_coverage"],
        "duplicate_rate": duplicate_count / valid_count if valid_count else 0.0,
    }
    summary: JsonObject = {
        "schema_version": EXPERIENCE_ANALYSIS_SCHEMA,
        "record_count": len(annotated),
        "improvement_count": sum(record.immediate_outcome.score_delta > 0 for record in annotated),
        "neutral_count": sum(record.immediate_outcome.score_delta == 0 for record in annotated),
        "regression_count": sum(record.immediate_outcome.score_delta < 0 for record in annotated),
        "archive_rejection_count": sum(
            record.immediate_outcome.archive_outcome == "rejected" for record in annotated
        ),
        "duplicate_count": duplicate_count,
        "invalid_proposal_count": sum(row["outcome"] == "rejected" for row in items),
        "orphan_raw_record_count": len(raw_records) - len(committed),
        "short_horizon_steps": short_horizon_steps,
        "run_outcomes": run_outcomes,
        "records_jsonl_hash": records_jsonl_hash,
        "records_csv_hash": records_csv_hash,
    }
    write_content_artifact(root / "summary.json", canonical_json(summary))
    human = "\n".join(
        (
            "# Contextual experience summary",
            "",
            f"- Evaluated parent→child records: {len(annotated)}",
            f"- Improvements / neutral / regressions: {summary['improvement_count']} / "
            f"{summary['neutral_count']} / {summary['regression_count']}",
            f"- Archive rejections: {summary['archive_rejection_count']}",
            f"- Canonical or semantic duplicates: {duplicate_count}",
            f"- Invalid (unevaluated) proposal items: {summary['invalid_proposal_count']}",
            f"- Exact solve: {run_outcomes['exact_solve']}",
            "",
            "Downstream fields describe later descendants; they are not causal labels.",
        )
    )
    write_content_artifact(root / "summary.md", human)
    return summary
