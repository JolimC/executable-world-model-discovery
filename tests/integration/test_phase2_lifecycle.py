from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from world_model_search.config import load_config
from world_model_search.evaluation.report import create_recorded_report
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.proposer import enumerative
from world_model_search.replay import replay_run
from world_model_search.search.loop import resume_run, start_run
from world_model_search.tasks import generate_benchmark, generate_phase2_benchmark


def _prepared_repository(repository_root: Path):
    repository_root.mkdir()
    phase1 = load_config(Path("configs/smoke.yaml"))
    generate_benchmark(repository_root, phase1)
    generate_phase2_benchmark(repository_root, phase1)
    return load_config(Path("configs/phase2-smoke.yaml"))


def test_phase2_run_interrupts_resumes_replays_and_reports_frozen_data(
    repository_root: Path, monkeypatch: MonkeyPatch
) -> None:
    config = _prepared_repository(repository_root)
    interrupted = start_run(
        repository_root=repository_root,
        config=config,
        config_source="test",
        run_id="phase2-resume",
        interrupt_after=5,
    )
    assert interrupted.status == "interrupted"
    assert interrupted.completed_steps == 5
    resumed = resume_run(
        repository_root=repository_root,
        runs_root=Path("artifacts/runs"),
        run_id="phase2-resume",
    )
    assert resumed.status == "completed"
    assert resumed.event_payload_hashes[:5] == interrupted.event_payload_hashes
    replay = replay_run(
        repository_root=repository_root,
        runs_root=Path("artifacts/runs"),
        run_id="phase2-resume",
    )
    assert replay.event_payload_hashes == resumed.event_payload_hashes
    assert replay.proposer_invocations == 0
    run_directory = resumed.run_directory
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["manifest_schema_version"] == 3
    assert manifest["versions"]["database_schema"] == 2
    gate = json.loads((run_directory / "analysis" / "gate-report.json").read_text())
    assert gate["enumeration"]["programs_emitted"] == 256
    assert gate["enumeration"]["cost_monotone"] is True
    assert gate["known_form_recovery"]["rule_90"]["discovery_index"] == 10
    complexity = json.loads((run_directory / "analysis" / "elementary-complexity.json").read_text())
    assert len(complexity["records"]) == 256
    assert {record["best_enumerated_status"] for record in complexity["records"]} == {"found"}
    access = json.loads((run_directory / "analysis" / "access-ledger.json").read_text())
    assert access["test_task_oracle_artifacts_accessed"] is False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("report invoked live Phase 2 computation")

    monkeypatch.setattr(enumerative, "enumerate_programs", forbidden)
    monkeypatch.setattr(ExactDslOracle, "evaluate", forbidden)
    report_json, report_markdown = create_recorded_report(
        repository_root=repository_root,
        runs_root=Path("artifacts/runs"),
        run_id="phase2-resume",
        output_directory=repository_root / "report",
    )
    assert report_json.is_file() and report_markdown.is_file()
    report = json.loads(report_json.read_text())
    assert report["source"] == "frozen-run-artifacts-only"
    assert report["recorded_counts"] == {"candidates": 16, "evaluations": 16, "events": 16}


def test_independent_phase2_runs_have_identical_deterministic_evidence(
    repository_root: Path,
) -> None:
    config = _prepared_repository(repository_root)
    first = start_run(
        repository_root=repository_root,
        config=config,
        config_source="first",
        run_id="phase2-a",
    )
    second = start_run(
        repository_root=repository_root,
        config=config,
        config_source="second",
        run_id="phase2-b",
    )
    assert first.event_payload_hashes == second.event_payload_hashes
    first_results = json.loads((first.run_directory / "results.json").read_text())
    second_results = json.loads((second.run_directory / "results.json").read_text())
    assert first_results == second_results
