from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from pytest import MonkeyPatch

from world_model_search.config import load_config
from world_model_search.domain.types import SplitLabel
from world_model_search.dsl.versions import PHASE3_OPERATOR_VERSION
from world_model_search.evaluation.report import create_recorded_report
from world_model_search.replay import replay_run
from world_model_search.scheduler.uniform import UniformScheduler
from world_model_search.search import phase3 as phase3_module
from world_model_search.search.loop import resume_run, start_run
from world_model_search.search.operators import (
    AttemptOutcome,
    MutationProposer,
    OperatorAttempt,
    OperatorId,
)
from world_model_search.search.phase3_types import SearchCondition
from world_model_search.serialization import canonical_json, sha256_text


def _config(repository_root: Path):
    config = load_config(Path("configs/phase3-smoke.yaml"))
    assert config.budget is not None
    return replace(
        config,
        budget=replace(config.budget, proposal_attempt_cap=40, oracle_call_cap=12),
    )


def test_phase3_interrupt_resume_replay_report_and_independent_determinism(
    phase2_repository: Path, monkeypatch: MonkeyPatch
) -> None:
    config = _config(phase2_repository)
    first = start_run(
        repository_root=phase2_repository,
        config=config,
        config_source="test",
        run_id="phase3-first",
        interrupt_after=9,
    )
    assert first.status == "interrupted"
    resumed = resume_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase3-first",
    )
    independent = start_run(
        repository_root=phase2_repository,
        config=config,
        config_source="test-independent",
        run_id="phase3-independent",
    )
    assert resumed.event_payload_hashes == independent.event_payload_hashes
    first_results = json.loads((resumed.run_directory / "results.json").read_text())
    second_results = json.loads((independent.run_directory / "results.json").read_text())
    assert first_results == second_results
    assert first_results["budget"]["counters"]["oracle_invocations"] == 12
    assert first_results["budget"]["counters"]["language_model_calls"] == 0

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("replay/report invoked live operator or scheduler selection")

    monkeypatch.setattr(MutationProposer, "propose", forbidden)
    monkeypatch.setattr(UniformScheduler, "select", forbidden)
    replay = replay_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase3-first",
    )
    assert replay.event_payload_hashes == resumed.event_payload_hashes
    monkeypatch.setattr(phase3_module.ExactDslOracle, "evaluate", forbidden)
    report_json, report_markdown = create_recorded_report(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase3-first",
        output_directory=phase2_repository / "phase3-report",
    )
    assert report_json.is_file() and report_markdown.is_file()
    report = json.loads(report_json.read_text())
    assert report["schema_version"] == 3
    assert report["recorded_counts"]["proposal_attempts"] >= 12
    runtime = report["diagnostics"]["runtime-diagnostics.json"]
    assert runtime["timed_attempts"] == report["recorded_counts"]["proposal_attempts"]
    assert runtime["attempt_cpu_ns"] >= runtime["oracle_cpu_ns"] >= 0
    assert runtime["attempt_elapsed_ns"] >= runtime["oracle_elapsed_ns"] >= 0
    assert runtime["language_model_calls"] == runtime["language_model_tokens"] == 0


def _deterministic_run_contract(run_directory: Path) -> object:
    proposal_bytes = tuple(
        path.read_bytes() for path in sorted((run_directory / "proposals").glob("*.json"))
    )
    queries = (
        "SELECT * FROM proposal_attempt ORDER BY attempt_index",
        "SELECT * FROM candidate ORDER BY first_attempt_index",
        "SELECT * FROM archive_transition ORDER BY attempt_index",
        "SELECT * FROM lineage_edge ORDER BY child_candidate_id, parent_order",
        "SELECT state_json, proposal_attempts, oracle_invocations, "
        "proposal_attempt_cap, oracle_call_cap FROM budget_state",
        "SELECT sequence, event_type, logical_cost, payload_json, payload_hash "
        "FROM event ORDER BY sequence",
    )
    with sqlite3.connect(run_directory / "run.sqlite3") as connection:
        rows = tuple(tuple(connection.execute(query).fetchall()) for query in queries)
        evaluation_rows = []
        for attempt, candidate_id, oracle_version, result_json in connection.execute(
            "SELECT attempt_index, candidate_id, oracle_version, result_json "
            "FROM evaluation ORDER BY attempt_index"
        ):
            result = json.loads(result_json)
            result.pop("runtime_ns")
            evaluation_rows.append((attempt, candidate_id, oracle_version, canonical_json(result)))
    analysis_bytes = tuple(
        (path.name, path.read_bytes())
        for path in sorted((run_directory / "analysis").iterdir())
        if path.name != "runtime-diagnostics.json"
    )
    return (
        proposal_bytes,
        rows,
        tuple(evaluation_rows),
        (run_directory / "results.json").read_bytes(),
        analysis_bytes,
    )


def test_phase3_resume_matches_uninterrupted_at_multiple_commit_boundaries(
    phase2_repository: Path,
) -> None:
    base = _config(phase2_repository)
    assert base.budget is not None
    config = replace(
        base,
        budget=replace(base.budget, proposal_attempt_cap=24, oracle_call_cap=10),
    )
    reference = start_run(
        repository_root=phase2_repository,
        config=config,
        config_source="boundary-reference",
        run_id="phase3-boundary-reference",
    )
    reference_contract = _deterministic_run_contract(reference.run_directory)
    for boundary in (1, 7, 9):
        interrupted = start_run(
            repository_root=phase2_repository,
            config=config,
            config_source="boundary-interrupted",
            run_id=f"phase3-boundary-{boundary}",
            interrupt_after=boundary,
        )
        assert interrupted.status == "interrupted"
        resumed = resume_run(
            repository_root=phase2_repository,
            runs_root=Path("artifacts/runs"),
            run_id=f"phase3-boundary-{boundary}",
        )
        assert _deterministic_run_contract(resumed.run_directory) == reference_contract


def test_phase3_full_480_child_development_aggregate_is_reproducible(
    phase2_repository: Path,
) -> None:
    base = load_config(Path("configs/phase3-smoke.yaml"))
    assert base.budget is not None
    assert (base.budget.proposal_attempt_cap, base.budget.oracle_call_cap) == (96, 32)
    task_manifest = json.loads(
        (phase2_repository / "artifacts/phase2-benchmark/manifest.json").read_text()
    )
    development_tasks = tuple(
        sorted(
            item["task_id"]
            for item in task_manifest["tasks"]
            if item["split"] == SplitLabel.DEVELOPMENT.value
        )[:12]
    )
    assert len(development_tasks) == 12
    aggregate_hashes: list[str] = []
    for repetition in range(2):
        child_rows: list[object] = []
        for task_id in development_tasks:
            for seed in range(20):
                for condition in SearchCondition:
                    config = replace(
                        base,
                        run=replace(
                            base.run,
                            task_id=task_id,
                            split=SplitLabel.DEVELOPMENT,
                            seed=7000 + seed,
                            condition_id=condition.value,
                        ),
                    )
                    prefix = "inc" if condition is SearchCondition.INCUMBENT else "div"
                    run = start_run(
                        repository_root=phase2_repository,
                        config=config,
                        config_source="full-development-reproducibility",
                        run_id=(f"p3-full-r{repetition}-{prefix}-{task_id[:8]}-{seed:02d}"),
                    )
                    contract_hash = sha256_text(
                        repr(_deterministic_run_contract(run.run_directory))
                    )
                    result = json.loads((run.run_directory / "results.json").read_text())
                    counters = result["budget"]["counters"]
                    assert counters["proposal_attempts"] == (
                        counters["oracle_invocations"]
                        + counters["invalid_outputs"]
                        + counters["noop_outputs"]
                    )
                    assert counters["oracle_invocations"] == counters["evaluated_candidates"] == 32
                    assert counters["scheduler_selections"] == counters["proposal_attempts"]
                    assert counters["language_model_calls"] == 0
                    assert counters["language_model_tokens"] == 0
                    child_rows.append([task_id, seed, condition.value, contract_hash])
        assert len(child_rows) == 480
        aggregate_hashes.append(sha256_text(canonical_json(child_rows)))
    assert aggregate_hashes[0] == aggregate_hashes[1]


def test_phase3_proposal_cap_stops_cleanly_under_noop_and_rejected_exhaustion(
    phase2_repository: Path, monkeypatch: MonkeyPatch
) -> None:
    base = _config(phase2_repository)
    assert base.budget is not None
    config = replace(
        base,
        budget=replace(base.budget, proposal_attempt_cap=10, oracle_call_cap=10),
    )
    for outcome in (AttemptOutcome.NO_OP, AttemptOutcome.REJECTED):

        def forced_attempt(
            _self: MutationProposer,
            context: object,
            budget: object,
            forced_outcome: AttemptOutcome = outcome,
        ) -> tuple[OperatorAttempt, ...]:
            del context, budget
            return (
                OperatorAttempt(
                    operator_id=OperatorId.LOCAL_MUTATION,
                    operator_version=PHASE3_OPERATOR_VERSION,
                    outcome=forced_outcome,
                    source_ast=None,
                    canonical_ast=None,
                    selected_paths=((),),
                    choices={"forced_exhaustion_test": True},
                    rejection_reason=(
                        "forced-invalid" if forced_outcome is AttemptOutcome.REJECTED else None
                    ),
                    crossover_arity=1,
                ),
            )

        monkeypatch.setattr(MutationProposer, "propose", forced_attempt)
        run = start_run(
            repository_root=phase2_repository,
            config=config,
            config_source="proposal-cap-exhaustion",
            run_id=f"phase3-exhaust-{outcome.value}",
        )
        assert run.status == "completed"
        result = json.loads((run.run_directory / "results.json").read_text())
        counters = result["budget"]["counters"]
        assert counters["proposal_attempts"] == counters["scheduler_selections"] == 10
        assert counters["operator_attempts"] == 3
        assert counters["oracle_invocations"] == counters["evaluated_candidates"] == 7
        assert counters["noop_outputs"] == (3 if outcome is AttemptOutcome.NO_OP else 0)
        assert counters["invalid_outputs"] == (3 if outcome is AttemptOutcome.REJECTED else 0)
        assert result["budget"]["remaining"] == {"oracle_calls": 3, "proposal_attempts": 0}
        runtime = json.loads(
            (run.run_directory / "analysis" / "runtime-diagnostics.json").read_text()
        )
        assert runtime["timed_attempts"] == 10
        assert runtime["language_model_tokens"] == 0
