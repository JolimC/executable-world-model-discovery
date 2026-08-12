from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from world_model_search.config import load_config
from world_model_search.evaluation.phase4_experiment import (
    _child_config,
    _child_id,
    _condition_order,
    load_phase4_experiment_registry,
    run_phase4_experiment,
)
from world_model_search.evaluation.report import create_recorded_report
from world_model_search.model.backends import OpenAIResponsesBackend, ScriptedBackend
from world_model_search.model.cache import ExactResponseCache
from world_model_search.model.types import (
    ModelError,
    ModelErrorCategory,
    ModelRequest,
    ModelResponse,
)
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.persistence.phase4_database import Phase4Database
from world_model_search.replay import replay_run
from world_model_search.search.loop import resume_run, start_run
from world_model_search.search.phase4 import Phase4RunEngine, start_phase4_run
from world_model_search.search.phase4_types import Phase4Condition


def _config(namespace: str, condition: Phase4Condition = Phase4Condition.DIVERSE):
    config = load_config(Path("configs/phase4-fake-smoke.yaml"))
    assert config.cache is not None
    return replace(
        config,
        run=replace(config.run, condition_id=condition.value),
        cache=replace(config.cache, namespace=namespace),
    )


def _install_policy(repository: Path) -> None:
    target = repository / "configs/phase4-price-policy-v1.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path("configs/phase4-price-policy-v1.yaml").read_bytes())


def _one_request_config(namespace: str):
    config = _config(namespace, Phase4Condition.DIRECT)
    assert config.phase4_budget is not None
    return replace(
        config,
        phase4_budget=replace(
            config.phase4_budget,
            model_request_cap=1,
            proposal_item_cap=2,
            oracle_call_cap=9,
        ),
    )


def _one_retry_config(namespace: str):
    config = _one_request_config(namespace)
    assert config.phase4_budget is not None
    return replace(
        config,
        phase4_budget=replace(config.phase4_budget, model_request_cap=2),
    )


def test_phase4_mid_batch_resume_replay_and_frozen_report(
    phase2_repository: Path, monkeypatch: MonkeyPatch
) -> None:
    _install_policy(phase2_repository)
    interrupted = start_run(
        repository_root=phase2_repository,
        config=_config("phase4-resume-a"),
        config_source="phase4-integration",
        run_id="phase4-resume",
        interrupt_after=8,
    )
    assert interrupted.status == "interrupted"
    response_artifacts = tuple((interrupted.run_directory / "responses").glob("*.json"))
    assert len(response_artifacts) == 1
    resumed = resume_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-resume",
    )
    assert resumed.status == "completed" and resumed.completed_steps == 11
    assert len(tuple((resumed.run_directory / "responses").glob("*.json"))) == 2
    independent = start_run(
        repository_root=phase2_repository,
        config=_config("phase4-resume-b"),
        config_source="phase4-independent",
        run_id="phase4-independent",
    )
    assert resumed.event_payload_hashes == independent.event_payload_hashes

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("offline Phase 4 replay/report attempted model/cache/oracle access")

    monkeypatch.setattr(OpenAIResponsesBackend, "dispatch", forbidden)
    monkeypatch.setattr(ExactResponseCache, "get", forbidden)
    replay = replay_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-resume",
    )
    assert replay.proposer_invocations == 0
    assert replay.event_payload_hashes == resumed.event_payload_hashes
    monkeypatch.setattr(ExactDslOracle, "evaluate", forbidden)
    report_json, report_markdown = create_recorded_report(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-resume",
        output_directory=phase2_repository / "phase4-report",
    )
    assert report_json.is_file() and report_markdown.is_file()
    report = json.loads(report_json.read_text())
    assert report["schema_version"] == 4
    assert report["evidence_class"] == "fake"
    assert report["recorded_counts"]["model_requests"] == 2
    assert report["recorded_counts"]["proposal_items"] == 4


def test_phase4_fake_registry_executes_matched_a_b_c_without_model_network(
    phase2_repository: Path,
) -> None:
    _install_policy(phase2_repository)
    registry = load_phase4_experiment_registry(Path("experiments/phase4-fake-smoke.yaml"))
    base = load_config(registry.base_config)
    task_id = registry.task_ids[0]
    seed = registry.search_seeds[0]
    condition = _condition_order(task_id, seed)[0]
    child = _child_config(base, registry, task_id=task_id, seed=seed, condition=condition)
    interrupted = start_phase4_run(
        repository_root=phase2_repository,
        config=child,
        config_source=str(registry.base_config),
        run_id=_child_id(task_id, seed, condition),
        interrupt_after=1,
        allow_live_model=False,
        authority=None,
    )
    assert interrupted.status == "interrupted"
    summary = run_phase4_experiment(
        repository_root=phase2_repository,
        registry_path=Path("experiments/phase4-fake-smoke.yaml"),
        allow_live_model=False,
    )
    assert summary["child_count"] == 3
    assert summary["evidence_class"] == "fake"
    assert summary["scientific_gate"] == "blocked-fake-evidence-only"
    analysis = summary["analysis"]
    assert isinstance(analysis, dict)
    assert set(analysis["contrasts"]) == {
        "H1_B_minus_A",
        "H2_C_minus_B",
        "secondary_C_minus_A",
    }
    root = phase2_repository / "artifacts/phase4-runs/phase4-fake-smoke-v1"
    conditions = {
        json.loads((path / "results.json").read_text())["condition_id"] for path in root.iterdir()
    }
    assert conditions == {condition.value for condition in Phase4Condition}
    report_root = phase2_repository / "artifacts/reports/phase4-fake-smoke-v1"
    artifact = json.loads((report_root / "phase4-artifact.json").read_text())
    assert artifact["source"] == "immutable-child-and-aggregate-records-only"
    assert artifact["scientific_gate"] == "blocked-fake-evidence-only"
    assert artifact["token_reconciliation"]["total_equals_input_plus_output"] is True


def test_phase4_bounded_retry_failure_artifact_and_uncertain_terminal_replay(
    phase2_repository: Path,
) -> None:
    _install_policy(phase2_repository)
    malformed_backend = ScriptedBackend(["{}"])
    retried = start_phase4_run(
        repository_root=phase2_repository,
        config=_config("phase4-retry", Phase4Condition.DIRECT),
        config_source="phase4-retry",
        run_id="phase4-retry",
        interrupt_after=None,
        allow_live_model=False,
        backend=malformed_backend,
    )
    assert retried.status == "completed"
    retry_results = json.loads((retried.run_directory / "results.json").read_text())
    assert retry_results["budget"]["counters"]["model_request_attempts"] == 2
    assert retry_results["budget"]["counters"]["retries"] == 1
    assert retry_results["metrics"]["request_states"] == {
        "completed": 1,
        "schema-failure": 1,
    }

    uncertain_backend = ScriptedBackend(
        [ModelError(ModelErrorCategory.TIMEOUT, retryable=True, usage_uncertain=True)]
    )
    uncertain = start_phase4_run(
        repository_root=phase2_repository,
        config=_config("phase4-uncertain", Phase4Condition.DIRECT),
        config_source="phase4-uncertain",
        run_id="phase4-uncertain",
        interrupt_after=None,
        allow_live_model=False,
        backend=uncertain_backend,
    )
    assert uncertain.status == "usage-uncertain"
    failure_paths = tuple((uncertain.run_directory / "responses").glob("*-failure.json"))
    assert len(failure_paths) == 1
    failure = json.loads(failure_paths[0].read_text())
    assert failure["error"]["category"] == "timeout"
    replay = replay_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-uncertain",
    )
    assert replay.proposer_invocations == 0
    report_json, _ = create_recorded_report(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-uncertain",
        output_directory=phase2_repository / "phase4-uncertain-report",
    )
    assert json.loads(report_json.read_text())["results"]["status"] == "usage-uncertain"


def test_phase4_request_crash_boundaries_fail_closed_without_duplicate_dispatch(
    phase2_repository: Path, monkeypatch: MonkeyPatch
) -> None:
    _install_policy(phase2_repository)
    original_mark_dispatched = Phase4Database.mark_dispatched

    def crash_before_dispatch(self: Phase4Database, request_index: int) -> None:
        del self, request_index
        raise RuntimeError("pre-dispatch crash")

    pending_backend = ScriptedBackend()
    monkeypatch.setattr(Phase4Database, "mark_dispatched", crash_before_dispatch)
    with pytest.raises(RuntimeError, match="pre-dispatch"):
        start_phase4_run(
            repository_root=phase2_repository,
            config=_one_request_config("phase4-boundary-pending"),
            config_source="phase4-boundary-pending",
            run_id="phase4-boundary-pending",
            interrupt_after=None,
            allow_live_model=False,
            backend=pending_backend,
        )
    assert pending_backend.dispatch_count == 0
    monkeypatch.setattr(Phase4Database, "mark_dispatched", original_mark_dispatched)
    pending = resume_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-boundary-pending",
    )
    assert pending.status == "completed" and pending.completed_steps == 9

    original_finalize = Phase4Database.finalize_request

    def crash_before_response_commit(self: Phase4Database, **_kwargs: object) -> None:
        del self
        raise RuntimeError("post-response crash")

    response_backend = ScriptedBackend()
    monkeypatch.setattr(Phase4Database, "finalize_request", crash_before_response_commit)
    with pytest.raises(RuntimeError, match="post-response"):
        start_phase4_run(
            repository_root=phase2_repository,
            config=_one_request_config("phase4-boundary-response"),
            config_source="phase4-boundary-response",
            run_id="phase4-boundary-response",
            interrupt_after=None,
            allow_live_model=False,
            backend=response_backend,
        )
    assert response_backend.dispatch_count == 1
    monkeypatch.setattr(Phase4Database, "finalize_request", original_finalize)
    original_scripted_dispatch = ScriptedBackend.dispatch

    def forbidden_dispatch(self: ScriptedBackend, request: object) -> object:
        del self, request
        raise AssertionError("durable response was dispatched twice")

    monkeypatch.setattr(ScriptedBackend, "dispatch", forbidden_dispatch)
    response_recovered = resume_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-boundary-response",
    )
    assert response_recovered.status == "completed"
    monkeypatch.setattr(ScriptedBackend, "dispatch", original_scripted_dispatch)

    original_process_items = Phase4RunEngine._process_items

    def crash_before_items(self: Phase4RunEngine, **_kwargs: object) -> object:
        del self
        raise RuntimeError("pre-item crash")

    items_backend = ScriptedBackend()
    monkeypatch.setattr(Phase4RunEngine, "_process_items", crash_before_items)
    with pytest.raises(RuntimeError, match="pre-item"):
        start_phase4_run(
            repository_root=phase2_repository,
            config=_one_request_config("phase4-boundary-items"),
            config_source="phase4-boundary-items",
            run_id="phase4-boundary-items",
            interrupt_after=None,
            allow_live_model=False,
            backend=items_backend,
        )
    assert items_backend.dispatch_count == 1
    monkeypatch.setattr(Phase4RunEngine, "_process_items", original_process_items)
    monkeypatch.setattr(ScriptedBackend, "dispatch", forbidden_dispatch)
    items_recovered = resume_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-boundary-items",
    )
    assert items_recovered.status == "completed" and items_recovered.completed_steps == 9
    monkeypatch.setattr(ScriptedBackend, "dispatch", original_scripted_dispatch)

    dispatched_backend = ScriptedBackend()

    def crash_after_dispatch(_request: object) -> object:
        dispatched_backend.dispatch_count += 1
        raise RuntimeError("post-dispatch crash")

    monkeypatch.setattr(dispatched_backend, "dispatch", crash_after_dispatch)
    with pytest.raises(RuntimeError, match="post-dispatch"):
        start_phase4_run(
            repository_root=phase2_repository,
            config=_one_request_config("phase4-boundary-dispatched"),
            config_source="phase4-boundary-dispatched",
            run_id="phase4-boundary-dispatched",
            interrupt_after=None,
            allow_live_model=False,
            backend=dispatched_backend,
        )
    assert dispatched_backend.dispatch_count == 1
    uncertain = resume_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-boundary-dispatched",
    )
    assert uncertain.status == "usage-uncertain"
    results = json.loads((uncertain.run_directory / "results.json").read_text())
    assert results["budget"]["counters"]["physical_provider_calls"] == 1
    assert results["metrics"]["request_states"] == {"usage-uncertain": 1}


@pytest.mark.parametrize(
    ("namespace", "script", "expected_failed_state"),
    (
        ("phase4-recover-malformed", ["{}"], "schema-failure"),
        (
            "phase4-recover-rate-limit",
            [ModelError(ModelErrorCategory.RATE_LIMIT, retryable=True, usage_uncertain=False)],
            "failed",
        ),
    ),
)
def test_phase4_recovery_continues_the_same_bounded_retry_sequence(
    phase2_repository: Path,
    monkeypatch: MonkeyPatch,
    namespace: str,
    script: list[str | ModelError],
    expected_failed_state: str,
) -> None:
    _install_policy(phase2_repository)
    original_finalize = Phase4Database.finalize_request

    def crash_before_failure_commit(self: Phase4Database, **_kwargs: object) -> None:
        del self
        raise RuntimeError("retry-boundary crash")

    monkeypatch.setattr(Phase4Database, "finalize_request", crash_before_failure_commit)
    backend = ScriptedBackend(script)
    with pytest.raises(RuntimeError, match="retry-boundary"):
        start_phase4_run(
            repository_root=phase2_repository,
            config=_one_retry_config(namespace),
            config_source=namespace,
            run_id=namespace,
            interrupt_after=None,
            allow_live_model=False,
            backend=backend,
        )
    assert backend.dispatch_count == 1
    monkeypatch.setattr(Phase4Database, "finalize_request", original_finalize)
    recovered = resume_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id=namespace,
    )
    assert recovered.status == "completed"
    results = json.loads((recovered.run_directory / "results.json").read_text())
    assert results["budget"]["counters"]["logical_model_calls"] == 1
    assert results["budget"]["counters"]["model_request_attempts"] == 2
    assert results["budget"]["counters"]["retries"] == 1
    assert results["metrics"]["request_states"] == {
        "completed": 1,
        expected_failed_state: 1,
    }


def test_phase4_paid_cost_cap_stops_before_dispatch_and_remains_replayable(
    phase2_repository: Path,
) -> None:
    _install_policy(phase2_repository)
    config = load_config(Path("configs/phase4-openai-canary.yaml"))
    assert config.cache is not None and config.phase4_budget is not None
    config = replace(
        config,
        cache=replace(config.cache, namespace="phase4-cost-cap"),
        phase4_budget=replace(config.phase4_budget, child_nano_usd_cap=1),
    )

    class NeverDispatchedOpenAI:
        backend_id = "openai-responses-sdk-v1"
        provider_id = "openai"

        def __init__(self) -> None:
            self.dispatch_count = 0

        def dispatch(self, _request: ModelRequest) -> ModelResponse:
            self.dispatch_count += 1
            raise AssertionError("cost-cap preflight dispatched a provider request")

    backend = NeverDispatchedOpenAI()
    outcome = start_phase4_run(
        repository_root=phase2_repository,
        config=config,
        config_source="phase4-cost-cap",
        run_id="phase4-cost-cap",
        interrupt_after=None,
        allow_live_model=False,
        backend=backend,
    )
    assert outcome.status == "cost-cap-exhausted"
    assert backend.dispatch_count == 0
    results = json.loads((outcome.run_directory / "results.json").read_text())
    assert results["budget"]["counters"]["model_request_attempts"] == 0
    assert results["budget"]["counters"]["physical_provider_calls"] == 0
    replay = replay_run(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-cost-cap",
    )
    assert replay.proposer_invocations == 0 and replay.event_count == 7
    report_json, _ = create_recorded_report(
        repository_root=phase2_repository,
        runs_root=Path("artifacts/runs"),
        run_id="phase4-cost-cap",
        output_directory=phase2_repository / "phase4-cost-cap-report",
    )
    assert json.loads(report_json.read_text())["results"]["status"] == "cost-cap-exhausted"


def test_phase4_dual_budget_cash_cap_stops_before_dispatch(
    phase2_repository: Path,
) -> None:
    dual_policy = phase2_repository / "configs/project-dual-budget-policy-v2.yaml"
    dual_policy.parent.mkdir(parents=True, exist_ok=True)
    dual_policy.write_text(
        Path("configs/project-dual-budget-policy-v2.yaml")
        .read_text()
        .replace("billed_nano_usd: 4650000000", "billed_nano_usd: 99999999999")
    )
    config = load_config(Path("configs/phase4-openai-canary.yaml"))
    assert config.cache is not None and config.phase4_policy is not None
    config = replace(
        config,
        cache=replace(config.cache, namespace="phase4-dual-cash-cap"),
        phase4_policy=replace(
            config.phase4_policy,
            price_policy=Path("configs/project-dual-budget-policy-v2.yaml"),
            ledger=Path("local_state/project-dual-budget-ledger.sqlite3"),
        ),
    )

    class NeverDispatchedOpenAI:
        backend_id = "openai-responses-sdk-v1"
        provider_id = "openai"

        def __init__(self) -> None:
            self.dispatch_count = 0

        def dispatch(self, _request: ModelRequest) -> ModelResponse:
            self.dispatch_count += 1
            raise AssertionError("dual cash-cap preflight dispatched a provider request")

    backend = NeverDispatchedOpenAI()
    outcome = start_phase4_run(
        repository_root=phase2_repository,
        config=config,
        config_source="phase4-dual-cash-cap",
        run_id="phase4-dual-cash-cap",
        interrupt_after=None,
        allow_live_model=False,
        backend=backend,
    )
    assert outcome.status == "cost-cap-exhausted"
    assert backend.dispatch_count == 0
    manifest = json.loads((outcome.run_directory / "manifest.json").read_text())
    assert (
        manifest["budget"]["project_budget_basis"]
        == "reconciled-cash-plus-unreconciled-published-v1"
    )
