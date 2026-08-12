from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from world_model_search.cli import main
from world_model_search.config import config_from_mapping, load_config
from world_model_search.domain.types import CandidateSummary, ProposalRole
from world_model_search.dsl.ast import AstLimits, At
from world_model_search.errors import BudgetExhaustedError, ConfigurationError, PersistenceError
from world_model_search.evaluation.phase4_experiment import (
    analyze_phase4_rows,
    load_phase4_experiment_registry,
    phase4_dry_run,
)
from world_model_search.model.backends import LiveOptIn, OpenAIResponsesBackend, ScriptedBackend
from world_model_search.model.cache import ExactResponseCache
from world_model_search.model.ledger import ProjectLedger, rebuild_project_ledger
from world_model_search.model.policy import load_price_policy
from world_model_search.model.prompts import ParentScoreFeedback
from world_model_search.model.schema import BatchEnvelopeError, parse_candidate_batch
from world_model_search.model.types import ModelDispatchError, ModelRequest, ModelUsage
from world_model_search.proposer.llm import LLMProposer
from world_model_search.search.phase4_types import Phase4BudgetState, Phase4Condition
from world_model_search.serialization import canonical_json, sha256_text
from world_model_search.tasks import load_public_task


def _proposer(phase2_repository: Path, cache: ExactResponseCache | None = None) -> LLMProposer:
    config = load_config(Path("configs/phase4-fake-smoke.yaml"))
    assert config.model is not None and config.dsl is not None
    return LLMProposer(
        backend=ScriptedBackend(),
        resolved_model=config.model.resolved_model,
        endpoint=config.model.endpoint,
        service_tier=config.model.service_tier,
        settings=config.model.request_settings(),
        limits=AstLimits(config.dsl.max_depth, config.dsl.max_nodes, config.dsl.max_cases),
        allowed_macros=frozenset(config.dsl.allowed_macros),
        cache=cache,
    )


def test_strict_batch_request_identity_cache_and_prompt_leakage(
    phase2_repository: Path, tmp_path: Path
) -> None:
    task = load_public_task(
        phase2_repository / "artifacts/phase2-benchmark", "d737b0ee219de6a676c139d1"
    ).public_view()
    cache = ExactResponseCache(tmp_path / "cache", "unit")
    proposer = _proposer(phase2_repository, cache)
    request = proposer.build_request(task=task, role=ProposalRole.EXPLOIT, batch_size=3)
    assert request.prompt_template == "direct"
    assert "selected_parent" not in request.rendered_input
    assert "semantic_hash" not in request.rendered_input
    assert "rollout" not in request.rendered_input.lower()
    assert "OPENAI_API_KEY" not in canonical_json(request.identity_value())
    response = proposer.dispatch(request)
    parsed = proposer.parse_response(request, response)
    assert [item.ordinal for item in parsed.batch.items] == [0, 1, 2]
    assert all(item.accepted for item in parsed.batch.items)
    cached = cache.get(request)
    assert cached is not None
    assert cached.deterministic_value() == response.deterministic_value()
    assert proposer.dispatch(request).deterministic_value() == response.deterministic_value()
    assert proposer.last_cache_hit is True
    changed = replace(request, requested_batch_size=2)
    assert changed.request_hash != request.request_hash
    malformed_identity = request.identity_value()
    malformed_identity["unknown"] = True
    with pytest.raises(ValueError, match="identity"):
        ModelRequest.from_identity_value(malformed_identity)

    path = cache.root / cache.namespace / f"{request.request_hash}.json"
    raw = json.loads(path.read_text())
    raw["content_hash"] = "0" * 64
    path.write_text(canonical_json(raw), encoding="utf-8")
    with pytest.raises(PersistenceError, match="hash or identity"):
        cache.get(request)
    raw = json.loads(path.read_text())
    raw["content"]["request_hash"] = "b" * 64
    raw["content_hash"] = sha256_text(canonical_json(raw["content"]))
    path.write_text(canonical_json(raw), encoding="utf-8")
    with pytest.raises(PersistenceError, match="fields are corrupt"):
        cache.get(request)


def test_batch_envelope_rejects_root_failures_and_keeps_item_failures() -> None:
    limits = AstLimits(8, 63, 8)
    valid = {
        "candidate_schema_version": 1,
        "dsl_version": "binary-ca-radius1-dsl-v1",
        "ast": {"op": "At", "offset": 0},
    }
    invalid = {
        "candidate_schema_version": 1,
        "dsl_version": "binary-ca-radius1-dsl-v1",
        "ast": {"op": "At", "offset": 2},
    }
    raw = canonical_json(
        {"batch_schema_version": 1, "role": "exploit", "candidates": [valid, invalid]}
    )
    batch = parse_candidate_batch(
        raw,
        expected_role=ProposalRole.EXPLOIT,
        requested_batch_size=2,
        limits=limits,
        allowed_macros=frozenset({"Parity", "Majority"}),
    )
    assert batch.items[0].accepted and not batch.items[1].accepted
    failures = (
        "```json\n" + raw + "\n```",
        raw + " trailing",
        '{"batch_schema_version":1,"batch_schema_version":1,"role":"exploit",'
        f'"candidates":[{canonical_json(valid)}]}}',
        canonical_json(
            {"batch_schema_version": 1, "role": "simplify", "candidates": [valid, valid]}
        ),
        canonical_json({"batch_schema_version": 1, "role": "exploit", "candidates": []}),
        '{"batch_schema_version":1,"role":"exploit","candidates":[NaN,NaN]}',
    )
    for failure in failures:
        with pytest.raises(BatchEnvelopeError):
            parse_candidate_batch(
                failure,
                expected_role=ProposalRole.EXPLOIT,
                requested_batch_size=2,
                limits=limits,
                allowed_macros=frozenset({"Parity", "Majority"}),
            )


def test_iterative_prompt_exposes_only_bounded_parent_score(phase2_repository: Path) -> None:
    task = load_public_task(
        phase2_repository / "artifacts/phase2-benchmark", "d737b0ee219de6a676c139d1"
    ).public_view()
    request = _proposer(phase2_repository).build_request(
        task=task,
        role=ProposalRole.EXPLOIT,
        batch_size=1,
        parent=CandidateSummary("a" * 64, At(0)),
        feedback=ParentScoreFeedback("a" * 64, True, True, 2, 8, False, 5, 7, 12),
    )
    prompt = json.loads(request.rendered_input)
    assert request.prompt_template == "iterative"
    assert set(prompt["selected_parent_score"]) == {
        "feedback_schema_version",
        "candidate_id",
        "type_valid",
        "total",
        "local_errors",
        "local_cases",
        "exact",
        "ast_bits",
        "residual_bits",
        "two_part_bits",
    }
    forbidden = {"runtime_ns", "rollout_pass", "semantic_hash", "counterexample"}
    assert not forbidden.intersection(prompt["selected_parent_score"])


def test_usage_price_budget_and_ledger_reconcile_exactly(tmp_path: Path) -> None:
    policy = load_price_policy(Path("configs/phase4-price-policy-v1.yaml"))
    usage = ModelUsage(100, 40, 10, 3, 110)
    assert policy.price.cost(usage) == 60 * 250 + 40 * 25 + 10 * 2000
    budget = Phase4BudgetState(2, 1000, 100, 1100, 4, 11, 150_000_000)
    budget.preflight(input_token_bound=100, max_output_tokens=20)
    updated = budget.updated(
        logical_model_calls=1,
        model_request_attempts=1,
        physical_provider_calls=1,
        input_tokens=100,
        cached_input_tokens=40,
        output_tokens=10,
        reasoning_tokens=3,
        total_tokens=110,
    )
    assert Phase4BudgetState.from_value(updated.to_value()) == updated
    malformed = updated.to_value()
    assert isinstance(malformed["counters"], dict)
    malformed["counters"]["unknown"] = 1
    with pytest.raises(ValueError, match="field set"):
        Phase4BudgetState.from_value(malformed)
    with pytest.raises(ValueError, match="reasoning"):
        ModelUsage(1, 0, 1, 2, 2)

    ledger_path = tmp_path / "ledger.sqlite3"
    with ProjectLedger(ledger_path, policy) as ledger:
        ledger.reserve(
            reservation_id="r1",
            run_id="child",
            stage="canary",
            request_hash="a" * 64,
            amount_nano_usd=1_000_000,
            child_cap_nano_usd=150_000_000,
        )
        assert ledger.balance().active_reserved_nano_usd == 1_000_000
        assert ledger.reconcile(
            reservation_id="r1", actual_nano_usd=250_000, usage_record={"usage": 1}
        ) == (250_000, 750_000)
        assert ledger.reconcile(
            reservation_id="r1", actual_nano_usd=250_000, usage_record={"usage": 1}
        ) == (250_000, 750_000)
        with pytest.raises(PersistenceError, match="record diverged"):
            ledger.reconcile(
                reservation_id="r1", actual_nano_usd=250_000, usage_record={"usage": 2}
            )
        assert ledger.balance().actual_nano_usd == 250_000
        ledger.reserve(
            reservation_id="r2",
            run_id="child",
            stage="canary",
            request_hash="c" * 64,
            amount_nano_usd=500_000,
            child_cap_nano_usd=150_000_000,
        )
        assert ledger.mark_uncertain(reservation_id="r2", failure_record={"failure": 1}) == 500_000
        assert ledger.mark_uncertain(reservation_id="r2", failure_record={"failure": 1}) == 500_000
        with pytest.raises(PersistenceError, match="record diverged"):
            ledger.mark_uncertain(reservation_id="r2", failure_record={"failure": 2})
        with pytest.raises(PersistenceError, match="request reservation"):
            ledger.reserve(
                reservation_id="too-large",
                run_id="child",
                stage="canary",
                request_hash="b" * 64,
                amount_nano_usd=10_000_001,
                child_cap_nano_usd=150_000_000,
            )
    rebuilt = rebuild_project_ledger(
        repository_root=tmp_path / "empty-repository",
        path=tmp_path / "rebuilt.sqlite3",
        policy=policy,
    )
    assert rebuilt["records"] == rebuilt["paid_artifacts"] == 0


def test_project_ledger_serializes_concurrent_reservations(tmp_path: Path) -> None:
    policy = load_price_policy(Path("configs/phase4-price-policy-v1.yaml"))
    ledger_path = tmp_path / "concurrent-ledger.sqlite3"
    with ProjectLedger(ledger_path, policy):
        pass

    def reserve(index: int) -> None:
        with ProjectLedger(ledger_path, policy) as ledger:
            ledger.reserve(
                reservation_id=f"concurrent-{index}",
                run_id="concurrent-child",
                stage="canary",
                request_hash=f"{index:064x}",
                amount_nano_usd=1_000_000,
                child_cap_nano_usd=150_000_000,
            )

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(reserve, range(8)))
    with ProjectLedger(ledger_path, policy) as ledger:
        assert ledger.balance().active_reserved_nano_usd == 8_000_000
        assert len(ledger.records()) == 8


def test_dual_budget_preserves_published_cost_and_reconciles_cash(tmp_path: Path) -> None:
    legacy = load_price_policy(Path("configs/phase4-price-policy-v1.yaml"))
    assert legacy.content_hash == "120ca1d0cb66d23230ff8267d4c0eb492421e8de55dc4e1e97950e5cd7fc93fa"
    policy = load_price_policy(Path("configs/project-dual-budget-policy-v2.yaml"))
    assert policy.uses_reconciled_cash_budget is True
    assert policy.opening_balance_nano_usd == 6_526_807_550
    assert policy.cash_budget is not None
    assert policy.cash_budget.opening_reconciled_cash_nano_usd == 4_650_000_000

    with ProjectLedger(tmp_path / "dual.sqlite3", policy) as ledger:
        initial = ledger.cash_balance()
        assert initial.reconciled_cash_nano_usd == 4_650_000_000
        assert initial.covered_published_nano_usd == 6_526_807_550
        assert initial.cash_upper_bound_nano_usd == 4_650_000_000
        ledger.reserve(
            reservation_id="dual-r1",
            run_id="dual-child",
            stage="canary",
            request_hash="d" * 64,
            amount_nano_usd=1_000_000,
            child_cap_nano_usd=150_000_000,
        )
        assert ledger.cash_balance().active_reserved_nano_usd == 1_000_000
        ledger.reconcile(
            reservation_id="dual-r1",
            actual_nano_usd=250_000,
            usage_record={"usage": "dual-r1"},
        )
        before_checkpoint = ledger.cash_balance()
        assert before_checkpoint.unreconciled_actual_nano_usd == 250_000
        assert before_checkpoint.cash_upper_bound_nano_usd == 4_650_250_000
        result = ledger.append_cash_checkpoint(
            cumulative_billed_nano_usd=4_660_000_000,
            covered_reservation_sequence=None,
            observed_at="2026-08-12T09:30:00-05:00",
            scope="provider project through dual-r1",
            source="user-reported-provider-dashboard",
            verification="user-reported-unverified",
        )
        duplicate = ledger.append_cash_checkpoint(
            cumulative_billed_nano_usd=4_660_000_000,
            covered_reservation_sequence=1,
            observed_at="2026-08-12T09:30:00-05:00",
            scope="provider project through dual-r1",
            source="user-reported-provider-dashboard",
            verification="user-reported-unverified",
        )
        assert duplicate["checkpoint"] == result["checkpoint"]
        cash = result["cash_budget"]
        assert isinstance(cash, dict)
        assert cash["reconciled_cash_nano_usd"] == 4_660_000_000
        assert cash["unreconciled_actual_nano_usd"] == 0
        assert cash["covered_published_nano_usd"] == 6_527_057_550
        assert ledger.status()["reconcilable_through_sequence"] == 1
        assert ledger.status()["cash_checkpoint_count_including_opening"] == 2
        assert len(ledger.cash_checkpoints()) == 2
        assert len(ledger.records()) == 1


def test_dual_budget_cash_cap_and_checkpoint_boundaries_fail_closed(tmp_path: Path) -> None:
    policy = load_price_policy(Path("configs/project-dual-budget-policy-v2.yaml"))
    assert policy.cash_budget is not None
    near_cap_cash = replace(
        policy.cash_budget,
        opening_reconciled_cash_nano_usd=99_999_500_000,
    )
    near_cap = replace(policy, cash_budget=near_cap_cash)
    with (
        ProjectLedger(tmp_path / "near-cap.sqlite3", near_cap) as ledger,
        pytest.raises(BudgetExhaustedError, match="personal cash ceiling"),
    ):
        ledger.reserve(
            reservation_id="blocked",
            run_id="dual-child",
            stage="canary",
            request_hash="e" * 64,
            amount_nano_usd=1_000_000,
            child_cap_nano_usd=150_000_000,
        )

    promoted_opening = 120_000_000_000
    promoted_cash = replace(
        policy.cash_budget,
        opening_covered_published_nano_usd=promoted_opening,
    )
    promoted = replace(
        policy,
        opening_balance_nano_usd=promoted_opening,
        cash_budget=promoted_cash,
    )
    with ProjectLedger(tmp_path / "promoted.sqlite3", promoted) as ledger:
        ledger.reserve(
            reservation_id="allowed-over-published-100",
            run_id="dual-child",
            stage="canary",
            request_hash="f" * 64,
            amount_nano_usd=1_000_000,
            child_cap_nano_usd=150_000_000,
        )
        with pytest.raises(PersistenceError, match="active or uncertain"):
            ledger.append_cash_checkpoint(
                cumulative_billed_nano_usd=4_650_000_000,
                covered_reservation_sequence=1,
                observed_at="2026-08-12",
                scope="invalid active coverage",
                source="user-reported-provider-dashboard",
                verification="user-reported-unverified",
            )


def test_dual_budget_cli_status_and_cash_reconciliation(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "project-dual-budget-policy-v2.yaml").write_bytes(
        Path("configs/project-dual-budget-policy-v2.yaml").read_bytes()
    )
    monkeypatch.chdir(tmp_path)
    assert main(["ledger", "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["cash_budget"]["reconciled_cash_nano_usd"] == 4_650_000_000
    assert status["published_rate_balance"]["opening_nano_usd"] == 6_526_807_550
    assert (
        main(
            [
                "ledger",
                "reconcile-cash",
                "--billed-usd",
                "4.66",
                "--observed-at",
                "2026-08-12T10:15:00-05:00",
                "--scope",
                "provider project through current finalized usage",
                "--through-current-finalized",
            ]
        )
        == 0
    )
    reconciled = json.loads(capsys.readouterr().out)
    assert reconciled["cash_budget"]["reconciled_cash_nano_usd"] == 4_660_000_000
    assert reconciled["checkpoint"]["verification"] == "user-reported-unverified"
    assert main(["ledger", "cash-history"]) == 0
    history = json.loads(capsys.readouterr().out)
    assert len(history["checkpoints"]) == 2


def test_dual_budget_serializes_concurrent_cash_reservations(tmp_path: Path) -> None:
    policy = load_price_policy(Path("configs/project-dual-budget-policy-v2.yaml"))
    assert policy.cash_budget is not None
    limited_cash = replace(
        policy.cash_budget,
        opening_reconciled_cash_nano_usd=99_996_000_000,
    )
    limited = replace(policy, cash_budget=limited_cash)
    ledger_path = tmp_path / "dual-concurrent.sqlite3"
    with ProjectLedger(ledger_path, limited):
        pass

    def reserve(index: int) -> bool:
        try:
            with ProjectLedger(ledger_path, limited) as ledger:
                ledger.reserve(
                    reservation_id=f"dual-concurrent-{index}",
                    run_id="dual-concurrent-child",
                    stage="canary",
                    request_hash=f"{index:064x}",
                    amount_nano_usd=1_000_000,
                    child_cap_nano_usd=150_000_000,
                )
        except BudgetExhaustedError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = tuple(executor.map(reserve, range(8)))
    assert outcomes.count(True) == outcomes.count(False) == 4
    with ProjectLedger(ledger_path, limited) as ledger:
        assert ledger.cash_balance().active_reserved_nano_usd == 4_000_000


def test_dual_policy_cannot_mutate_or_open_the_legacy_ledger(tmp_path: Path) -> None:
    legacy = load_price_policy(Path("configs/phase4-price-policy-v1.yaml"))
    dual = load_price_policy(Path("configs/project-dual-budget-policy-v2.yaml"))
    ledger_path = tmp_path / "legacy.sqlite3"
    with ProjectLedger(ledger_path, legacy):
        pass
    with pytest.raises(PersistenceError, match="schema or price-policy identity mismatch"):
        ProjectLedger(ledger_path, dual)
    connection = sqlite3.connect(ledger_path)
    try:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()
    assert "cash_checkpoint" not in tables


def test_dual_budget_rejects_checkpoint_column_tampering(tmp_path: Path) -> None:
    policy = load_price_policy(Path("configs/project-dual-budget-policy-v2.yaml"))
    ledger_path = tmp_path / "tampered-checkpoint.sqlite3"
    with ProjectLedger(ledger_path, policy) as ledger:
        ledger.append_cash_checkpoint(
            cumulative_billed_nano_usd=4_660_000_000,
            covered_reservation_sequence=None,
            observed_at="2026-08-12",
            scope="provider project through opening usage",
            source="user-reported-provider-dashboard",
            verification="user-reported-unverified",
        )
    connection = sqlite3.connect(ledger_path)
    try:
        with connection:
            connection.execute("UPDATE cash_checkpoint SET cumulative_billed_nano_usd=1")
    finally:
        connection.close()
    with (
        ProjectLedger(ledger_path, policy) as ledger,
        pytest.raises(PersistenceError, match="indexed fields diverge"),
    ):
        ledger.cash_balance()


def test_live_adapter_requires_both_opt_ins_and_never_reads_key_early(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.delenv("WMS_ALLOW_LIVE_MODEL", raising=False)
    with pytest.raises(ModelDispatchError):
        OpenAIResponsesBackend(opt_in=LiveOptIn.resolve(cli_allowed=True))
    monkeypatch.setenv("WMS_ALLOW_LIVE_MODEL", "1")
    with pytest.raises(ModelDispatchError):
        OpenAIResponsesBackend(opt_in=LiveOptIn.resolve(cli_allowed=False))


def test_phase4_registry_dry_run_and_null_negative_analysis(
    phase2_repository: Path,
) -> None:
    fake = load_phase4_experiment_registry(Path("experiments/phase4-fake-smoke.yaml"))
    assert len(fake.task_ids) == len(fake.search_seeds) == 1
    pilot = load_phase4_experiment_registry(Path("experiments/phase4-primary-pilot.yaml"))
    assert len(pilot.task_ids) == 10
    assert len(pilot.search_seeds) == 2
    assert pilot.prerequisite_canary is not None
    policy_target = phase2_repository / "configs/phase4-price-policy-v1.yaml"
    policy_target.parent.mkdir(parents=True, exist_ok=True)
    policy_target.write_bytes(Path("configs/phase4-price-policy-v1.yaml").read_bytes())
    forecast = phase4_dry_run(repository_root=phase2_repository, registry=pilot)
    assert forecast["network_calls"] == forecast["hidden_oracle_accesses"] == 0
    assert forecast["status"] == "blocked"
    assert forecast["children"] == 60
    assert forecast["planned_work"] == {
        "logical_model_calls_per_child": 63,
        "logical_model_calls_all_children": 3780,
        "maximum_physical_attempts_all_children": 7560,
        "proposal_items_all_children": 14940,
        "oracle_calls_all_children": 15360,
    }
    assert forecast["prerequisite_canary"] == {
        "required": True,
        "status": "missing-or-unreadable",
        "run_id": "PHASE4-LIVE-CANARY-V2",
    }
    dual_policy_target = phase2_repository / "configs/project-dual-budget-policy-v2.yaml"
    dual_policy_target.write_bytes(Path("configs/project-dual-budget-policy-v2.yaml").read_bytes())
    dual_config = phase2_repository / "configs/phase4-openai-pilot-dual-test.yaml"
    dual_config.write_text(
        Path("configs/phase4-openai-pilot.yaml")
        .read_text()
        .replace(
            "configs/phase4-price-policy-v1.yaml",
            "configs/project-dual-budget-policy-v2.yaml",
        )
        .replace(
            "local_state/project-cost-ledger.sqlite3",
            "local_state/project-dual-budget-dry-run-test.sqlite3",
        )
    )
    dual_forecast = phase4_dry_run(
        repository_root=phase2_repository,
        registry=replace(pilot, base_config=dual_config),
    )
    dual_ledger = dual_forecast["ledger"]
    assert isinstance(dual_ledger, dict)
    assert dual_ledger["state"] == "not-created"
    dual_cash = dual_ledger["cash_budget"]
    assert isinstance(dual_cash, dict)
    assert dual_cash["reconciled_cash_nano_usd"] == 4_650_000_000
    worst_case = dual_forecast["worst_case_nano_usd"]
    assert isinstance(worst_case, dict)
    assert (
        worst_case["project_enforcement_basis"] == "reconciled-cash-plus-unreconciled-published-v1"
    )
    rows = []
    values = {
        Phase4Condition.DIRECT.value: 0.4,
        Phase4Condition.INCUMBENT.value: 0.3,
        Phase4Condition.DIVERSE.value: 0.3,
    }
    for task in ("task-a", "task-b"):
        for seed in (1, 2):
            for condition, value in values.items():
                rows.append(
                    {
                        "task_id": task,
                        "search_seed": seed,
                        "condition_id": condition,
                        "metrics": {"normalized_exact_auc": value},
                    }
                )
    analysis = analyze_phase4_rows(rows, bootstrap_seed=7, bootstrap_replicates=1000)
    contrasts = analysis["contrasts"]
    assert isinstance(contrasts, dict)
    assert contrasts["H1_B_minus_A"]["point_estimate"] < 0
    assert contrasts["H2_C_minus_B"]["point_estimate"] == pytest.approx(0)
    assert contrasts["H1_B_minus_A"]["holm_reject_0_05"] is True
    assert contrasts["H1_B_minus_A"]["superiority_established"] is False
    assert contrasts["H2_C_minus_B"]["holm_adjusted_p_value"] == pytest.approx(1.0)

    positive_rows = []
    positive = {
        Phase4Condition.DIRECT.value: 0.1,
        Phase4Condition.INCUMBENT.value: 0.3,
        Phase4Condition.DIVERSE.value: 0.5,
    }
    for task in ("task-a", "task-b", "task-c"):
        for seed in (1, 2):
            for condition, value in positive.items():
                positive_rows.append(
                    {
                        "task_id": task,
                        "search_seed": seed,
                        "condition_id": condition,
                        "metrics": {"normalized_exact_auc": value},
                    }
                )
    positive_analysis = analyze_phase4_rows(
        positive_rows, bootstrap_seed=8, bootstrap_replicates=1000
    )
    positive_contrasts = positive_analysis["contrasts"]
    assert isinstance(positive_contrasts, dict)
    assert positive_contrasts["H1_B_minus_A"]["superiority_established"] is True
    assert positive_contrasts["H2_C_minus_B"]["superiority_established"] is True


def test_phase4_config_remains_strict() -> None:
    config = load_config(Path("configs/phase4-fake-smoke.yaml"))
    raw = config.to_mapping()
    raw["unexpected"] = True
    with pytest.raises(ConfigurationError):
        config_from_mapping(raw)
