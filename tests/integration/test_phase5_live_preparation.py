from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from world_model_search.errors import ConfigurationError, ReplayError
from world_model_search.evaluation.phase5_live import (
    load_phase5_live_experiment,
    phase5_live_dry_run,
    replay_phase5_live_experiment,
    run_phase5_live_experiment,
)
from world_model_search.model.ledger import ProjectLedger
from world_model_search.model.policy import load_price_policy
from world_model_search.model.types import (
    ModelDispatchRequest,
    ModelResponse,
    ModelUsage,
)
from world_model_search.serialization import canonical_json


class _CompatibleFakeOpenAI:
    backend_id = "openai-responses-sdk-v1"
    provider_id = "openai"

    def __init__(self) -> None:
        self.dispatch_count = 0

    def dispatch(self, request: ModelDispatchRequest) -> ModelResponse:
        self.dispatch_count += 1
        raw = canonical_json(
            {
                "batch_schema_version": 1,
                "role": "transfer",
                "candidates": [
                    {
                        "candidate_schema_version": 1,
                        "dsl_version": "binary-ca-radius1-dsl-v1",
                        "ast": {"op": "Const", "value": 0},
                    }
                ],
            }
        )
        usage = ModelUsage(
            request.conservative_input_token_bound,
            0,
            100,
            0,
            request.conservative_input_token_bound + 100,
        )
        return ModelResponse(
            request_hash=request.request_hash,
            raw_text=raw,
            usage=usage,
            provider_request_id=f"fake-phase5-{self.dispatch_count}",
            resolved_model=request.resolved_model,
            service_tier=request.service_tier,
            system_fingerprint="fake-phase5-compatible-v1",
            provider_latency_ns=0,
        )


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    sources = (
        Path("configs/project-dual-budget-policy-v2.yaml"),
        Path("configs/phase5-exposure-policy-v2.yaml"),
        Path("configs/phase5-transfer-split-v1.yaml"),
        Path("experiments/phase5-canary.yaml"),
        Path("experiments/phase5-canary.pending.yaml"),
        Path("experiments/phase5-development.yaml"),
        Path("experiments/phase5-development.pending.yaml"),
        Path("experiments/phase5-freeze/memory-snapshot.json"),
        Path("experiments/phase5-freeze/primitive-registry.json"),
    )
    for source in sources:
        destination = repository / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (repository / "local_state").mkdir()
    return repository


def _authorize(path: Path, evidence: str) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["status"] = "authorized"
    raw["authorization"] = {
        "model_calls": True,
        "oracle_access": True,
        "user_reviewed_exposure_policy": True,
        "user_authorized_live_run": True,
    }
    raw["authorization_evidence"] = evidence
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def test_frozen_canary_and_development_preflights_are_no_cost(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    canary = phase5_live_dry_run(
        repository_root=repository,
        registry_path=repository / "experiments/phase5-canary.yaml",
        authority_path=repository / "experiments/phase5-canary.pending.yaml",
    )
    development = phase5_live_dry_run(
        repository_root=repository,
        registry_path=repository / "experiments/phase5-development.yaml",
        authority_path=repository / "experiments/phase5-development.pending.yaml",
    )
    assert canary["provider_calls"] == canary["oracle_accesses"] == 0
    assert canary["total_request_cap"] == 1
    assert canary["aggregate_nano_usd_max"] == 7_096_000
    assert canary["live_authorized"] is False
    assert development["provider_calls"] == development["oracle_accesses"] == 0
    assert development["child_count"] == 16
    assert development["total_request_cap"] == 256
    assert development["aggregate_nano_usd_max"] == 1_816_576_000
    observed_bound = development["maximum_observed_request_identity_input_bound"]
    assert isinstance(observed_bound, int) and observed_bound <= 12_000


def test_pending_authority_refuses_before_backend_or_oracle(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    backend = _CompatibleFakeOpenAI()
    with pytest.raises(ConfigurationError, match="pending explicit user authorization"):
        run_phase5_live_experiment(
            repository_root=repository,
            registry_path=repository / "experiments/phase5-canary.yaml",
            authority_path=repository / "experiments/phase5-canary.pending.yaml",
            allow_live_model=True,
            backend=backend,
        )
    assert backend.dispatch_count == 0


def test_authorized_fake_canary_runs_and_replays_without_real_provider(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    authority = repository / "experiments/phase5-canary.pending.yaml"
    _authorize(authority, "test-only-fake-backend-authorization")
    backend = _CompatibleFakeOpenAI()
    summary = run_phase5_live_experiment(
        repository_root=repository,
        registry_path=repository / "experiments/phase5-canary.yaml",
        authority_path=authority,
        allow_live_model=True,
        backend=backend,
    )
    assert summary["status"] == "passed-live-training-canary"
    assert summary["physical_provider_calls"] == 1
    assert summary["valid_candidates"] == 1
    assert summary["sealed_test_accesses"] == 0
    assert backend.dispatch_count == 1
    replay = replay_phase5_live_experiment(
        repository_root=repository,
        registry_path=repository / "experiments/phase5-canary.yaml",
    )
    assert replay["status"] == "verified-provider-disabled"
    assert replay["provider_calls"] == replay["oracle_accesses"] == 0
    result = next(
        (repository / "artifacts/phase5/live-canary-v1/children").glob(
            "*/results/request-00000.json"
        )
    )
    result.write_text(result.read_text().replace('"valid_candidates":1', '"valid_candidates":0'))
    with pytest.raises(ReplayError, match="request/response/result identity"):
        replay_phase5_live_experiment(
            repository_root=repository,
            registry_path=repository / "experiments/phase5-canary.yaml",
        )


def test_development_requires_recorded_successful_canary_before_dispatch(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    authority = repository / "experiments/phase5-development.pending.yaml"
    _authorize(authority, "test-only-development-prerequisite-check")
    backend = _CompatibleFakeOpenAI()
    with pytest.raises(ConfigurationError, match="successful Phase 5 canary summary"):
        run_phase5_live_experiment(
            repository_root=repository,
            registry_path=repository / "experiments/phase5-development.yaml",
            authority_path=authority,
            allow_live_model=True,
            backend=backend,
        )
    assert backend.dispatch_count == 0


def test_fake_matched_development_pilot_writes_nonconfirmatory_analysis(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    canary_authority = repository / "experiments/phase5-canary.pending.yaml"
    _authorize(canary_authority, "test-only-canary-prerequisite")
    run_phase5_live_experiment(
        repository_root=repository,
        registry_path=repository / "experiments/phase5-canary.yaml",
        authority_path=canary_authority,
        allow_live_model=True,
        backend=_CompatibleFakeOpenAI(),
    )

    development_path = repository / "experiments/phase5-development.yaml"
    development_raw = yaml.safe_load(development_path.read_text(encoding="utf-8"))
    assert isinstance(development_raw, dict)
    development_raw["task_selection"]["task_ids"] = ["2c2c7c8b36360e3ceff351b3"]
    development_raw["search_seeds"] = [55001]
    matched = development_raw["matched_contract"]
    matched["requests_per_child"] = 1
    matched["input_token_cap"] = 12000
    matched["output_token_cap"] = 2048
    matched["total_token_cap"] = 14048
    matched["proposal_item_cap"] = 1
    matched["oracle_call_cap"] = 1
    development_path.write_text(yaml.safe_dump(development_raw, sort_keys=False), encoding="utf-8")
    experiment = load_phase5_live_experiment(development_path)
    development_authority = repository / "experiments/phase5-development.pending.yaml"
    authority_raw = yaml.safe_load(development_authority.read_text(encoding="utf-8"))
    authority_raw["experiment_hash"] = experiment.source_hash
    development_authority.write_text(
        yaml.safe_dump(authority_raw, sort_keys=False), encoding="utf-8"
    )
    _authorize(development_authority, "test-only-matched-development")
    summary = run_phase5_live_experiment(
        repository_root=repository,
        registry_path=development_path,
        authority_path=development_authority,
        allow_live_model=True,
        backend=_CompatibleFakeOpenAI(),
    )
    assert summary["status"] == "completed-live-development-pilot"
    assert summary["child_count"] == 2
    assert summary["scientific_status"] == "development-evidence-only-h3-unconfirmed"
    analysis = yaml.safe_load(
        (repository / "artifacts/phase5/live-development-v1/analysis.json").read_text()
    )
    assert analysis["confirmatory"] is False
    assert analysis["h3_confirmed"] is False
    assert analysis["library_definition_cost_bits_charged_once"] == 50
    assert all(
        cell["definition_cost_allocated_to_cell_bits"] == 0 for cell in analysis["transfer_matrix"]
    )
    replay = replay_phase5_live_experiment(
        repository_root=repository, registry_path=development_path
    )
    assert replay["status"] == "verified-provider-disabled"


def test_live_freeze_rejects_policy_or_artifact_drift(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    policy = repository / "configs/phase5-exposure-policy-v2.yaml"
    policy.write_text(policy.read_text().replace("one_request: 10000000", "one_request: 1"))
    with pytest.raises(ConfigurationError, match="exposure policy hash differs"):
        phase5_live_dry_run(
            repository_root=repository,
            registry_path=repository / "experiments/phase5-canary.yaml",
            authority_path=repository / "experiments/phase5-canary.pending.yaml",
        )


def test_canary_preflight_counts_existing_stage_exposure(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    price = load_price_policy(repository / "configs/project-dual-budget-policy-v2.yaml")
    with ProjectLedger(
        repository / "local_state/project-dual-budget-ledger.sqlite3", price
    ) as ledger:
        ledger.reserve(
            reservation_id="prior-phase5-canary-exposure",
            run_id="prior-phase5-canary",
            stage="canary",
            request_hash="0" * 64,
            amount_nano_usd=4_000_000,
            child_cap_nano_usd=150_000_000,
        )
        ledger.reconcile(
            reservation_id="prior-phase5-canary-exposure",
            actual_nano_usd=4_000_000,
            usage_record={"test_only_prior_phase5_canary_nano_usd": 4_000_000},
        )
    with pytest.raises(ConfigurationError, match="remaining stage exposure"):
        phase5_live_dry_run(
            repository_root=repository,
            registry_path=repository / "experiments/phase5-canary.yaml",
            authority_path=repository / "experiments/phase5-canary.pending.yaml",
        )


def test_live_registry_hashes_are_stably_frozen() -> None:
    canary = load_phase5_live_experiment(Path("experiments/phase5-canary.yaml"))
    development = load_phase5_live_experiment(Path("experiments/phase5-development.yaml"))
    assert canary.source_hash == "09d543e54bf61149f7fb167c259ed6b31a9ac1508756c3a3c3849fa8270546b6"
    assert (
        development.source_hash
        == "f4e15986338ab042713c90a50953b983076deb5b65efbd5a45a770314263d5d3"
    )
