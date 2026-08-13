from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from world_model_search.dsl.primitives import load_primitive_registry
from world_model_search.errors import ConfigurationError, ReplayError
from world_model_search.evaluation.phase5_experiment import (
    load_phase5_experiment,
    phase5_dry_run,
    replay_phase5_smoke,
    run_phase5_smoke,
)
from world_model_search.evaluation.phase5_transfer import load_transfer_registry
from world_model_search.memory.types import load_memory_snapshot
from world_model_search.serialization import sha256_json


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    for directory in ("configs", "experiments", "local_state"):
        (repository / directory).mkdir(parents=True)
    for source in (
        Path("configs/phase5-transfer-split-v1.yaml"),
        Path("configs/phase5-exposure-policy-v1.yaml"),
        Path("configs/project-dual-budget-policy-v2.yaml"),
        Path("experiments/phase5-smoke.yaml"),
    ):
        shutil.copy2(source, repository / source)
    return repository


def test_phase5_smoke_replay_report_and_exact_resume_are_no_cost(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    registry_path = repository / "experiments/phase5-smoke.yaml"
    experiment = load_phase5_experiment(registry_path)
    dry = phase5_dry_run(repository_root=repository, experiment=experiment)
    assert dry["sealed_test_authorized"] is False
    assert dry["model_request_cap"] == 0
    forecast = dry["forecast"]
    assert isinstance(forecast, dict) and forecast["fits_current_cash_headroom"] is True

    first = run_phase5_smoke(repository_root=repository, registry_path=registry_path)
    second = run_phase5_smoke(repository_root=repository, registry_path=registry_path)
    assert first == second
    assert first["provider_calls"] == 0
    assert first["sealed_test_accesses"] == 0
    assert first["promotion_status"] == "promoted-development-evidence"
    net_gain = first["net_gain_bits"]
    assert isinstance(net_gain, int) and net_gain > 0
    replay = replay_phase5_smoke(repository_root=repository, registry_path=registry_path)
    assert replay["status"] == "verified-provider-disabled"
    assert replay["provider_calls"] == 0

    root = repository / experiment.output_root
    matrix = json.loads((root / "transfer-matrix.json").read_text())
    assert matrix["library_definition_cost_bits_separate"] > 0
    assert all(cell["definition_cost_allocated_to_cell_bits"] == 0 for cell in matrix["cells"])
    off = json.loads((root / "condition-c-manifest.json").read_text())
    on = json.loads((root / "condition-d-manifest.json").read_text())
    allowed = {
        "condition",
        "memory_snapshot_hash",
        "retrieval_record_hashes",
        "rendered_memory_hashes",
        "primitive_registry_hash",
    }
    assert all(off[key] == on[key] for key in off.keys() - allowed)


def test_phase5_interruption_resumes_exact_frozen_memory_without_provider_calls(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    registry_path = repository / "experiments/phase5-smoke.yaml"
    interrupted = run_phase5_smoke(
        repository_root=repository,
        registry_path=registry_path,
        interrupt_after_memory=True,
    )
    assert interrupted["status"] == "interrupted-after-frozen-memory"
    assert interrupted["provider_calls"] == 0
    resumed = run_phase5_smoke(repository_root=repository, registry_path=registry_path)
    assert resumed["status"] == "completed-no-cost-development-smoke"
    assert resumed["provider_calls"] == 0
    replay = replay_phase5_smoke(repository_root=repository, registry_path=registry_path)
    assert replay["status"] == "verified-provider-disabled"


def test_phase5_replay_rejects_artifact_corruption(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    registry_path = repository / "experiments/phase5-smoke.yaml"
    experiment = load_phase5_experiment(registry_path)
    run_phase5_smoke(repository_root=repository, registry_path=registry_path)
    analysis = repository / experiment.output_root / "analysis.json"
    analysis.write_text(analysis.read_text().replace('"model_tokens":0', '"model_tokens":1'))
    with pytest.raises(ReplayError, match="artifact hash mismatch"):
        replay_phase5_smoke(repository_root=repository, registry_path=registry_path)


def test_phase5_budget_forecast_fails_closed_above_partition(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    policy = repository / "configs/phase5-exposure-policy-v1.yaml"
    policy.write_text(policy.read_text().replace("one_request: 10000000", "one_request: 1"))
    experiment = load_phase5_experiment(repository / "experiments/phase5-smoke.yaml")
    with pytest.raises(ConfigurationError, match="exceeds a published exposure partition"):
        phase5_dry_run(repository_root=repository, experiment=experiment)


def test_phase5_live_dispatch_and_sealed_test_remain_unauthorized(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    registry_path = repository / "experiments/phase5-smoke.yaml"
    with pytest.raises(ConfigurationError, match="forbids live model calls"):
        run_phase5_smoke(
            repository_root=repository,
            registry_path=registry_path,
            allow_live_model=True,
        )


def test_phase5_live_and_sealed_declarations_bind_hashes_and_preserve_test_denial() -> None:
    transfer = load_transfer_registry(Path("configs/phase5-transfer-split-v1.yaml"))
    exposure = yaml.safe_load(Path("configs/phase5-exposure-policy-v2.yaml").read_text())
    assert isinstance(exposure, dict)
    exposure_hash = sha256_json(exposure)
    canary = yaml.safe_load(Path("experiments/phase5-canary.pending.yaml").read_text())
    development = yaml.safe_load(Path("experiments/phase5-development.pending.yaml").read_text())
    sealed = yaml.safe_load(Path("experiments/phase5-test.sealed.yaml").read_text())
    assert isinstance(canary, dict) and isinstance(development, dict) and isinstance(sealed, dict)
    assert canary["exposure_policy_hash"] == exposure_hash
    assert canary["status"] == "authorized"
    assert all(canary["authorization"].values())
    assert development["exposure_policy_hash"] == exposure_hash
    assert development["status"] == "authorized"
    assert all(development["authorization"].values())
    assert sealed["transfer_registry_hash"] == transfer.content_hash
    assert sealed["exposure_policy_hash"] == exposure_hash
    assert sealed["status"] == "sealed-not-authorized"
    plan = yaml.safe_load(Path("experiments/phase5-final-freeze/analysis-plan.json").read_text())
    memory = load_memory_snapshot(Path("experiments/phase5-final-freeze/memory-snapshot.json"))
    primitives = load_primitive_registry(
        Path("experiments/phase5-final-freeze/primitive-registry.json")
    )
    assert sealed["analysis_plan_hash"] == sha256_json(plan)
    assert sealed["memory_snapshot_hash"] == memory.snapshot_hash
    assert sealed["primitive_registry_hash"] == primitives.registry_hash
    assert sealed["authorization"] == {
        "model_calls": False,
        "test_oracle": False,
        "one_time_authority_hash": None,
    }
