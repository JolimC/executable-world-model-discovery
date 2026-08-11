from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from world_model_search.config import config_from_mapping, load_config
from world_model_search.domain.types import Archive, Proposer, Scheduler
from world_model_search.errors import ConfigurationError
from world_model_search.evaluation.phase3_experiment import (
    _bootstrap_interval,
    _csv,
    _task_clustered_bootstrap_interval,
    load_experiment_registry,
)
from world_model_search.proposer.mock import MockProposer
from world_model_search.scheduler.uniform import UniformScheduler
from world_model_search.search.archive import MapElitesArchive, SingleIncumbent
from world_model_search.search.operators import MutationProposer
from world_model_search.tasks import benchmark_root_for_config, load_public_task


def test_phase3_config_and_twenty_seed_registry_are_strict() -> None:
    config = load_config(Path("configs/phase3-smoke.yaml"))
    assert config.schema_version == 3
    assert config.budget is not None and config.budget.oracle_call_cap == 32
    registry = load_experiment_registry(Path("experiments/phase3-archive-smoke.yaml"))
    assert len(registry.task_ids) == 12
    assert len(registry.search_seeds) == 20
    assert registry.oracle_call_cap == 32
    raw = config.to_mapping()
    raw["unexpected"] = True
    with pytest.raises(ConfigurationError):
        config_from_mapping(raw)


def test_phase3_benchmark_authority_is_output_root_independent() -> None:
    config = load_config(Path("configs/phase3-smoke.yaml"))
    nested = replace(config, run=replace(config.run, root=Path("artifacts/nested/children")))
    assert benchmark_root_for_config(Path("/repository"), nested) == Path(
        "/repository/artifacts/phase2-benchmark"
    )


def test_experiment_csv_projection_ignores_json_only_fields() -> None:
    assert _csv([{"selected": 1, "json_only": {"inserted": 2}}], ("selected",)) == ("selected\n1")


def test_phase3_components_implement_shared_runtime_interfaces(phase2_repository: Path) -> None:
    task = load_public_task(
        phase2_repository / "artifacts/phase2-benchmark", "d737b0ee219de6a676c139d1"
    ).public_view()
    assert isinstance(MockProposer(), Proposer)
    assert isinstance(MutationProposer(), Proposer)
    assert isinstance(MapElitesArchive(task), Archive)
    assert isinstance(SingleIncumbent(task), Archive)
    assert isinstance(UniformScheduler(), Scheduler)


def test_task_clustered_bootstrap_is_deterministic_and_preserves_task_dependence() -> None:
    by_task = {
        "task-a": (-1.0,) * 20,
        "task-b": (-1.0,) * 20,
        "task-c": (1.0,) * 20,
        "task-d": (1.0,) * 20,
    }
    flattened = tuple(value for values in by_task.values() for value in values)
    flat = _bootstrap_interval(flattened, seed=71, replicates=2000)
    clustered = _task_clustered_bootstrap_interval(by_task, seed=71, replicates=2000)
    assert clustered == _task_clustered_bootstrap_interval(
        dict(reversed(tuple(by_task.items()))), seed=71, replicates=2000
    )
    assert clustered[0] <= flat[0]
    assert clustered[1] >= flat[1]
