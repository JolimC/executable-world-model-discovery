from __future__ import annotations

from pathlib import Path

import pytest

from world_model_search.config import AppConfig, config_from_mapping, load_config
from world_model_search.tasks import generate_phase2_benchmark


@pytest.fixture
def app_config() -> AppConfig:
    return config_from_mapping(
        {
            "schema_version": 1,
            "run": {
                "root": "runs",
                "seed": 314159,
                "max_steps": 4,
                "task_id": "test-fixture",
                "split": "training",
            },
            "proposer": {"id": "mock", "batch_size": 1},
            "oracle": {"id": "mock-v1"},
            "logging": {"level": "ERROR"},
        }
    )


@pytest.fixture
def repository_root(tmp_path: Path) -> Path:
    return tmp_path / "repository"


@pytest.fixture(scope="session")
def phase2_repository(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create the ignored Phase 2 benchmark for tests in an isolated repository."""

    repository = tmp_path_factory.mktemp("phase2-repository")
    generate_phase2_benchmark(repository, load_config(Path("configs/smoke.yaml")))
    return repository
