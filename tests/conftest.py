from __future__ import annotations

from pathlib import Path

import pytest

from world_model_search.config import AppConfig, config_from_mapping


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
