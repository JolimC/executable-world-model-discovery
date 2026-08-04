from __future__ import annotations

from pathlib import Path

import pytest

from world_model_search.config import config_from_mapping, load_config
from world_model_search.domain.types import SplitLabel
from world_model_search.errors import ConfigurationError


def test_configuration_round_trip_uses_external_schema() -> None:
    raw = {
        "schema_version": 1,
        "run": {
            "root": "runs",
            "seed": 7,
            "max_steps": 2,
            "task_id": "fixture",
            "split": "development",
        },
        "proposer": {"id": "mock", "batch_size": 1},
        "oracle": {"id": "mock-v1"},
        "logging": {"level": "info"},
    }
    config = config_from_mapping(raw)
    assert config.run.split is SplitLabel.DEVELOPMENT
    assert config.to_mapping()["proposer"] == {"id": "mock", "batch_size": 1}
    assert config_from_mapping(config.to_mapping()) == config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("seed", -1),
        ("split", "meta-test"),
        ("root", "/tmp/outside"),
        ("root", "../outside"),
    ],
)
def test_invalid_run_configuration_is_rejected(field: str, value: object) -> None:
    raw: dict[str, object] = {
        "schema_version": 1,
        "run": {
            "root": "runs",
            "seed": 7,
            "max_steps": 2,
            "task_id": "fixture",
            "split": "training",
        },
        "proposer": {"id": "mock", "batch_size": 1},
        "oracle": {"id": "mock-v1"},
        "logging": {"level": "INFO"},
    }
    run = raw["run"]
    assert isinstance(run, dict)
    run[field] = value
    with pytest.raises(ConfigurationError):
        config_from_mapping(raw)


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("run: [", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)
