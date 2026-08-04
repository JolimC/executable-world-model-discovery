from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from world_model_search.cli import main


def test_invalid_configuration_creates_no_run_artifacts(
    repository_root: Path, monkeypatch: MonkeyPatch
) -> None:
    repository_root.mkdir()
    config_path = repository_root / "invalid.yaml"
    config_path.write_text(
        """schema_version: 1
run:
  root: runs
  seed: 3
  max_steps: 0
  task_id: invalid
  split: training
proposer:
  id: mock
  batch_size: 1
oracle:
  id: mock-v1
logging:
  level: INFO
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(repository_root)
    exit_code = main(["solve", "--config", str(config_path), "--run-id", "must-not-exist"])
    assert exit_code == 2
    assert not (repository_root / "runs").exists()
    assert tuple(repository_root.iterdir()) == (config_path,)
