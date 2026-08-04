from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from world_model_search.config import AppConfig
from world_model_search.evaluation.report import create_recorded_report
from world_model_search.oracle.mock import MockOracle
from world_model_search.proposer.mock import MockProposer
from world_model_search.search.loop import start_run


def test_report_uses_frozen_run_data(
    repository_root: Path, app_config: AppConfig, monkeypatch: MonkeyPatch
) -> None:
    start_run(
        repository_root=repository_root,
        config=app_config,
        config_source="test",
        run_id="reportable",
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("report invoked live computation")

    monkeypatch.setattr(MockProposer, "propose", forbidden)
    monkeypatch.setattr(MockOracle, "evaluate", forbidden)
    json_path, markdown_path = create_recorded_report(
        repository_root=repository_root,
        runs_root=Path("runs"),
        run_id="reportable",
        output_directory=repository_root / "report",
    )
    assert json_path.is_file()
    assert markdown_path.is_file()
