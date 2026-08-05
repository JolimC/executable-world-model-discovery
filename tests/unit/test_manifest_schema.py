from __future__ import annotations

from pathlib import Path

import pytest

from world_model_search.errors import PersistenceError
from world_model_search.search.loop import load_manifest


def test_legacy_manifest_schema_is_rejected_before_resume_or_replay(tmp_path: Path) -> None:
    run_directory = tmp_path / "legacy-run"
    run_directory.mkdir()
    (run_directory / "manifest.json").write_text(
        '{"manifest_schema_version":1}\n', encoding="utf-8"
    )
    with pytest.raises(PersistenceError, match="requires schema 2"):
        load_manifest(run_directory)
