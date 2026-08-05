from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from world_model_search.config import AppConfig
from world_model_search.persistence.database import RunDatabase
from world_model_search.proposer.mock import MockProposer
from world_model_search.replay import replay_run
from world_model_search.search.loop import resume_run, start_run


def test_run_can_interrupt_resume_and_replay_deterministically(
    repository_root: Path, app_config: AppConfig
) -> None:
    interrupted = start_run(
        repository_root=repository_root,
        config=app_config,
        config_source="test",
        run_id="resumable",
        interrupt_after=2,
    )
    assert interrupted.status == "interrupted"
    assert interrupted.completed_steps == 2
    assert not (interrupted.run_directory / "results.json").exists()
    manifest = json.loads((interrupted.run_directory / "manifest.json").read_text(encoding="utf-8"))
    task_manifest = manifest["tasks"][0]
    assert task_manifest["internal_family_id"] == "phase0-no-ca-fixture"
    assert task_manifest["public_world_spec"]["candidate_type"] == "phase0-rule-expr-stub-v1"
    assert "family" not in task_manifest
    with RunDatabase(interrupted.run_directory / "run.sqlite3", read_only=True) as database:
        assert database.state().status == "interrupted"

    resumed = resume_run(
        repository_root=repository_root,
        runs_root=Path("runs"),
        run_id="resumable",
    )
    assert resumed.status == "completed"
    assert resumed.event_payload_hashes[:2] == interrupted.event_payload_hashes
    assert len(resumed.event_payload_hashes) == app_config.run.max_steps

    replay = replay_run(
        repository_root=repository_root,
        runs_root=Path("runs"),
        run_id="resumable",
    )
    assert replay.event_payload_hashes == resumed.event_payload_hashes
    assert replay.proposer_invocations == 0


def test_identical_seeds_have_identical_event_payload_hashes(
    repository_root: Path, app_config: AppConfig
) -> None:
    first = start_run(
        repository_root=repository_root,
        config=app_config,
        config_source="test-a",
        run_id="independent-a",
    )
    second = start_run(
        repository_root=repository_root,
        config=app_config,
        config_source="test-b",
        run_id="independent-b",
    )
    assert first.event_payload_hashes == second.event_payload_hashes


def test_replay_does_not_invoke_proposal_generation(
    repository_root: Path, app_config: AppConfig, monkeypatch: MonkeyPatch
) -> None:
    outcome = start_run(
        repository_root=repository_root,
        config=app_config,
        config_source="test",
        run_id="no-proposer-replay",
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("replay called a proposer")

    monkeypatch.setattr(MockProposer, "propose", forbidden)
    replay = replay_run(
        repository_root=repository_root,
        runs_root=Path("runs"),
        run_id="no-proposer-replay",
    )
    assert replay.event_payload_hashes == outcome.event_payload_hashes
