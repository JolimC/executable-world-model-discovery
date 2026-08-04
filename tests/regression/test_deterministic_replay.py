"""Phase 0 presentable deterministic replay regression test."""

from __future__ import annotations

from pathlib import Path

from world_model_search.config import AppConfig
from world_model_search.replay import replay_run
from world_model_search.search.loop import start_run


def test_frozen_proposals_reproduce_every_event_hash(
    repository_root: Path, app_config: AppConfig
) -> None:
    recorded = start_run(
        repository_root=repository_root,
        config=app_config,
        config_source="regression-test",
        run_id="deterministic-replay",
    )
    replayed = replay_run(
        repository_root=repository_root,
        runs_root=Path("runs"),
        run_id="deterministic-replay",
    )
    assert replayed.event_count == app_config.run.max_steps
    assert replayed.event_payload_hashes == recorded.event_payload_hashes
    assert replayed.proposer_invocations == 0
