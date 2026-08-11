from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pytest import MonkeyPatch

from world_model_search.config import load_config
from world_model_search.search.phase4 import start_phase4_run


def test_phase4_provider_boundary_artifacts_exclude_hidden_state_key_and_paths(
    phase2_repository: Path, monkeypatch: MonkeyPatch
) -> None:
    policy = phase2_repository / "configs/phase4-price-policy-v1.yaml"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_bytes(Path("configs/phase4-price-policy-v1.yaml").read_bytes())
    config = load_config(Path("configs/phase4-fake-smoke.yaml"))
    assert config.cache is not None
    config = replace(
        config,
        cache=replace(config.cache, namespace="phase4-leakage-boundary"),
    )
    sentinel = "phase4-secret-sentinel-must-not-cross-boundary"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    outcome = start_phase4_run(
        repository_root=phase2_repository,
        config=config,
        config_source="phase4-leakage",
        run_id="phase4-leakage",
        interrupt_after=None,
        allow_live_model=False,
    )
    boundary_roots = (
        outcome.run_directory / "prompts",
        outcome.run_directory / "requests",
        outcome.run_directory / "responses",
        phase2_repository / config.cache.root / config.cache.namespace,
    )
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in boundary_roots
        for path in sorted(root.glob("*.json"))
    )
    forbidden = (
        sentinel,
        str(phase2_repository),
        "internal_family",
        "semantic_hash",
        "hidden_cases",
        "hidden_rollout",
        "reference_rule",
        "test_assignment",
    )
    assert not [token for token in forbidden if token in text]
