from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from world_model_search.config import load_config
from world_model_search.evaluation.phase5_contextual import (
    FROZEN_ARMS,
    _check_pair_prompt_isolation,
    _child_run_id,
    load_contextual_experiment_registry,
)
from world_model_search.memory.contextual import (
    ContextMode,
    PromptProjection,
    RetrievalMode,
    SelectionPolicy,
)
from world_model_search.memory.contextual_retrieval import (
    ContextualExperienceRuntime,
    ContextualMemoryConfig,
)
from world_model_search.search.phase4 import start_phase4_run


def _install_policy(repository: Path) -> None:
    target = repository / "configs/phase4-price-policy-v1.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path("configs/phase4-price-policy-v1.yaml").read_bytes())


def test_contextual_registry_freezes_all_ablation_arms() -> None:
    registry = load_contextual_experiment_registry(
        Path("experiments/phase5-contextual-v3-smoke.yaml")
    )
    assert registry.arms == FROZEN_ARMS
    assert registry.memory_config.weights.error_pattern == 0
    assert registry.memory_config.exposure.enabled is False
    assert registry.memory_config.prompt_projection is PromptProjection.FULL_GREEDY_V1
    assert registry.memory_config.selection_policy is SelectionPolicy.SIMILARITY_RECORD_ID_V1
    assert registry.memory_config.include_aggregate_summary is False
    assert all(
        len(_child_run_id(task_id="a" * 24, seed=55001, arm=arm)) <= 80 for arm in FROZEN_ARMS
    )


def test_repair_smoke_registry_enables_v2_projection_and_selection() -> None:
    registry = load_contextual_experiment_registry(
        Path("experiments/phase5-contextual-v3-repair-smoke.yaml")
    )
    assert registry.arms == FROZEN_ARMS
    assert registry.memory_config.prompt_projection is PromptProjection.COMPACT_ADAPTIVE_V2
    assert registry.memory_config.selection_policy is SelectionPolicy.TASK_BALANCED_CONTRAST_V2
    assert registry.memory_config.include_aggregate_summary is True
    assert registry.memory_config.aggregate_max_signatures == 6


def test_prompt_isolation_skips_budget_capped_children_without_prompts(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    treatment = tmp_path / "treatment"
    control.mkdir()
    treatment.mkdir()
    assert _check_pair_prompt_isolation(control, treatment) is False


def test_phase4_contextual_control_logs_empty_retrieval_and_all_valid_transitions(
    phase2_repository: Path,
) -> None:
    _install_policy(phase2_repository)
    config = load_config(Path("configs/phase4-fake-smoke.yaml"))
    assert config.cache is not None
    config = replace(
        config,
        cache=replace(config.cache, namespace="contextual-integration"),
    )
    runtime = ContextualExperienceRuntime(
        "A-no-memory",
        None,
        ContextualMemoryConfig(
            retrieval_mode=RetrievalMode.DISABLED,
            context_mode=ContextMode.RICH,
        ),
        (config.run.task_id,),
    )
    outcome = start_phase4_run(
        repository_root=phase2_repository,
        config=config,
        config_source="contextual-integration",
        run_id="contextual-control",
        interrupt_after=None,
        allow_live_model=False,
        contextual_experience_runtime=runtime,
    )
    assert outcome.status == "completed"
    root = outcome.run_directory / "experience_v3"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["record_count"] == 4
    assert summary["orphan_raw_record_count"] == 0
    assert summary["duplicate_count"] > 0
    retrievals = sorted((root / "retrieval").glob("*.json"))
    assert len(retrievals) == 2
    for path in retrievals:
        retrieval = json.loads(path.read_text())
        assert retrieval["retrieval"]["all_eligible_scores"] == []
        assert retrieval["rendered_memory"]["shown_record_ids"] == []
    prompts = sorted((outcome.run_directory / "prompts").glob("*.json"))
    assert len(prompts) == 2
    for path in prompts:
        prompt = json.loads(path.read_text())
        assert prompt["cross_task_memory"]["cross_task_experience"] == []
