from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from world_model_search.config import AppConfig
from world_model_search.domain.types import ProposalContext, SplitLabel
from world_model_search.search.fixture import make_fixture_task
from world_model_search.serialization import canonical_json


def test_split_labels_are_the_phase0_contract() -> None:
    assert tuple(label.value for label in SplitLabel) == (
        "training",
        "development",
        "validation",
        "test",
    )


def test_task_split_is_immutable(app_config: AppConfig) -> None:
    task = make_fixture_task(app_config)
    with pytest.raises(FrozenInstanceError):
        task.split = SplitLabel.TEST  # type: ignore[misc]


def test_proposal_context_structurally_excludes_oracle_only_task_fields(
    app_config: AppConfig,
) -> None:
    task = make_fixture_task(app_config)
    context = ProposalContext(task=task.public_view())
    context_fields = {field.name for field in fields(context.task)}
    forbidden_fields = {
        "family",
        "internal_family_id",
        "seed",
        "hidden_artifact_id",
        "exact_case_set_id",
        "rollout_suite_id",
        "public_artifact_hash",
    }
    assert context_fields.isdisjoint(forbidden_fields)
    encoded = canonical_json(context)
    assert task.hidden_artifact_id not in encoded
    assert task.exact_case_set_id not in encoded
    assert task.internal_family_id not in encoded
    assert str(task.seed) not in encoded
    assert str(app_config.run.seed) not in encoded


def test_public_world_spec_contains_mechanics_not_internal_family(
    app_config: AppConfig,
) -> None:
    task = make_fixture_task(app_config)
    public_task = task.public_view()
    assert public_task.public_world_spec == task.public_world_spec
    assert public_task.public_world_spec.observation_type == "opaque-string-v1"
    assert public_task.public_world_spec.candidate_type == "phase0-rule-expr-stub-v1"
    assert task.internal_family_id == "phase0-no-ca-fixture"
    assert "family" not in canonical_json(public_task)
