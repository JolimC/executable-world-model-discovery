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
    assert str(task.seed) not in encoded
    assert str(app_config.run.seed) not in encoded
