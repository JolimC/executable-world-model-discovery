"""One typed end-to-end fixture with deliberately no cellular-automaton logic."""

from __future__ import annotations

from world_model_search.config import AppConfig
from world_model_search.domain.types import PublicDemonstration, PublicTask, Task
from world_model_search.serialization import derive_seed, sha256_json


def make_fixture_task(config: AppConfig) -> Task:
    demonstrations = (
        PublicDemonstration(observation="fixture:initial", successor="fixture:changed"),
    )
    public_task = PublicTask(
        task_id=config.run.task_id,
        family="phase0-no-ca-fixture",
        split=config.run.split,
        demonstrations=demonstrations,
        active_queries_enabled=False,
        query_budget=0,
    )
    return Task(
        task_id=public_task.task_id,
        family=public_task.family,
        split=public_task.split,
        public_demonstrations=public_task.demonstrations,
        active_queries_enabled=public_task.active_queries_enabled,
        query_budget=public_task.query_budget,
        exact_case_set_id="phase0-mock-cases-v1",
        rollout_suite_id="phase0-mock-rollout-v1",
        public_artifact_hash=sha256_json(public_task),
        hidden_artifact_id="phase0-mock-oracle-artifact-v1",
        generator_version="phase0-fixture-v1",
        seed=derive_seed(config.run.seed, "task"),
    )
