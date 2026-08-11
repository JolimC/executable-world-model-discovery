from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from pytest import MonkeyPatch

from world_model_search.config import load_config
from world_model_search.domain.types import ProposalBudget, ProposalContext, PublicTask
from world_model_search.search.loop import start_run
from world_model_search.search.operators import MutationProposer, OperatorAttempt
from world_model_search.tasks import load_public_task

FORBIDDEN_KEYS = {
    "reference_rule",
    "ordered_semantics_000_to_111",
    "hidden_artifact_id",
    "hidden_seed",
    "internal_family",
    "semantic_hash",
    "rollout_suite_id",
    "exact_case_set_id",
}


def _scan(value: object, forbidden_values: frozenset[str] = frozenset()) -> None:
    if isinstance(value, dict):
        assert not (set(value) & FORBIDDEN_KEYS)
        for item in value.values():
            _scan(item, forbidden_values)
    elif isinstance(value, list):
        for item in value:
            _scan(item, forbidden_values)
    elif isinstance(value, str):
        assert all(secret not in value for secret in forbidden_values)


def test_phase3_proposal_parent_archive_and_scheduler_contexts_do_not_leak(
    phase2_repository: Path, monkeypatch: MonkeyPatch
) -> None:
    config = load_config(Path("configs/phase3-smoke.yaml"))
    assert config.budget is not None
    config = replace(
        config,
        budget=replace(config.budget, proposal_attempt_cap=32, oracle_call_cap=10),
    )
    benchmark_root = phase2_repository / "artifacts/phase2-benchmark"
    internal_task = load_public_task(benchmark_root, config.run.task_id)
    hidden = json.loads((benchmark_root / "oracle" / f"{config.run.task_id}.json").read_text())
    forbidden_values = frozenset(
        {
            str(hidden["semantic_hash"]),
            *(str(value) for value in hidden["seeds"].values()),
            str(internal_task.seed),
            internal_task.exact_case_set_id,
            internal_task.rollout_suite_id,
        }
    )
    captured_contexts: list[ProposalContext] = []
    original_propose = MutationProposer.propose

    def capturing_propose(
        self: MutationProposer, context: ProposalContext, budget: ProposalBudget
    ) -> tuple[OperatorAttempt, ...]:
        captured_contexts.append(context)
        return original_propose(self, context, budget)

    monkeypatch.setattr(MutationProposer, "propose", capturing_propose)
    outcome = start_run(
        repository_root=phase2_repository,
        config=config,
        config_source="leakage-test",
        run_id="phase3-leakage",
    )
    for proposal_path in sorted((outcome.run_directory / "proposals").glob("*.json")):
        proposal = json.loads(proposal_path.read_text())
        _scan(proposal["public_context"], forbidden_values)
        assert set(proposal["public_context"]) == {"task", "parents", "feedback"}
        assert "proposer_seed" not in proposal_path.read_text()
    assert captured_contexts
    assert all(type(context.task) is PublicTask for context in captured_contexts)
    assert all(not context.feedback for context in captured_contexts)
    assert set(PublicTask.__dataclass_fields__) == {
        "task_id",
        "public_world_spec",
        "split",
        "demonstrations",
        "active_queries_enabled",
        "query_budget",
    }
    connection = sqlite3.connect(outcome.run_directory / "run.sqlite3")
    try:
        for (payload_text,) in connection.execute("SELECT payload_json FROM event"):
            payload = json.loads(payload_text)
            _scan(payload["scheduler"], forbidden_values)
            _scan(payload["archive_decision"], forbidden_values)
    finally:
        connection.close()
