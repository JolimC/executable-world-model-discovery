from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import time
from dataclasses import fields
from pathlib import Path

from pytest import MonkeyPatch

from world_model_search.config import load_config
from world_model_search.domain.types import ProposalContext, SplitLabel
from world_model_search.dsl.ast import At, Xor
from world_model_search.dsl.interpreter import evaluate
from world_model_search.serialization import canonical_json
from world_model_search.tasks import HiddenTaskStore, benchmark_root_for_config, load_public_task


def test_nested_public_context_excludes_hidden_training_values_and_encodings(
    phase2_repository: Path,
) -> None:
    config = load_config(Path("configs/phase2-smoke.yaml"))
    root = benchmark_root_for_config(phase2_repository, config)
    task = load_public_task(root, config.run.task_id)
    hidden = HiddenTaskStore(root).load(
        task.task_id,
        allowed_splits=frozenset({SplitLabel.TRAINING}),
        purpose="leakage-test-training-only",
    )
    context = ProposalContext(task=task.public_view())
    encoded = canonical_json(context)
    public_fields = {field.name for field in fields(context.task)}
    assert public_fields.isdisjoint(
        {
            "internal_family_id",
            "seed",
            "hidden_artifact_id",
            "exact_case_set_id",
            "rollout_suite_id",
            "semantic_hash",
        }
    )
    for forbidden_key in (
        "reference_rule",
        "ordered_semantics_000_to_111",
        "semantic_hash",
        "internal_family",
        "locked_rollout",
        "hidden_artifact_id",
        "exact_case_set_id",
        "rollout_suite_id",
        "seeds",
    ):
        assert forbidden_key not in encoded
    for sensitive_value in (
        hidden.semantic_hash,
        hidden.internal_family,
        str(hidden.task_seed),
        str(hidden.rollout_case_seed),
        canonical_json(hidden.ordered_semantics),
        canonical_json(hidden.locked_rollout.states),
    ):
        assert sensitive_value not in encoded
    assert context.task.public_world_spec.candidate_type == "typed-json-ast-v1"
    assert context.task.public_world_spec.neighborhood_order == ("left", "center", "right")


def test_all_public_bundles_remain_free_of_oracle_field_encodings(
    phase2_repository: Path,
) -> None:
    public_root = phase2_repository / "artifacts/phase2-benchmark/public"
    paths = sorted(public_root.glob("*.json"))
    assert len(paths) == 256
    text = "".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in (
        "reference_rule",
        "ordered_semantics_000_to_111",
        "semantic_hash",
        "locked_rollout",
        "internal_family",
        '"seeds"',
        '"task_seed"',
        '"hidden_artifact_id"',
    ):
        assert forbidden not in text
    for path in paths:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        assert bundle["public_local_case_coverage"] == ["000", "111"]
        observed_cases: set[str] = set()
        for demo in bundle["demonstrations"]:
            state = demo["observation"]
            for index in range(len(state)):
                observed_cases.add(
                    state[index - 1] + state[index] + state[(index + 1) % len(state)]
                )
        assert observed_cases == {"000", "111"}


def test_interpreter_has_no_file_environment_network_clock_random_or_subprocess_capability(
    monkeypatch: MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("candidate interpreter crossed its capability boundary")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(time, "time", forbidden)
    monkeypatch.setattr(random, "random", forbidden)
    assert evaluate(Xor(At(-1), At(1)), (1, 0, 0)) == 1
