from __future__ import annotations

import json
from pathlib import Path

import pytest

from world_model_search.domain.types import ProposalRole, SplitLabel
from world_model_search.dsl.primitives import empty_primitive_registry
from world_model_search.errors import OracleVerificationError
from world_model_search.evaluation.phase5_transfer import (
    Phase5TaskStore,
    generate_transfer_benchmark,
    load_transfer_public_task,
    load_transfer_registry,
)
from world_model_search.memory.retrieval import retrieve_memory
from world_model_search.memory.types import (
    MemoryApplicability,
    MemoryKind,
    MemorySnapshot,
    SafeMemoryItem,
)
from world_model_search.model.phase5_prompts import render_phase5_prompt
from world_model_search.serialization import sha256_json


def test_phase5_public_task_retrieval_and_prompt_omit_evaluator_metadata(tmp_path: Path) -> None:
    registry = load_transfer_registry(Path("configs/phase5-transfer-split-v1.yaml"))
    benchmark = generate_transfer_benchmark(tmp_path, registry)
    tasks = benchmark.manifest["tasks"]
    assert isinstance(tasks, list)
    development = next(
        item for item in tasks if isinstance(item, dict) and item["split"] == "development"
    )
    task = load_transfer_public_task(benchmark.root, str(development["task_id"])).public_view()
    snapshot = MemorySnapshot(registry.content_hash, sha256_json({"empty": True}), ())
    retrieval = retrieve_memory(
        task=task,
        snapshot=snapshot,
        public_search_state={"search_stage": "initial"},
        max_items=4,
        max_bytes=4096,
        max_tokens=4096,
    )
    prompt = render_phase5_prompt(
        task=task,
        role=ProposalRole.TRANSFER,
        requested_batch_size=1,
        retrieval=retrieval,
        primitives=empty_primitive_registry(registry.content_hash, sha256_json({"plan": 1})),
    )
    value = json.loads(prompt)
    rendered = json.dumps(value, sort_keys=True)
    for forbidden in (
        "generator_family",
        "family_id",
        "reference_ast",
        "reference_rule",
        "semantic_hash",
        "oracle_handle",
        "artifact_path",
    ):
        assert forbidden not in rendered


def test_phase5_sealed_test_refuses_oracle_access_without_explicit_authority(
    tmp_path: Path,
) -> None:
    registry = load_transfer_registry(Path("configs/phase5-transfer-split-v1.yaml"))
    benchmark = generate_transfer_benchmark(tmp_path, registry)
    tasks = benchmark.manifest["tasks"]
    assert isinstance(tasks, list)
    sealed = next(item for item in tasks if isinstance(item, dict) and item["split"] == "test")
    store = Phase5TaskStore(benchmark.root)
    with pytest.raises(OracleVerificationError, match="explicit authority"):
        store.load(
            str(sealed["task_id"]),
            allowed_splits=frozenset({SplitLabel.TEST}),
            purpose="forbidden-test-access",
        )
    assert store.accesses == ()


def test_phase5_retrieval_fails_when_bounds_cannot_hold_explicit_empty_block(
    tmp_path: Path,
) -> None:
    registry = load_transfer_registry(Path("configs/phase5-transfer-split-v1.yaml"))
    benchmark = generate_transfer_benchmark(tmp_path, registry)
    tasks = benchmark.manifest["tasks"]
    assert isinstance(tasks, list)
    task_id = next(str(item["task_id"]) for item in tasks if isinstance(item, dict))
    task = load_transfer_public_task(benchmark.root, task_id).public_view()
    snapshot = MemorySnapshot(registry.content_hash, sha256_json({"empty": True}), ())
    with pytest.raises(ValueError, match="explicit empty memory block"):
        retrieve_memory(
            task=task,
            snapshot=snapshot,
            public_search_state={},
            max_items=0,
            max_bytes=0,
            max_tokens=0,
        )


def test_phase5_retrieval_is_stable_bounded_and_scope_aware(tmp_path: Path) -> None:
    registry = load_transfer_registry(Path("configs/phase5-transfer-split-v1.yaml"))
    benchmark = generate_transfer_benchmark(tmp_path, registry)
    tasks = benchmark.manifest["tasks"]
    assert isinstance(tasks, list)
    task_id = next(str(item["task_id"]) for item in tasks if isinstance(item, dict))
    task = load_transfer_public_task(benchmark.root, task_id).public_view()
    applicability = MemoryApplicability("elementary-public-world-v1", "binary-ca-radius1-dsl-v1")
    eligible_a = SafeMemoryItem(
        sha256_json({"item": "a"}),
        MemoryKind.SEARCH_LESSON,
        "Prefer exact typed compact programs.",
        "global-f0",
        applicability,
    )
    eligible_b = SafeMemoryItem(
        sha256_json({"item": "b"}),
        MemoryKind.HYPOTHESIS,
        "Prefer exact typed compact programs.",
        "global-f0",
        applicability,
    )
    ineligible = SafeMemoryItem(
        sha256_json({"item": "c"}),
        MemoryKind.EPISODIC,
        "This record has an inapplicable public scope.",
        "world:another-public-world-v1",
        applicability,
    )
    snapshot = MemorySnapshot(
        registry.content_hash,
        sha256_json({"export": 1}),
        tuple(sorted((eligible_a, eligible_b, ineligible), key=lambda item: item.record_id)),
    )
    first = retrieve_memory(
        task=task,
        snapshot=snapshot,
        public_search_state={"search_stage": "initial"},
        max_items=1,
        max_bytes=4096,
        max_tokens=4096,
    )
    second = retrieve_memory(
        task=task,
        snapshot=snapshot,
        public_search_state={"search_stage": "initial"},
        max_items=1,
        max_bytes=4096,
        max_tokens=4096,
    )
    assert first == second
    assert len(first.selected_record_ids) == 1
    assert first.selected_record_ids[0] == min(eligible_a.record_id, eligible_b.record_id)
    assert (ineligible.record_id, "scope-not-publicly-applicable") in first.exclusions
    assert first.rendered_bytes <= 4096
