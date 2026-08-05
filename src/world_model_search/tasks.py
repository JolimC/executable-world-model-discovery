"""Deterministic Phase 1 benchmark generation with capability-separated artifacts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from world_model_search.config import AppConfig
from world_model_search.domain.types import SplitLabel
from world_model_search.errors import ConfigurationError
from world_model_search.oracle.elementary import (
    ElementaryRule,
    independent_rollout_matches,
    local_errors,
    rollout,
)
from world_model_search.persistence.artifacts import write_text_exclusive
from world_model_search.serialization import canonical_json, derive_seed, sha256_json

GENERATOR_VERSION = "elementary-generator-v1"
ARTIFACT_VERSION = "phase1-task-bundle-v1"
SPLIT_VERSION = "semantic-shuffle-v1"
ANALYSIS_VERSION = "phase1-validation-v1"


@dataclass(frozen=True, slots=True)
class GeneratedBenchmark:
    root: Path
    manifest: dict[str, object]


def _bits(rng: random.Random, size: int) -> tuple[int, ...]:
    return tuple(rng.randrange(2) for _ in range(size))


def generate_benchmark(repository_root: Path, config: AppConfig) -> GeneratedBenchmark:
    """Generate all semantics once; test outcomes are deliberately never computed."""

    root = repository_root / config.run.root.parent / "phase1-benchmark"
    if root.exists():
        manifest_path = root / "manifest.json"
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"invalid existing benchmark: {root}") from exc
        if not isinstance(existing, dict) or existing.get("master_seed") != config.run.seed:
            raise ConfigurationError(f"existing benchmark is incompatible: {root}")
        existing_tasks = existing.get("tasks")
        if not isinstance(existing_tasks, list) or len(existing_tasks) != 256:
            raise ConfigurationError(f"existing benchmark is incomplete: {root}")
        for raw_record in existing_tasks:
            if not isinstance(raw_record, dict) or not isinstance(raw_record.get("task_id"), str):
                raise ConfigurationError(f"invalid task record in benchmark: {root}")
            task_id = raw_record["task_id"]
            try:
                public = json.loads((root / "public" / f"{task_id}.json").read_text())
                hidden = json.loads((root / "oracle" / f"{task_id}.json").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigurationError(f"missing or invalid task artifact: {task_id}") from exc
            if sha256_json(public) != raw_record.get("public_hash") or sha256_json(
                hidden
            ) != raw_record.get("hidden_hash"):
                raise ConfigurationError(f"task artifact content check failed: {task_id}")
        return GeneratedBenchmark(root=root, manifest=existing)
    public_dir, hidden_dir = root / "public", root / "oracle"
    public_dir.mkdir(parents=True)
    hidden_dir.mkdir()
    order = list(range(256))
    random.Random(derive_seed(config.run.seed, SPLIT_VERSION)).shuffle(order)
    split_for: dict[int, SplitLabel] = {}
    labels = (SplitLabel.TRAINING, SplitLabel.DEVELOPMENT, SplitLabel.VALIDATION, SplitLabel.TEST)
    for position, number in enumerate(order):
        split_for[number] = labels[position % 4]
    records: list[dict[str, object]] = []
    validation_records: list[dict[str, object]] = []
    for number in range(256):
        rule = ElementaryRule(number)
        # Public identity follows shuffled assignment position, never the rule number.
        position = order.index(number)
        task_seed = derive_seed(config.run.seed, f"{GENERATOR_VERSION}:task-slot:{position}")
        task_id = sha256_json({"domain": "opaque-task-id-v1", "seed": task_seed})[:24]
        demo_rng = random.Random(derive_seed(task_seed, "public-demonstrations-v1"))
        demos: list[dict[str, object]] = []
        for _ in range(3):
            state = _bits(demo_rng, 17)
            demos.append(
                {
                    "observation": "".join(map(str, state)),
                    "successor": "".join(map(str, rollout(rule, state, 1)[1])),
                }
            )
        public = {
            "artifact_version": ARTIFACT_VERSION,
            "task_id": task_id,
            "split": split_for[number].value,
            "world": {
                "dimension": 1,
                "alphabet": [0, 1],
                "radius": 1,
                "neighborhood_order": ["left", "center", "right"],
                "boundary": "periodic",
                "update": "synchronous",
                "encoding": "binary-string",
            },
            "demonstrations": demos,
            "active_queries_enabled": False,
        }
        public_text = canonical_json(public)
        public_hash = sha256_json(public)
        rollout_seed = derive_seed(task_seed, "locked-rollout-case:0")
        initial_seed = derive_seed(rollout_seed, "initial-state")
        initial = _bits(random.Random(initial_seed), 19)
        trajectory = rollout(rule, initial, 6)
        hidden = {
            "artifact_version": ARTIFACT_VERSION,
            "task_id": task_id,
            "reference_rule": number,
            "ordered_semantics_000_to_111": rule.ordered_semantics,
            "semantic_hash": rule.semantic_hash,
            "internal_family": "elementary-radius1-binary",
            "seeds": {
                "task": task_seed,
                "rollout_case": rollout_seed,
                "rollout_initial_state": initial_seed,
            },
            "locked_rollout": {"horizon": 6, "states": trajectory},
        }
        hidden_text = canonical_json(hidden)
        write_text_exclusive(public_dir / f"{task_id}.json", public_text + "\n")
        write_text_exclusive(hidden_dir / f"{task_id}.json", hidden_text + "\n")
        record: dict[str, object] = {
            "task_id": task_id,
            "split": split_for[number].value,
            "public_hash": public_hash,
            "hidden_hash": sha256_json(hidden),
            "semantic_hash": rule.semantic_hash,
            "task_seed": task_seed,
        }
        records.append(record)
        if split_for[number] == SplitLabel.VALIDATION:
            reference_pass = not local_errors(rule, rule)
            mutation_failures = sum(
                bool(local_errors(ElementaryRule(number ^ (1 << bit)), rule)) for bit in range(8)
            )
            rollout_pass = independent_rollout_matches(rule, trajectory)
            validation_records.append(
                {
                    "task_id": task_id,
                    "semantic_hash": rule.semantic_hash,
                    "local_reference_pass": reference_pass,
                    "one_bit_mutations_failed": mutation_failures,
                    "rollout_reference_pass": rollout_pass,
                }
            )
    manifest: dict[str, object] = {
        "manifest_schema_version": 3,
        "artifact_version": ARTIFACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split_version": SPLIT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "simulator_version": "elementary-ca-scalar-vector-v1",
        "oracle_version": "elementary-exact-v1",
        "rollout_version": "elementary-rollout-independent-v1",
        "master_seed": config.run.seed,
        "split_seed": derive_seed(config.run.seed, SPLIT_VERSION),
        "tasks": records,
        "validation": {
            "consumed": True,
            "task_count": len(validation_records),
            "results": validation_records,
        },
        "test_outcomes_accessed": False,
    }
    write_text_exclusive(root / "manifest.json", canonical_json(manifest) + "\n")
    report = {
        "source": "frozen-benchmark-artifacts-only",
        "counts_by_split": {
            label.value: sum(r["split"] == label.value for r in records) for label in labels
        },
        "semantic_duplicates": 0,
        "validation_tasks": len(validation_records),
        "local_reference_passes": len(validation_records),
        "one_bit_mutation_failures": 8 * len(validation_records),
        "rollout_reference_passes": len(validation_records),
        "test_outcomes_accessed": False,
        "validation_consumed": True,
    }
    write_text_exclusive(root / "validation-report.json", canonical_json(report) + "\n")
    return GeneratedBenchmark(root=root, manifest=manifest)


def load_hidden_task(benchmark_root: Path, task_id: str) -> dict[str, object]:
    """Oracle-authority API; proposer code must never call this function."""

    value = json.loads((benchmark_root / "oracle" / f"{task_id}.json").read_text())
    if not isinstance(value, dict):
        raise ValueError("hidden task is not an object")
    return value
