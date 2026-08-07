"""Deterministic Phase 1 benchmark generation with capability-separated artifacts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from world_model_search.config import AppConfig
from world_model_search.domain.types import (
    ElementaryPublicWorldSpec,
    PublicDemonstration,
    PublicTask,
    SplitLabel,
    Task,
)
from world_model_search.dsl.versions import (
    CANDIDATE_SCHEMA_VERSION,
    CANONICALIZER_VERSION,
    DSL_VERSION,
    PREFIX_CODE_VERSION,
)
from world_model_search.errors import ConfigurationError, OracleVerificationError
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
PHASE2_GENERATOR_VERSION = "elementary-generator-phase2-public-contract-v1"
PHASE2_ARTIFACT_VERSION = "phase2-task-bundle-v1"
PHASE2_MANIFEST_SCHEMA_VERSION = 4


@dataclass(frozen=True, slots=True)
class GeneratedBenchmark:
    root: Path
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class LockedRollout:
    horizon: int
    states: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class HiddenTaskBundle:
    """Typed oracle-only task data; never accepted by proposer APIs."""

    artifact_version: str
    task_id: str
    reference_rule: ElementaryRule
    ordered_semantics: tuple[int, ...]
    semantic_hash: str
    internal_family: str
    task_seed: int
    rollout_case_seed: int
    rollout_initial_state_seed: int
    locked_rollout: LockedRollout


@dataclass(frozen=True, slots=True)
class OracleTaskAccess:
    task_id: str
    split: SplitLabel
    purpose: str


def benchmark_root_for_config(repository_root: Path, config: AppConfig) -> Path:
    return repository_root / config.run.root.parent / "phase2-benchmark"


def generate_phase2_benchmark(repository_root: Path, config: AppConfig) -> GeneratedBenchmark:
    """Generate Phase 2 bundles whose public traces cannot encode all eight local cases."""

    root = repository_root / config.run.root.parent / "phase2-benchmark"
    if root.exists():
        manifest_path = root / "manifest.json"
        try:
            existing: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"invalid existing Phase 2 benchmark: {root}") from exc
        if (
            not isinstance(existing, dict)
            or existing.get("master_seed") != config.run.seed
            or existing.get("artifact_version") != PHASE2_ARTIFACT_VERSION
        ):
            raise ConfigurationError(f"existing Phase 2 benchmark is incompatible: {root}")
        existing_records = existing.get("tasks")
        if not isinstance(existing_records, list) or len(existing_records) != 256:
            raise ConfigurationError(f"existing Phase 2 benchmark is incomplete: {root}")
        for raw_record in existing_records:
            if not isinstance(raw_record, dict) or not isinstance(raw_record.get("task_id"), str):
                raise ConfigurationError(f"invalid Phase 2 task record: {root}")
            task_id = raw_record["task_id"]
            try:
                public = json.loads((root / "public" / f"{task_id}.json").read_text())
                hidden = json.loads((root / "oracle" / f"{task_id}.json").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigurationError(f"missing Phase 2 task artifact: {task_id}") from exc
            if sha256_json(public) != raw_record.get("public_hash") or sha256_json(
                hidden
            ) != raw_record.get("hidden_hash"):
                raise ConfigurationError(f"Phase 2 task content check failed: {task_id}")
        return GeneratedBenchmark(root=root, manifest=existing)
    public_dir, hidden_dir = root / "public", root / "oracle"
    public_dir.mkdir(parents=True)
    hidden_dir.mkdir()
    order = list(range(256))
    random.Random(derive_seed(config.run.seed, SPLIT_VERSION)).shuffle(order)
    labels = (SplitLabel.TRAINING, SplitLabel.DEVELOPMENT, SplitLabel.VALIDATION, SplitLabel.TEST)
    split_for = {number: labels[position % 4] for position, number in enumerate(order)}
    records: list[dict[str, object]] = []
    for number in range(256):
        rule = ElementaryRule(number)
        position = order.index(number)
        task_seed = derive_seed(config.run.seed, f"{GENERATOR_VERSION}:task-slot:{position}")
        task_id = sha256_json({"domain": "opaque-task-id-v1", "seed": task_seed})[:24]
        demonstrations: list[dict[str, object]] = []
        for cell in (0, 1):
            state = (cell,) * 17
            demonstrations.append(
                {
                    "observation": "".join(map(str, state)),
                    "successor": "".join(map(str, rollout(rule, state, 1)[1])),
                }
            )
        public = {
            "artifact_version": PHASE2_ARTIFACT_VERSION,
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
            "demonstrations": demonstrations,
            "active_queries_enabled": False,
            "public_local_case_coverage": ["000", "111"],
        }
        rollout_seed = derive_seed(task_seed, "locked-rollout-case:0")
        initial_seed = derive_seed(rollout_seed, "initial-state")
        initial = _bits(random.Random(initial_seed), 19)
        hidden = {
            "artifact_version": PHASE2_ARTIFACT_VERSION,
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
            "locked_rollout": {"horizon": 6, "states": rollout(rule, initial, 6)},
        }
        write_text_exclusive(public_dir / f"{task_id}.json", canonical_json(public) + "\n")
        write_text_exclusive(hidden_dir / f"{task_id}.json", canonical_json(hidden) + "\n")
        records.append(
            {
                "task_id": task_id,
                "split": split_for[number].value,
                "public_hash": sha256_json(public),
                "hidden_hash": sha256_json(hidden),
                "semantic_hash": rule.semantic_hash,
                "task_seed": task_seed,
            }
        )
    manifest: dict[str, object] = {
        "manifest_schema_version": PHASE2_MANIFEST_SCHEMA_VERSION,
        "artifact_version": PHASE2_ARTIFACT_VERSION,
        "generator_version": PHASE2_GENERATOR_VERSION,
        "split_version": SPLIT_VERSION,
        "simulator_version": "elementary-ca-scalar-vector-v1",
        "oracle_version": "typed-elementary-exact-v1",
        "rollout_version": "elementary-rollout-independent-v1",
        "master_seed": config.run.seed,
        "split_seed": derive_seed(config.run.seed, SPLIT_VERSION),
        "public_trace_contract": "uniform-000-and-111-only-v1",
        "validation": {"consumed": False, "task_count": 0, "results": []},
        "test_outcomes_accessed": False,
        "tasks": records,
    }
    write_text_exclusive(root / "manifest.json", canonical_json(manifest) + "\n")
    return GeneratedBenchmark(root=root, manifest=manifest)


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


def _mapping(value: object, expected: set[str], location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise OracleVerificationError(f"{location} must be an object")
    if set(value) != expected:
        raise OracleVerificationError(f"{location} has missing or unknown fields")
    return value


def _integer(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OracleVerificationError(f"{location} must be an integer >= {minimum}")
    return value


def _binary_tuple(value: object, location: str, *, length: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or item not in (0, 1) for item in value
    ):
        raise OracleVerificationError(f"{location} must be an array of integer bits")
    result = tuple(value)
    if length is not None and len(result) != length:
        raise OracleVerificationError(f"{location} must contain {length} bits")
    return result


def _read_benchmark_manifest(benchmark_root: Path) -> dict[str, object]:
    try:
        raw: object = json.loads((benchmark_root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleVerificationError("benchmark manifest is missing or invalid") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("manifest_schema_version") != PHASE2_MANIFEST_SCHEMA_VERSION
        or raw.get("artifact_version") != PHASE2_ARTIFACT_VERSION
    ):
        raise OracleVerificationError("unsupported benchmark manifest")
    return raw


def _task_record(benchmark_root: Path, task_id: str) -> dict[str, object]:
    if len(task_id) != 24 or any(character not in "0123456789abcdef" for character in task_id):
        raise OracleVerificationError("task id must be 24 lowercase hexadecimal characters")
    tasks = _read_benchmark_manifest(benchmark_root).get("tasks")
    if not isinstance(tasks, list):
        raise OracleVerificationError("benchmark manifest has no task records")
    matches = [
        record for record in tasks if isinstance(record, dict) and record.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise OracleVerificationError("task is unavailable")
    return matches[0]


def load_public_task(benchmark_root: Path, task_id: str) -> Task:
    """Load and type-check proposer-visible mechanics without opening oracle data."""

    record = _task_record(benchmark_root, task_id)
    try:
        raw: object = json.loads(
            (benchmark_root / "public" / f"{task_id}.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleVerificationError("public task artifact is missing or invalid") from exc
    public = _mapping(
        raw,
        {
            "artifact_version",
            "task_id",
            "split",
            "world",
            "demonstrations",
            "active_queries_enabled",
            "public_local_case_coverage",
        },
        "public task",
    )
    if public["artifact_version"] != PHASE2_ARTIFACT_VERSION or public["task_id"] != task_id:
        raise OracleVerificationError("public task version or identity mismatch")
    if sha256_json(public) != record.get("public_hash"):
        raise OracleVerificationError("public task content hash mismatch")
    split_raw = public["split"]
    try:
        if not isinstance(split_raw, str):
            raise ValueError
        split = SplitLabel(split_raw)
    except (TypeError, ValueError) as exc:
        raise OracleVerificationError("public task split is invalid") from exc
    world = _mapping(
        public["world"],
        {
            "dimension",
            "alphabet",
            "radius",
            "neighborhood_order",
            "boundary",
            "update",
            "encoding",
        },
        "public world",
    )
    if world != {
        "dimension": 1,
        "alphabet": [0, 1],
        "radius": 1,
        "neighborhood_order": ["left", "center", "right"],
        "boundary": "periodic",
        "update": "synchronous",
        "encoding": "binary-string",
    }:
        raise OracleVerificationError("task mechanics are outside the Phase 2 world boundary")
    demonstrations_raw = public["demonstrations"]
    if not isinstance(demonstrations_raw, list):
        raise OracleVerificationError("public demonstrations must be an array")
    demonstrations: list[PublicDemonstration] = []
    for raw_demo in demonstrations_raw:
        demo = _mapping(raw_demo, {"observation", "successor"}, "public demonstration")
        observation, successor = demo["observation"], demo["successor"]
        if (
            not isinstance(observation, str)
            or not isinstance(successor, str)
            or not observation
            or len(observation) != len(successor)
            or set(observation + successor) - {"0", "1"}
        ):
            raise OracleVerificationError("public demonstration strings are invalid")
        demonstrations.append(PublicDemonstration(observation=observation, successor=successor))
    if public["active_queries_enabled"] is not False:
        raise OracleVerificationError("active queries must remain disabled in Phase 2")
    if public["public_local_case_coverage"] != ["000", "111"]:
        raise OracleVerificationError("Phase 2 public trace coverage contract is invalid")
    specification = ElementaryPublicWorldSpec(
        specification_version="elementary-public-world-v1",
        dimension=1,
        alphabet=(0, 1),
        radius=1,
        neighborhood_order=("left", "center", "right"),
        offsets=(-1, 0, 1),
        boundary="periodic",
        update="synchronous",
        observation_type="binary-string-v1",
        successor_type="binary-string-v1",
        candidate_type="typed-json-ast-v1",
        candidate_schema_version=CANDIDATE_SCHEMA_VERSION,
        dsl_version=DSL_VERSION,
        bit_constructors=(
            "Const",
            "At",
            "Not",
            "And",
            "Or",
            "Xor",
            "If",
            "TruthTable",
        ),
        int_constructors=("IntConst", "Count", "AddConst"),
        predicate_constructors=("Eq", "Le", "Ge", "Between"),
        macros=("Parity", "Majority"),
        max_depth=8,
        max_nodes=63,
        canonicalizer_version=CANONICALIZER_VERSION,
        prefix_code_version=PREFIX_CODE_VERSION,
    )
    public_task = PublicTask(
        task_id=task_id,
        public_world_spec=specification,
        split=split,
        demonstrations=tuple(demonstrations),
        active_queries_enabled=False,
        query_budget=0,
    )
    task_seed = _integer(record.get("task_seed"), "task record seed")
    return Task(
        task_id=task_id,
        internal_family_id="elementary-radius1-binary",
        public_world_spec=specification,
        split=split,
        public_demonstrations=public_task.demonstrations,
        active_queries_enabled=False,
        query_budget=0,
        exact_case_set_id="elementary-exhaustive-8-v1",
        rollout_suite_id="elementary-locked-rollout-v1",
        public_artifact_hash=str(record["public_hash"]),
        hidden_artifact_id=task_id,
        generator_version=PHASE2_GENERATOR_VERSION,
        seed=task_seed,
    )


def _parse_hidden_task(value: object, task_id: str) -> HiddenTaskBundle:
    hidden = _mapping(
        value,
        {
            "artifact_version",
            "task_id",
            "reference_rule",
            "ordered_semantics_000_to_111",
            "semantic_hash",
            "internal_family",
            "seeds",
            "locked_rollout",
        },
        "hidden task",
    )
    if hidden["artifact_version"] != PHASE2_ARTIFACT_VERSION or hidden["task_id"] != task_id:
        raise OracleVerificationError("hidden task version or identity mismatch")
    number = _integer(hidden["reference_rule"], "reference rule")
    try:
        rule = ElementaryRule(number)
    except ValueError as exc:
        raise OracleVerificationError("reference rule is invalid") from exc
    semantics = _binary_tuple(
        hidden["ordered_semantics_000_to_111"], "reference semantics", length=8
    )
    semantic_digest = hidden["semantic_hash"]
    if semantics != rule.ordered_semantics or semantic_digest != rule.semantic_hash:
        raise OracleVerificationError("hidden semantic identity is inconsistent")
    if hidden["internal_family"] != "elementary-radius1-binary":
        raise OracleVerificationError("hidden task family is unsupported")
    seeds = _mapping(
        hidden["seeds"],
        {"task", "rollout_case", "rollout_initial_state"},
        "hidden seeds",
    )
    locked = _mapping(hidden["locked_rollout"], {"horizon", "states"}, "locked rollout")
    horizon = _integer(locked["horizon"], "locked rollout horizon")
    states_raw = locked["states"]
    if not isinstance(states_raw, list):
        raise OracleVerificationError("locked rollout states must be an array")
    states = tuple(_binary_tuple(state, "locked rollout state") for state in states_raw)
    if len(states) != horizon + 1 or not states or not states[0]:
        raise OracleVerificationError("locked rollout horizon or state count is invalid")
    if len({len(state) for state in states}) != 1 or not independent_rollout_matches(rule, states):
        raise OracleVerificationError("locked rollout failed its independent integrity check")
    return HiddenTaskBundle(
        artifact_version=PHASE2_ARTIFACT_VERSION,
        task_id=task_id,
        reference_rule=rule,
        ordered_semantics=semantics,
        semantic_hash=str(semantic_digest),
        internal_family="elementary-radius1-binary",
        task_seed=_integer(seeds["task"], "hidden task seed"),
        rollout_case_seed=_integer(seeds["rollout_case"], "hidden rollout seed"),
        rollout_initial_state_seed=_integer(
            seeds["rollout_initial_state"], "hidden rollout initial-state seed"
        ),
        locked_rollout=LockedRollout(horizon=horizon, states=states),
    )


class HiddenTaskStore:
    """Oracle authority that enforces split policy and records every authorized load."""

    def __init__(self, benchmark_root: Path) -> None:
        self.benchmark_root = benchmark_root
        self._accesses: list[OracleTaskAccess] = []

    @property
    def accesses(self) -> tuple[OracleTaskAccess, ...]:
        return tuple(self._accesses)

    def load(
        self,
        task_id: str,
        *,
        allowed_splits: frozenset[SplitLabel],
        purpose: str,
    ) -> HiddenTaskBundle:
        record = _task_record(self.benchmark_root, task_id)
        split_raw = record.get("split")
        try:
            if not isinstance(split_raw, str):
                raise ValueError
            split = SplitLabel(split_raw)
        except (TypeError, ValueError) as exc:
            raise OracleVerificationError("task split metadata is invalid") from exc
        if split not in allowed_splits:
            raise OracleVerificationError("task split is not authorized for this operation")
        try:
            raw_text = (self.benchmark_root / "oracle" / f"{task_id}.json").read_text(
                encoding="utf-8"
            )
            raw: object = json.loads(raw_text)
        except (OSError, json.JSONDecodeError) as exc:
            raise OracleVerificationError("oracle task artifact is unavailable") from exc
        if sha256_json(raw) != record.get("hidden_hash"):
            raise OracleVerificationError("oracle task content hash mismatch")
        bundle = _parse_hidden_task(raw, task_id)
        self._accesses.append(OracleTaskAccess(task_id=task_id, split=split, purpose=purpose))
        return bundle


def load_hidden_task(benchmark_root: Path, task_id: str) -> HiddenTaskBundle:
    """Legacy oracle-authority API; Phase 2 run code uses ``HiddenTaskStore`` policy."""

    record = _task_record(benchmark_root, task_id)
    split_raw = record.get("split")
    try:
        if not isinstance(split_raw, str):
            raise ValueError
        split = SplitLabel(split_raw)
    except (TypeError, ValueError) as exc:
        raise OracleVerificationError("task split metadata is invalid") from exc
    return HiddenTaskStore(benchmark_root).load(
        task_id,
        allowed_splits=frozenset({split}),
        purpose="explicit-oracle-load",
    )
