"""Versioned F0 structural-family transfer benchmark for Phase 5.

Family metadata and reference programs are evaluator-only.  Proposers receive only the
ordinary ``PublicTask`` view and the same F0 mechanics used in Phase 2-4.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from world_model_search.domain.types import (
    ElementaryPublicWorldSpec,
    PublicDemonstration,
    PublicTask,
    SplitLabel,
    Task,
)
from world_model_search.dsl.ast import (
    And,
    At,
    BitExpr,
    Count,
    Eq,
    If,
    IntConst,
    Not,
    Or,
    Xor,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.interpreter import semantic_hash, truth_table
from world_model_search.dsl.json_schema import DslCandidateDocument
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
    rollout,
)
from world_model_search.persistence.artifacts import write_text_exclusive
from world_model_search.phase5_versions import (
    PHASE5_TRANSFER_ARTIFACT_VERSION,
    PHASE5_TRANSFER_GENERATOR_VERSION,
    PHASE5_TRANSFER_MANIFEST_VERSION,
    PHASE5_TRANSFER_REGISTRY_VERSION,
)
from world_model_search.serialization import JsonObject, canonical_json, derive_seed, sha256_json
from world_model_search.tasks import HiddenTaskBundle, LockedRollout, OracleTaskAccess


@dataclass(frozen=True, slots=True)
class TransferFamily:
    family_id: str
    role: SplitLabel
    generator: str
    variants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransferRegistry:
    registry_id: str
    master_seed: int
    output_root: Path
    families: tuple[TransferFamily, ...]
    endpoints: tuple[str, ...]
    exclusions: tuple[str, ...]
    sealed_test_authorized: bool

    @property
    def content_hash(self) -> str:
        return sha256_json(self.to_value())

    def to_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "registry_version": PHASE5_TRANSFER_REGISTRY_VERSION,
                "registry_id": self.registry_id,
                "master_seed": self.master_seed,
                "output_root": str(self.output_root),
                "families": [
                    {
                        "family_id": family.family_id,
                        "role": family.role.value,
                        "generator": family.generator,
                        "variants": list(family.variants),
                    }
                    for family in self.families
                ],
                "endpoints": list(self.endpoints),
                "exclusions": list(self.exclusions),
                "sealed_test_authorized": self.sealed_test_authorized,
            },
        )


@dataclass(frozen=True, slots=True)
class GeneratedTransferBenchmark:
    root: Path
    manifest: JsonObject


@dataclass(frozen=True, slots=True)
class Phase5HiddenTask:
    task_id: str
    role: SplitLabel
    family_id: str
    reference_ast: BitExpr
    semantic_hash: str
    oracle_bundle: HiddenTaskBundle


_GENERATORS = frozenset(
    {
        "selector-source-xor-v1",
        "selector-source-and-v1",
        "selector-target-or-v1",
        "selector-target-not-xor-v1",
        "selector-test-not-and-v1",
        "selector-test-gated-v1",
    }
)
_VARIANTS = frozenset({"left", "center", "right", "not-left", "not-center", "not-right"})


def _mapping(value: object, expected: set[str], location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping")
    if set(value) != expected:
        raise ConfigurationError(f"{location} has missing or unknown keys")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{location} must be a nonempty string")
    return value


def _strings(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ConfigurationError(f"{location} must be a nonempty string list")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ConfigurationError(f"{location} contains duplicates")
    return result


def load_transfer_registry(path: Path) -> TransferRegistry:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("Phase 5 transfer registry is unavailable or invalid") from exc
    root = _mapping(
        value,
        {
            "registry_version",
            "registry_id",
            "master_seed",
            "output_root",
            "families",
            "endpoints",
            "exclusions",
            "sealed_test_authorized",
        },
        "transfer registry",
    )
    if root["registry_version"] != PHASE5_TRANSFER_REGISTRY_VERSION:
        raise ConfigurationError("unsupported Phase 5 transfer registry version")
    seed = root["master_seed"]
    output_root = Path(_string(root["output_root"], "output_root"))
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or output_root.is_absolute()
        or ".." in output_root.parts
    ):
        raise ConfigurationError("transfer seed/output root is invalid")
    families_raw = root["families"]
    if not isinstance(families_raw, list) or len(families_raw) < 6:
        raise ConfigurationError("transfer registry requires at least six structural families")
    families: list[TransferFamily] = []
    for index, item in enumerate(families_raw):
        raw = _mapping(item, {"family_id", "role", "generator", "variants"}, f"family {index}")
        try:
            role = SplitLabel(_string(raw["role"], f"family {index} role"))
        except ValueError as exc:
            raise ConfigurationError("transfer family role is invalid") from exc
        if role not in {SplitLabel.TRAINING, SplitLabel.DEVELOPMENT, SplitLabel.TEST}:
            raise ConfigurationError("Phase 5 transfer roles are training, development, or test")
        generator = _string(raw["generator"], f"family {index} generator")
        variants = _strings(raw["variants"], f"family {index} variants")
        if generator not in _GENERATORS or set(variants) - _VARIANTS:
            raise ConfigurationError("unknown structural generator or variant")
        families.append(
            TransferFamily(
                family_id=_string(raw["family_id"], f"family {index} id"),
                role=role,
                generator=generator,
                variants=variants,
            )
        )
    ids = tuple(family.family_id for family in families)
    if len(ids) != len(set(ids)):
        raise ConfigurationError("transfer family IDs must be unique")
    role_counts = {
        role: sum(family.role is role for family in families)
        for role in (SplitLabel.TRAINING, SplitLabel.DEVELOPMENT, SplitLabel.TEST)
    }
    if any(count < 2 for count in role_counts.values()):
        raise ConfigurationError("each Phase 5 role requires at least two whole families")
    authorized = root["sealed_test_authorized"]
    if not isinstance(authorized, bool) or authorized:
        raise ConfigurationError(
            "the Phase 5 transfer registry must leave sealed test unauthorized"
        )
    exclusions = _strings(root["exclusions"], "exclusions")
    required_exclusions = {
        "semantic-duplicates-across-all-roles",
        "phase2-all-256-analysis",
        "phase3-consumed-validation",
        "phase4-development-pilot",
    }
    if not required_exclusions <= set(exclusions):
        raise ConfigurationError("transfer exclusions do not protect consumed evidence")
    return TransferRegistry(
        registry_id=_string(root["registry_id"], "registry_id"),
        master_seed=seed,
        output_root=output_root,
        families=tuple(families),
        endpoints=_strings(root["endpoints"], "endpoints"),
        exclusions=exclusions,
        sealed_test_authorized=False,
    )


def selector_core() -> BitExpr:
    """The shared structural subtree; this is evaluator training material, not a built-in."""

    return canonicalize(
        If(
            Eq(Count((-1, 0, 1)), IntConst(1)),
            At(-1),
            At(1),
        )
    )


def _leaf(variant: str) -> BitExpr:
    negated = variant.startswith("not-")
    name = variant.removeprefix("not-")
    offset = {"left": -1, "center": 0, "right": 1}[name]
    value: BitExpr = At(offset)
    return Not(value) if negated else value


def reference_program(generator: str, variant: str) -> BitExpr:
    """Expand one predeclared structural grammar stratum without consulting outcomes."""

    core, leaf = selector_core(), _leaf(variant)
    if generator == "selector-source-xor-v1":
        raw: BitExpr = Xor(core, leaf)
    elif generator == "selector-source-and-v1":
        raw = And(core, leaf)
    elif generator == "selector-target-or-v1":
        raw = Or(core, leaf)
    elif generator == "selector-target-not-xor-v1":
        raw = Not(Xor(core, leaf))
    elif generator == "selector-test-not-and-v1":
        raw = Not(And(core, leaf))
    elif generator == "selector-test-gated-v1":
        raw = If(Eq(Count((-1, 0, 1)), IntConst(2)), core, leaf)
    else:  # pragma: no cover - registry validation makes this unreachable
        raise ConfigurationError("unknown reference generator")
    return canonicalize(raw)


def _public_spec() -> ElementaryPublicWorldSpec:
    return ElementaryPublicWorldSpec(
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
        bit_constructors=("Const", "At", "Not", "And", "Or", "Xor", "If", "TruthTable"),
        int_constructors=("IntConst", "Count", "AddConst"),
        predicate_constructors=("Eq", "Le", "Ge", "Between"),
        macros=("Parity", "Majority"),
        max_depth=8,
        max_nodes=63,
        canonicalizer_version=CANONICALIZER_VERSION,
        prefix_code_version=PREFIX_CODE_VERSION,
    )


def _benchmark_records(registry: TransferRegistry) -> list[tuple[TransferFamily, str, BitExpr]]:
    return [
        (family, variant, reference_program(family.generator, variant))
        for family in registry.families
        for variant in family.variants
    ]


def generate_transfer_benchmark(
    repository_root: Path, registry: TransferRegistry
) -> GeneratedTransferBenchmark:
    """Generate/verify immutable public and evaluator task artifacts.

    Semantic identities are computed only for split-disjointness proof and exact-oracle setup;
    no test candidate is evaluated and no test outcome is recorded.
    """

    root = repository_root / registry.output_root
    manifest_path = root / "manifest.json"
    if root.exists():
        try:
            manifest_value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError("existing Phase 5 benchmark is invalid") from exc
        if not isinstance(manifest_value, dict):
            raise ConfigurationError("existing Phase 5 benchmark manifest is not an object")
        existing_manifest = cast(JsonObject, manifest_value)
        if existing_manifest.get("registry_hash") != registry.content_hash:
            raise ConfigurationError("existing Phase 5 benchmark binds another registry")
        tasks = existing_manifest.get("tasks")
        if not isinstance(tasks, list):
            raise ConfigurationError("existing Phase 5 benchmark task index is invalid")
        for item in tasks:
            if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
                raise ConfigurationError("existing Phase 5 task index is invalid")
            task_id = item["task_id"]
            for directory, field in (("public", "public_hash"), ("evaluator", "hidden_hash")):
                try:
                    value = json.loads((root / directory / f"{task_id}.json").read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise ConfigurationError("existing Phase 5 task artifact is invalid") from exc
                if sha256_json(value) != item.get(field):
                    raise ConfigurationError("existing Phase 5 task artifact hash mismatch")
        return GeneratedTransferBenchmark(root, existing_manifest)

    records = _benchmark_records(registry)
    semantics: dict[str, tuple[str, str, SplitLabel]] = {}
    prepared: list[tuple[TransferFamily, str, BitExpr, tuple[int, ...], str]] = []
    for family, variant, ast in records:
        table = truth_table(ast)
        digest = semantic_hash(ast)
        if digest in semantics:
            prior_family, prior_variant, prior_role = semantics[digest]
            raise ConfigurationError(
                "family protocol gate: semantic duplicate across roles/families: "
                f"{prior_family}/{prior_variant}/{prior_role.value} and "
                f"{family.family_id}/{variant}/{family.role.value}"
            )
        semantics[digest] = (family.family_id, variant, family.role)
        prepared.append((family, variant, ast, table, digest))

    public_dir, hidden_dir = root / "public", root / "evaluator"
    public_dir.mkdir(parents=True)
    hidden_dir.mkdir()
    manifest_tasks: list[JsonObject] = []
    for family, variant, ast, table, digest in prepared:
        seed = derive_seed(
            registry.master_seed,
            f"{PHASE5_TRANSFER_GENERATOR_VERSION}:{family.family_id}:{variant}",
        )
        task_id = sha256_json({"domain": "phase5-opaque-task-id-v1", "seed": seed})[:24]
        rule_number = sum(bit << index for index, bit in enumerate(table))
        rule = ElementaryRule(rule_number)
        demos: list[JsonObject] = []
        for cell in (0, 1):
            state = (cell,) * 17
            demos.append(
                {
                    "observation": "".join(map(str, state)),
                    "successor": "".join(map(str, rollout(rule, state, 1)[1])),
                }
            )
        public = cast(
            JsonObject,
            {
                "artifact_version": PHASE5_TRANSFER_ARTIFACT_VERSION,
                "task_id": task_id,
                "split": family.role.value,
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
                "public_local_case_coverage": ["000", "111"],
            },
        )
        initial_seed = derive_seed(seed, "phase5-locked-rollout-initial-v1")
        initial = tuple(random.Random(initial_seed).randrange(2) for _ in range(19))
        trajectory = rollout(rule, initial, 6)
        hidden: JsonObject = {
            "artifact_version": PHASE5_TRANSFER_ARTIFACT_VERSION,
            "task_id": task_id,
            "generator_family": family.family_id,
            "generator_stratum": family.generator,
            "variant": variant,
            "split": family.role.value,
            "reference_ast": json.loads(DslCandidateDocument(ast).to_json()),
            "reference_rule": rule_number,
            "ordered_semantics_000_to_111": list(table),
            "semantic_hash": digest,
            "task_seed": seed,
            "locked_rollout": {
                "initial_state_seed": initial_seed,
                "horizon": 6,
                "states": [list(state) for state in trajectory],
            },
        }
        write_text_exclusive(public_dir / f"{task_id}.json", canonical_json(public) + "\n")
        write_text_exclusive(hidden_dir / f"{task_id}.json", canonical_json(hidden) + "\n")
        manifest_tasks.append(
            {
                "task_id": task_id,
                "family_id": family.family_id,
                "generator_stratum": family.generator,
                "variant": variant,
                "split": family.role.value,
                "semantic_hash": digest,
                "public_hash": sha256_json(public),
                "hidden_hash": sha256_json(hidden),
            }
        )
    manifest = cast(
        JsonObject,
        {
            "manifest_schema_version": PHASE5_TRANSFER_MANIFEST_VERSION,
            "artifact_version": PHASE5_TRANSFER_ARTIFACT_VERSION,
            "generator_version": PHASE5_TRANSFER_GENERATOR_VERSION,
            "registry_id": registry.registry_id,
            "registry_hash": registry.content_hash,
            "family_split_policy": "whole-structural-generator-family-v1",
            "semantic_disjointness_proof": {
                "algorithm": "complete-eight-case-semantic-hash-pairwise-v1",
                "task_count": len(manifest_tasks),
                "unique_semantic_count": len(semantics),
                "duplicates": [],
            },
            "consumed_evidence_excluded": list(registry.exclusions),
            "test_outcomes_accessed": False,
            "tasks": manifest_tasks,
        },
    )
    write_text_exclusive(manifest_path, canonical_json(manifest) + "\n")
    return GeneratedTransferBenchmark(root, manifest)


def _task_record(root: Path, task_id: str) -> dict[str, object]:
    if len(task_id) != 24 or set(task_id) - set("0123456789abcdef"):
        raise OracleVerificationError("Phase 5 task id is malformed")
    try:
        value: object = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleVerificationError("Phase 5 benchmark manifest is unavailable") from exc
    if not isinstance(value, dict) or value.get("manifest_schema_version") != 1:
        raise OracleVerificationError("unsupported Phase 5 benchmark manifest")
    tasks = value.get("tasks")
    if not isinstance(tasks, list):
        raise OracleVerificationError("Phase 5 benchmark task index is unavailable")
    matches = [item for item in tasks if isinstance(item, dict) and item.get("task_id") == task_id]
    if len(matches) != 1:
        raise OracleVerificationError("Phase 5 task is unavailable")
    return matches[0]


def load_transfer_public_task(root: Path, task_id: str) -> Task:
    record = _task_record(root, task_id)
    try:
        value: object = json.loads((root / "public" / f"{task_id}.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleVerificationError("Phase 5 public task is unavailable") from exc
    if not isinstance(value, dict) or sha256_json(value) != record.get("public_hash"):
        raise OracleVerificationError("Phase 5 public task hash mismatch")
    allowed = {
        "artifact_version",
        "task_id",
        "split",
        "world",
        "demonstrations",
        "active_queries_enabled",
        "public_local_case_coverage",
    }
    if set(value) != allowed or value.get("artifact_version") != PHASE5_TRANSFER_ARTIFACT_VERSION:
        raise OracleVerificationError("Phase 5 public task schema mismatch")
    try:
        split = SplitLabel(str(value["split"]))
    except ValueError as exc:
        raise OracleVerificationError("Phase 5 public task split is invalid") from exc
    demos_raw = value.get("demonstrations")
    if not isinstance(demos_raw, list):
        raise OracleVerificationError("Phase 5 public demonstrations are invalid")
    demonstrations: list[PublicDemonstration] = []
    for item in demos_raw:
        if not isinstance(item, dict) or set(item) != {"observation", "successor"}:
            raise OracleVerificationError("Phase 5 public demonstration schema mismatch")
        observation, successor = item["observation"], item["successor"]
        if not isinstance(observation, str) or not isinstance(successor, str):
            raise OracleVerificationError("Phase 5 public demonstration is invalid")
        demonstrations.append(PublicDemonstration(observation, successor))
    public = PublicTask(task_id, _public_spec(), split, tuple(demonstrations), False, 0)
    return Task(
        task_id=task_id,
        internal_family_id=str(record["family_id"]),
        public_world_spec=public.public_world_spec,
        split=split,
        public_demonstrations=public.demonstrations,
        active_queries_enabled=False,
        query_budget=0,
        exact_case_set_id="elementary-exhaustive-8-v1",
        rollout_suite_id="phase5-locked-rollout-v1",
        public_artifact_hash=str(record["public_hash"]),
        hidden_artifact_id=task_id,
        generator_version=PHASE5_TRANSFER_GENERATOR_VERSION,
        seed=derive_seed(0, task_id),
    )


class Phase5TaskStore:
    """Evaluator authority with an explicit sealed-test refusal."""

    def __init__(self, root: Path) -> None:
        self.root = root
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
        sealed_test_authority: str | None = None,
    ) -> Phase5HiddenTask:
        record = _task_record(self.root, task_id)
        try:
            role = SplitLabel(str(record["split"]))
        except ValueError as exc:
            raise OracleVerificationError("Phase 5 task role is invalid") from exc
        if role not in allowed_splits:
            raise OracleVerificationError("Phase 5 family role is not authorized")
        if role is SplitLabel.TEST and sealed_test_authority is None:
            raise OracleVerificationError("sealed Phase 5 test access requires explicit authority")
        try:
            value: object = json.loads((self.root / "evaluator" / f"{task_id}.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise OracleVerificationError("Phase 5 evaluator task is unavailable") from exc
        if not isinstance(value, dict) or sha256_json(value) != record.get("hidden_hash"):
            raise OracleVerificationError("Phase 5 evaluator task hash mismatch")
        document_raw = value.get("reference_ast")
        if not isinstance(document_raw, dict):
            raise OracleVerificationError("Phase 5 reference AST is unavailable")
        try:
            ast = DslCandidateDocument.from_json(canonical_json(document_raw)).ast
            table_raw = value["ordered_semantics_000_to_111"]
            states_raw = cast(dict[str, object], value["locked_rollout"])["states"]
            if not isinstance(table_raw, list) or not isinstance(states_raw, list):
                raise ValueError
            if any(isinstance(bit, bool) or not isinstance(bit, int) for bit in table_raw):
                raise ValueError
            table = tuple(cast(int, bit) for bit in table_raw)
            if any(
                not isinstance(state, list)
                or any(isinstance(bit, bool) or not isinstance(bit, int) for bit in state)
                for state in states_raw
            ):
                raise ValueError
            states = tuple(
                tuple(cast(int, bit) for bit in cast(list[object], state)) for state in states_raw
            )
            rule_number = value["reference_rule"]
            task_seed = value["task_seed"]
            if any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in (rule_number, task_seed)
            ):
                raise ValueError
            rule = ElementaryRule(cast(int, rule_number))
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleVerificationError("Phase 5 evaluator task fields are invalid") from exc
        if truth_table(ast) != table or rule.ordered_semantics != table:
            raise OracleVerificationError("Phase 5 reference semantics are inconsistent")
        if semantic_hash(ast) != value.get("semantic_hash"):
            raise OracleVerificationError("Phase 5 reference semantic hash is inconsistent")
        if not independent_rollout_matches(rule, states):
            raise OracleVerificationError("Phase 5 locked rollout failed verification")
        locked_raw = cast(dict[str, object], value["locked_rollout"])
        initial_seed = locked_raw["initial_state_seed"]
        horizon = locked_raw["horizon"]
        if any(
            isinstance(item, bool) or not isinstance(item, int) for item in (initial_seed, horizon)
        ):
            raise OracleVerificationError("Phase 5 locked rollout metadata is invalid")
        bundle = HiddenTaskBundle(
            artifact_version=PHASE5_TRANSFER_ARTIFACT_VERSION,
            task_id=task_id,
            reference_rule=rule,
            ordered_semantics=table,
            semantic_hash=str(value["semantic_hash"]),
            internal_family=str(value["generator_family"]),
            task_seed=cast(int, task_seed),
            rollout_case_seed=derive_seed(cast(int, task_seed), "phase5-rollout"),
            rollout_initial_state_seed=cast(int, initial_seed),
            locked_rollout=LockedRollout(cast(int, horizon), states),
        )
        self._accesses.append(OracleTaskAccess(task_id, role, purpose))
        return Phase5HiddenTask(
            task_id,
            role,
            str(value["generator_family"]),
            ast,
            str(value["semantic_hash"]),
            bundle,
        )
