"""Typed contextual action/outcome memory for cross-task search experience."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

from world_model_search.domain.types import Candidate, OracleResult, PublicTask, SplitLabel
from world_model_search.dsl.ast import BitExpr, Expr, ast_size, children
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.interpreter import evaluate
from world_model_search.dsl.json_schema import ast_to_value
from world_model_search.persistence.artifacts import write_content_artifact
from world_model_search.search.archive import ArchiveCoordinate, InsertionOutcome
from world_model_search.serialization import (
    JsonObject,
    JsonValue,
    canonical_json,
    sha256_json,
)

EXPERIENCE_V3_SCHEMA = "contextual-action-outcome-experience-v3"
AST_DELTA_SCHEMA = "deterministic-typed-ast-delta-v1"
CONTEXT_SCHEMA = "public-context-features-v1"
RETRIEVAL_SCHEMA = "nearest-contextual-experience-retrieval-v1"
MEMORY_BLOCK_SCHEMA = "cross-task-experience-block-v1"


class EditClass(StrEnum):
    NO_STRUCTURAL_CHANGE = "NO_STRUCTURAL_CHANGE"
    ADD_POSITION_REFERENCE = "ADD_POSITION_REFERENCE"
    REMOVE_POSITION_REFERENCE = "REMOVE_POSITION_REFERENCE"
    ADD_NEGATION = "ADD_NEGATION"
    REMOVE_NEGATION = "REMOVE_NEGATION"
    REPLACE_BOOLEAN_OPERATOR = "REPLACE_BOOLEAN_OPERATOR"
    COMPOSE_BOOLEAN_SUBTREES = "COMPOSE_BOOLEAN_SUBTREES"
    INTRODUCE_CONDITIONAL = "INTRODUCE_CONDITIONAL"
    REMOVE_CONDITIONAL = "REMOVE_CONDITIONAL"
    MODIFY_CONDITIONAL_PREDICATE = "MODIFY_CONDITIONAL_PREDICATE"
    INTRODUCE_COUNT = "INTRODUCE_COUNT"
    MODIFY_COUNT_THRESHOLD = "MODIFY_COUNT_THRESHOLD"
    INTRODUCE_MAJORITY = "INTRODUCE_MAJORITY"
    INTRODUCE_PARITY = "INTRODUCE_PARITY"
    SIMPLIFY_SUBTREE = "SIMPLIFY_SUBTREE"
    EXPAND_SUBTREE = "EXPAND_SUBTREE"
    REPRESENTATION_FAMILY_TRANSITION = "REPRESENTATION_FAMILY_TRANSITION"
    OTHER = "OTHER"


class RetrievalMode(StrEnum):
    DISABLED = "no-memory"
    POSITIVE_ONLY = "positive-nearest"
    CONTRASTIVE = "contrastive-nearest"


class ContextMode(StrEnum):
    FAMILY_ONLY = "representation-family-only"
    RICH = "rich-context"


@dataclass(frozen=True, slots=True)
class AstSubtreeChange:
    path: tuple[str, ...]
    before: JsonValue
    after: JsonValue

    def to_value(self) -> JsonObject:
        return {"path": list(self.path), "before": self.before, "after": self.after}


@dataclass(frozen=True, slots=True)
class AstDelta:
    changes: tuple[AstSubtreeChange, ...]
    edit_classes: tuple[EditClass, ...]
    affected_paths: tuple[tuple[str, ...], ...]
    constructors_added: tuple[str, ...]
    constructors_removed: tuple[str, ...]
    operators_replaced: tuple[tuple[str, str], ...]
    representation_family_before: str
    representation_family_after: str
    size_delta: int

    def to_value(self) -> JsonObject:
        return {
            "schema_version": AST_DELTA_SCHEMA,
            "exact_subtree_changes": [change.to_value() for change in self.changes],
            "normalized_edit_classes": [item.value for item in self.edit_classes],
            "affected_subtree_paths": [list(path) for path in self.affected_paths],
            "constructors_added": list(self.constructors_added),
            "constructors_removed": list(self.constructors_removed),
            "operators_replaced": [list(pair) for pair in self.operators_replaced],
            "representation_family_before": self.representation_family_before,
            "representation_family_after": self.representation_family_after,
            "canonical_size_delta": self.size_delta,
        }


def _node_counter(expr: Expr) -> Counter[str]:
    result: Counter[str] = Counter()

    def visit(node: Expr) -> None:
        result[type(node).__name__] += 1
        for child in children(node):
            visit(child)

    visit(expr)
    return result


def _expanded_difference(left: Counter[str], right: Counter[str]) -> tuple[str, ...]:
    return tuple(
        name for name in sorted(left) for _ in range(max(0, left[name] - right.get(name, 0)))
    )


def _subtree_changes(before: JsonObject, after: JsonObject) -> tuple[AstSubtreeChange, ...]:
    changes: list[AstSubtreeChange] = []

    def visit(left: JsonValue, right: JsonValue, path: tuple[str, ...]) -> None:
        if left == right:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            left_op, right_op = left.get("op"), right.get("op")
            if left_op != right_op or set(left) != set(right):
                changes.append(AstSubtreeChange(path, left, right))
                return
            child_changed = False
            for key in sorted(left):
                if key == "op":
                    continue
                lvalue, rvalue = left[key], right[key]
                if isinstance(lvalue, dict) and isinstance(rvalue, dict):
                    before_count = len(changes)
                    visit(lvalue, rvalue, (*path, key))
                    child_changed = child_changed or len(changes) > before_count
                elif lvalue != rvalue:
                    changes.append(AstSubtreeChange(path, left, right))
                    return
            if not child_changed:
                changes.append(AstSubtreeChange(path, left, right))
            return
        changes.append(AstSubtreeChange(path, left, right))

    visit(before, after, ())
    unique = {canonical_json(change.to_value()): change for change in changes}
    return tuple(unique[key] for key in sorted(unique))


def extract_ast_delta(parent: BitExpr, child: BitExpr) -> AstDelta:
    """Derive exact structural changes and deterministic normalized edit tags."""

    from world_model_search.search.archive import representation_family

    before = canonicalize(parent)
    after = canonicalize(child)
    before_value, after_value = ast_to_value(before), ast_to_value(after)
    changes = _subtree_changes(before_value, after_value)
    before_counts, after_counts = _node_counter(before), _node_counter(after)
    added = _expanded_difference(after_counts, before_counts)
    removed = _expanded_difference(before_counts, after_counts)
    replacements = tuple(
        sorted(
            (str(change.before.get("op")), str(change.after.get("op")))
            for change in changes
            if isinstance(change.before, dict)
            and isinstance(change.after, dict)
            and change.before.get("op") != change.after.get("op")
        )
    )
    family_before = representation_family(before).value
    family_after = representation_family(after).value
    size_delta = ast_size(after)[0] - ast_size(before)[0]
    tags: set[EditClass] = set()
    if before_value == after_value:
        tags.add(EditClass.NO_STRUCTURAL_CHANGE)
    if "At" in added:
        tags.add(EditClass.ADD_POSITION_REFERENCE)
    if "At" in removed:
        tags.add(EditClass.REMOVE_POSITION_REFERENCE)
    if "Not" in added:
        tags.add(EditClass.ADD_NEGATION)
    if "Not" in removed:
        tags.add(EditClass.REMOVE_NEGATION)
    if any(
        left in {"And", "Or", "Xor"} and right in {"And", "Or", "Xor"}
        for left, right in replacements
    ):
        tags.add(EditClass.REPLACE_BOOLEAN_OPERATOR)
    if sum(after_counts[name] - before_counts[name] for name in ("And", "Or", "Xor")) >= 1:
        tags.add(EditClass.COMPOSE_BOOLEAN_SUBTREES)
    if "If" in added:
        tags.add(EditClass.INTRODUCE_CONDITIONAL)
    if "If" in removed:
        tags.add(EditClass.REMOVE_CONDITIONAL)
    if any(path and path[-1] == "condition" for path in (change.path for change in changes)):
        tags.add(EditClass.MODIFY_CONDITIONAL_PREDICATE)
    if "Count" in added:
        tags.add(EditClass.INTRODUCE_COUNT)
    threshold_nodes = ("IntConst", "Eq", "Le", "Ge", "Between", "AddConst")
    if any(name in added or name in removed for name in threshold_nodes):
        tags.add(EditClass.MODIFY_COUNT_THRESHOLD)
    if "Majority" in added:
        tags.add(EditClass.INTRODUCE_MAJORITY)
    if "Parity" in added:
        tags.add(EditClass.INTRODUCE_PARITY)
    if size_delta < 0:
        tags.add(EditClass.SIMPLIFY_SUBTREE)
    if size_delta > 0:
        tags.add(EditClass.EXPAND_SUBTREE)
    if family_before != family_after:
        tags.add(EditClass.REPRESENTATION_FAMILY_TRANSITION)
    if not tags:
        tags.add(EditClass.OTHER)
    return AstDelta(
        changes=changes,
        edit_classes=tuple(sorted(tags, key=str)),
        affected_paths=tuple(change.path for change in changes),
        constructors_added=added,
        constructors_removed=removed,
        operators_replaced=replacements,
        representation_family_before=family_before,
        representation_family_after=family_after,
        size_delta=size_delta,
    )


def _histogram_value(expr: BitExpr) -> JsonObject:
    return cast(JsonObject, dict(sorted(_node_counter(expr).items())))


def _motifs(expr: BitExpr, maximum_depth: int = 2) -> tuple[str, ...]:
    motifs: set[str] = set()

    def visit(node: Expr, depth: int) -> None:
        descendants = children(node)
        if descendants:
            child_names = ",".join(type(item).__name__ for item in descendants)
            motifs.add(f"{type(node).__name__}({child_names})")
        else:
            value = ast_to_value(node)
            motifs.add(canonical_json(value))
        if depth < maximum_depth:
            for child in descendants:
                visit(child, depth + 1)

    visit(expr, 0)
    return tuple(sorted(motifs))


def _public_probe_behavior(expr: BitExpr, task: PublicTask) -> str:
    from world_model_search.search.archive import public_probe_contract

    return "".join(str(evaluate(expr, case)) for case, _expected in public_probe_contract(task))


@dataclass(frozen=True, slots=True)
class ContextFeatures:
    representation_family: str
    parent_canonical_ast: JsonObject
    parent_canonical_size: int
    constructor_histogram: JsonObject
    structural_motifs: tuple[str, ...]
    parent_score: int
    parent_error_count: int
    parent_error_pattern: tuple[str, ...] | None
    public_probe_behavior: str
    behavior_cluster: str
    archive_descriptor: JsonObject
    search_step: int
    lineage_depth: int
    plateau_length: int
    recent_edit_classes: tuple[str, ...]

    def to_value(self) -> JsonObject:
        return {
            "schema_version": CONTEXT_SCHEMA,
            "representation_family": self.representation_family,
            "parent_canonical_ast": self.parent_canonical_ast,
            "parent_canonical_size": self.parent_canonical_size,
            "constructor_histogram": self.constructor_histogram,
            "structural_motifs": list(self.structural_motifs),
            "parent_score": self.parent_score,
            "parent_error_count": self.parent_error_count,
            "parent_error_pattern": (
                list(self.parent_error_pattern) if self.parent_error_pattern is not None else None
            ),
            "parent_error_pattern_publicly_available": self.parent_error_pattern is not None,
            "public_probe_behavior": self.public_probe_behavior,
            "behavior_cluster": self.behavior_cluster,
            "archive_descriptor": self.archive_descriptor,
            "search_step": self.search_step,
            "lineage_depth": self.lineage_depth,
            "plateau_length": self.plateau_length,
            "recent_edit_classes": list(self.recent_edit_classes),
        }

    @classmethod
    def from_value(cls, value: object) -> ContextFeatures:
        if not isinstance(value, dict):
            raise ValueError("context features must be an object")
        pattern = value.get("parent_error_pattern")
        histogram = value.get("constructor_histogram")
        descriptor = value.get("archive_descriptor")
        if not isinstance(histogram, dict) or not isinstance(descriptor, dict):
            raise ValueError("context histogram/descriptor is malformed")
        return cls(
            representation_family=str(value["representation_family"]),
            parent_canonical_ast=cast(JsonObject, value["parent_canonical_ast"]),
            parent_canonical_size=int(value["parent_canonical_size"]),
            constructor_histogram=cast(JsonObject, histogram),
            structural_motifs=tuple(
                str(item) for item in cast(list[object], value["structural_motifs"])
            ),
            parent_score=int(value["parent_score"]),
            parent_error_count=int(value["parent_error_count"]),
            parent_error_pattern=(
                tuple(str(item) for item in pattern) if isinstance(pattern, list) else None
            ),
            public_probe_behavior=str(value["public_probe_behavior"]),
            behavior_cluster=str(value["behavior_cluster"]),
            archive_descriptor=cast(JsonObject, descriptor),
            search_step=int(value["search_step"]),
            lineage_depth=int(value["lineage_depth"]),
            plateau_length=int(value["plateau_length"]),
            recent_edit_classes=tuple(
                str(item) for item in cast(list[object], value["recent_edit_classes"])
            ),
        )


class ContextFeatureExtractor:
    """Extract deterministic proposer-safe context after parent selection."""

    def extract(
        self,
        *,
        parent: BitExpr,
        result: OracleResult,
        task: PublicTask,
        coordinate: ArchiveCoordinate,
        search_step: int,
        lineage_depth: int,
        plateau_length: int,
        recent_edit_classes: tuple[str, ...],
    ) -> ContextFeatures:
        canonical = canonicalize(parent)
        return ContextFeatures(
            representation_family=coordinate.representation_family.value,
            parent_canonical_ast=ast_to_value(canonical),
            parent_canonical_size=ast_size(canonical)[0],
            constructor_histogram=_histogram_value(canonical),
            structural_motifs=_motifs(canonical),
            parent_score=result.local_cases - result.local_errors,
            parent_error_count=result.local_errors,
            # The current score-only proposer contract does not expose per-case errors.
            parent_error_pattern=None,
            public_probe_behavior=_public_probe_behavior(canonical, task),
            behavior_cluster=coordinate.error_signature_cluster,
            archive_descriptor=coordinate.to_value(),
            search_step=search_step,
            lineage_depth=lineage_depth,
            plateau_length=plateau_length,
            recent_edit_classes=recent_edit_classes,
        )


@dataclass(frozen=True, slots=True)
class ExperienceProvenance:
    task_generator_family: str
    task_id: str
    source_split: str
    run_id: str
    search_seed: int
    request_index: int
    item_ordinal: int
    parent_candidate_id: str
    child_candidate_id: str
    parent_lineage_id: str
    sequence_index: int

    def to_value(self) -> JsonObject:
        return {
            "provenance_version": "experience-provenance-v1",
            "task_generator_family": self.task_generator_family,
            "task_id": self.task_id,
            "source_split": self.source_split,
            "run_id": self.run_id,
            "search_seed": self.search_seed,
            "request_index": self.request_index,
            "item_ordinal": self.item_ordinal,
            "parent_candidate_id": self.parent_candidate_id,
            "child_candidate_id": self.child_candidate_id,
            "parent_lineage_id": self.parent_lineage_id,
            "sequence_index": self.sequence_index,
        }


@dataclass(frozen=True, slots=True)
class ImmediateOutcome:
    child_score: int
    score_delta: int
    child_error_count: int
    error_delta: int
    exact_solution: bool
    exact_regression: bool
    archive_inserted: bool
    archive_replaced_elite: bool
    archive_outcome: str
    descriptor_transition: JsonObject
    canonical_duplicate: bool
    semantic_duplicate: bool
    novelty_measure: int | None

    def to_value(self) -> JsonObject:
        return {
            "outcome_version": "immediate-outcome-v1",
            "child_score": self.child_score,
            "score_delta": self.score_delta,
            "child_error_count": self.child_error_count,
            "error_delta": self.error_delta,
            "exact_solution": self.exact_solution,
            "exact_regression": self.exact_regression,
            "archive_inserted": self.archive_inserted,
            "archive_replaced_elite": self.archive_replaced_elite,
            "archive_outcome": self.archive_outcome,
            "descriptor_transition": self.descriptor_transition,
            "canonical_duplicate": self.canonical_duplicate,
            "semantic_duplicate": self.semantic_duplicate,
            "novelty_measure": self.novelty_measure,
        }


@dataclass(frozen=True, slots=True)
class DownstreamOutcome:
    annotation_status: str
    eventually_had_exact_descendant: bool
    evaluations_until_exact_descendant: int | None
    best_descendant_score: int
    max_descendant_improvement: int
    lineage_survival_length: int
    short_horizon_steps: int
    short_horizon_best_score_gain: int
    short_horizon_exact_descendant: bool

    def to_value(self) -> JsonObject:
        return {
            "outcome_version": "downstream-lineage-outcome-v1",
            "annotation_status": self.annotation_status,
            "eventually_had_exact_descendant": self.eventually_had_exact_descendant,
            "evaluations_until_exact_descendant": self.evaluations_until_exact_descendant,
            "best_descendant_score": self.best_descendant_score,
            "max_descendant_improvement": self.max_descendant_improvement,
            "lineage_survival_length": self.lineage_survival_length,
            "short_horizon_steps": self.short_horizon_steps,
            "short_horizon_best_score_gain": self.short_horizon_best_score_gain,
            "short_horizon_exact_descendant": self.short_horizon_exact_descendant,
        }


PENDING_DOWNSTREAM = DownstreamOutcome(
    "pending-run-finalization", False, None, 0, 0, 0, 0, 0, False
)


@dataclass(frozen=True, slots=True)
class ExperienceMetadata:
    training_eligible: bool
    sealed_test: bool
    evidence_timing: str

    def to_value(self) -> JsonObject:
        return {
            "metadata_version": "experience-memory-metadata-v1",
            "training_eligible": self.training_eligible,
            "sealed_test": self.sealed_test,
            "evidence_timing": self.evidence_timing,
        }


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    provenance: ExperienceProvenance
    context: ContextFeatures
    action: AstDelta
    immediate_outcome: ImmediateOutcome
    downstream_outcome: DownstreamOutcome
    memory_metadata: ExperienceMetadata

    @property
    def record_id(self) -> str:
        return sha256_json(self.identity_value())

    def identity_value(self) -> JsonObject:
        return {
            "schema_version": EXPERIENCE_V3_SCHEMA,
            "provenance": self.provenance.to_value(),
            "context": self.context.to_value(),
            "action": self.action.to_value(),
            "immediate_outcome": self.immediate_outcome.to_value(),
            "memory_metadata": self.memory_metadata.to_value(),
        }

    def to_value(self) -> JsonObject:
        return {
            "record_id": self.record_id,
            **self.identity_value(),
            "downstream_outcome": self.downstream_outcome.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> ExperienceRecord:
        if not isinstance(value, dict) or value.get("schema_version") != EXPERIENCE_V3_SCHEMA:
            raise ValueError("experience record schema is invalid")
        objects = tuple(
            value.get(name)
            for name in (
                "provenance",
                "context",
                "action",
                "immediate_outcome",
                "downstream_outcome",
                "memory_metadata",
            )
        )
        if not all(isinstance(item, dict) for item in objects):
            raise ValueError("experience record sections must be objects")
        provenance = cast(dict[str, object], value["provenance"])
        action = cast(dict[str, object], value["action"])
        immediate = cast(dict[str, object], value["immediate_outcome"])
        downstream = cast(dict[str, object], value["downstream_outcome"])
        metadata = cast(dict[str, object], value["memory_metadata"])
        changes_raw = cast(list[object], action["exact_subtree_changes"])
        delta = AstDelta(
            tuple(
                AstSubtreeChange(
                    tuple(
                        str(part)
                        for part in cast(list[object], cast(dict[str, object], item)["path"])
                    ),
                    cast(JsonValue, cast(dict[str, object], item)["before"]),
                    cast(JsonValue, cast(dict[str, object], item)["after"]),
                )
                for item in changes_raw
            ),
            tuple(
                EditClass(str(item))
                for item in cast(list[object], action["normalized_edit_classes"])
            ),
            tuple(
                tuple(str(part) for part in cast(list[object], path))
                for path in cast(list[object], action["affected_subtree_paths"])
            ),
            tuple(str(item) for item in cast(list[object], action["constructors_added"])),
            tuple(str(item) for item in cast(list[object], action["constructors_removed"])),
            tuple(
                cast(tuple[str, str], tuple(str(part) for part in cast(list[object], pair)))
                for pair in cast(list[object], action["operators_replaced"])
            ),
            str(action["representation_family_before"]),
            str(action["representation_family_after"]),
            _as_int(action["canonical_size_delta"]),
        )
        record = cls(
            ExperienceProvenance(
                str(provenance["task_generator_family"]),
                str(provenance["task_id"]),
                str(provenance["source_split"]),
                str(provenance["run_id"]),
                _as_int(provenance["search_seed"]),
                _as_int(provenance["request_index"]),
                _as_int(provenance["item_ordinal"]),
                str(provenance["parent_candidate_id"]),
                str(provenance["child_candidate_id"]),
                str(provenance["parent_lineage_id"]),
                _as_int(provenance["sequence_index"]),
            ),
            ContextFeatures.from_value(value["context"]),
            delta,
            ImmediateOutcome(
                _as_int(immediate["child_score"]),
                _as_int(immediate["score_delta"]),
                _as_int(immediate["child_error_count"]),
                _as_int(immediate["error_delta"]),
                bool(immediate["exact_solution"]),
                bool(immediate["exact_regression"]),
                bool(immediate["archive_inserted"]),
                bool(immediate["archive_replaced_elite"]),
                str(immediate["archive_outcome"]),
                cast(JsonObject, immediate["descriptor_transition"]),
                bool(immediate["canonical_duplicate"]),
                bool(immediate["semantic_duplicate"]),
                (
                    _as_int(immediate["novelty_measure"])
                    if immediate.get("novelty_measure") is not None
                    else None
                ),
            ),
            DownstreamOutcome(
                str(downstream["annotation_status"]),
                bool(downstream["eventually_had_exact_descendant"]),
                (
                    _as_int(downstream["evaluations_until_exact_descendant"])
                    if downstream.get("evaluations_until_exact_descendant") is not None
                    else None
                ),
                _as_int(downstream["best_descendant_score"]),
                _as_int(downstream["max_descendant_improvement"]),
                _as_int(downstream["lineage_survival_length"]),
                _as_int(downstream["short_horizon_steps"]),
                _as_int(downstream["short_horizon_best_score_gain"]),
                bool(downstream["short_horizon_exact_descendant"]),
            ),
            ExperienceMetadata(
                bool(metadata["training_eligible"]),
                bool(metadata["sealed_test"]),
                str(metadata["evidence_timing"]),
            ),
        )
        if record.record_id != value.get("record_id"):
            raise ValueError("experience record identity hash mismatch")
        return record


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected an integer experience field")
    return value


class ExperienceRecorder:
    """Write immutable raw and downstream-annotated transition records."""

    def __init__(self, run_directory: Path) -> None:
        self.root = run_directory / "experience_v3"

    def write_raw(self, evaluation_index: int, record: ExperienceRecord) -> str:
        return write_content_artifact(
            self.root / "raw" / f"evaluation-{evaluation_index:05d}.json",
            canonical_json(record.to_value()),
        )

    def write_annotated(self, evaluation_index: int, record: ExperienceRecord) -> str:
        if record.downstream_outcome.annotation_status != "complete":
            raise ValueError("annotated experience must have complete downstream outcomes")
        return write_content_artifact(
            self.root / "annotated" / f"evaluation-{evaluation_index:05d}.json",
            canonical_json(record.to_value()),
        )


def create_experience_record(
    *,
    task_generator_family: str,
    task_split: SplitLabel,
    run_id: str,
    search_seed: int,
    request_index: int,
    item_ordinal: int,
    evaluation_index: int,
    parent: Candidate,
    parent_result: OracleResult,
    parent_context: ContextFeatures,
    child: Candidate,
    child_result: OracleResult,
    child_coordinate: ArchiveCoordinate,
    archive_outcome: InsertionOutcome,
    canonical_duplicate: bool,
    semantic_duplicate: bool,
    sealed_test: bool,
    evidence_timing: str = "prospective",
) -> ExperienceRecord:
    if not isinstance(parent.ast, BitExpr) or not isinstance(child.ast, BitExpr):
        raise TypeError("experience records require typed parent and child ASTs")
    delta = extract_ast_delta(parent.ast, child.ast)
    inserted = archive_outcome in {
        InsertionOutcome.INSERTED,
        InsertionOutcome.REPLACED,
        InsertionOutcome.RESERVED,
    }
    immediate = ImmediateOutcome(
        child_score=child_result.local_cases - child_result.local_errors,
        score_delta=parent_result.local_errors - child_result.local_errors,
        child_error_count=child_result.local_errors,
        error_delta=child_result.local_errors - parent_result.local_errors,
        exact_solution=child_result.exact,
        exact_regression=parent_result.exact and not child_result.exact,
        archive_inserted=inserted,
        archive_replaced_elite=archive_outcome is InsertionOutcome.REPLACED,
        archive_outcome=archive_outcome.value,
        descriptor_transition={
            "before": parent_context.archive_descriptor,
            "after": child_coordinate.to_value(),
        },
        canonical_duplicate=canonical_duplicate,
        semantic_duplicate=semantic_duplicate,
        novelty_measure=None,
    )
    metadata = ExperienceMetadata(
        training_eligible=(
            not sealed_test and task_split in {SplitLabel.TRAINING, SplitLabel.DEVELOPMENT}
        ),
        sealed_test=sealed_test,
        evidence_timing=evidence_timing,
    )
    return ExperienceRecord(
        ExperienceProvenance(
            task_generator_family,
            child.task_id,
            task_split.value,
            run_id,
            search_seed,
            request_index,
            item_ordinal,
            parent.candidate_id,
            child.candidate_id,
            sha256_json({"task_id": child.task_id, "parent_candidate_id": parent.candidate_id}),
            evaluation_index,
        ),
        parent_context,
        delta,
        immediate,
        PENDING_DOWNSTREAM,
        metadata,
    )


class DownstreamOutcomeAnnotator:
    """Add descriptive descendant outcomes without assigning causal credit."""

    def annotate(
        self,
        records: tuple[ExperienceRecord, ...],
        *,
        parent_ids: dict[str, tuple[str, ...]],
        score_by_candidate: dict[str, int],
        exact_by_candidate: dict[str, bool],
        evaluation_by_candidate: dict[str, int],
        short_horizon_steps: int = 8,
    ) -> tuple[ExperienceRecord, ...]:
        if short_horizon_steps < 1:
            raise ValueError("short horizon must be positive")
        children_by_parent: dict[str, list[str]] = {}
        for child_id, parents in parent_ids.items():
            for parent_id in parents:
                children_by_parent.setdefault(parent_id, []).append(child_id)
        annotated: list[ExperienceRecord] = []
        for record in records:
            root = record.provenance.child_candidate_id
            queue = [(child, 1) for child in sorted(children_by_parent.get(root, []))]
            descendants: list[tuple[str, int]] = []
            seen: set[str] = set()
            while queue:
                candidate_id, depth = queue.pop(0)
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                if (
                    evaluation_by_candidate.get(candidate_id, -1)
                    <= record.provenance.sequence_index
                ):
                    continue
                descendants.append((candidate_id, depth))
                queue.extend(
                    (child, depth + 1) for child in sorted(children_by_parent.get(candidate_id, []))
                )
            best = max(
                (score_by_candidate[candidate_id] for candidate_id, _depth in descendants),
                default=record.immediate_outcome.child_score,
            )
            exact_descendants = [
                candidate_id
                for candidate_id, _depth in descendants
                if exact_by_candidate.get(candidate_id, False)
            ]
            until = (
                min(evaluation_by_candidate[item] for item in exact_descendants)
                - record.provenance.sequence_index
                if exact_descendants
                else None
            )
            short_horizon = [
                candidate_id
                for candidate_id, _depth in descendants
                if evaluation_by_candidate[candidate_id]
                <= record.provenance.sequence_index + short_horizon_steps
            ]
            short_best = max(
                (score_by_candidate[candidate_id] for candidate_id in short_horizon),
                default=record.immediate_outcome.child_score,
            )
            downstream = DownstreamOutcome(
                "complete",
                bool(exact_descendants),
                until,
                best,
                best - record.immediate_outcome.child_score,
                max((depth for _item, depth in descendants), default=0),
                short_horizon_steps,
                short_best - record.immediate_outcome.child_score,
                any(exact_by_candidate.get(candidate_id, False) for candidate_id in short_horizon),
            )
            annotated.append(replace(record, downstream_outcome=downstream))
        return tuple(annotated)
