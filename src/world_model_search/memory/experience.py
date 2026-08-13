"""Experience-derived, archive-family-scoped cross-task memory.

This v2 path is intentionally separate from the completed Phase 5 v1 memory.  Training
evidence is a matched contrast between a successful search lineage and unsuccessful siblings
from the same request and selected parent.  Retrieval is keyed by the public MAP-Elites
representation family of the branch selected for processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from world_model_search.domain.types import Candidate, OracleResult, SplitLabel
from world_model_search.dsl.ast import BitExpr
from world_model_search.dsl.json_schema import ast_to_value
from world_model_search.search.archive import (
    ArchiveCoordinate,
    ArchiveLayer,
    RepresentationFamily,
    representation_family,
    size_bin,
)
from world_model_search.serialization import JsonObject, canonical_json, sha256_json

EXPERIENCE_LINEAGE_VERSION = "phase5-experience-lineage-v2"
EXPERIENCE_CONTRAST_VERSION = "phase5-matched-lineage-contrast-v2"
EXPERIENCE_LESSON_VERSION = "phase5-experience-lesson-v2"
EXPERIENCE_SNAPSHOT_VERSION = "phase5-experience-snapshot-v2"
EXPERIENCE_RETRIEVAL_VERSION = "phase5-cell-conditioned-retrieval-v2"
EXPERIENCE_MEMORY_BLOCK_VERSION = "phase5-cell-conditioned-memory-block-v2"
MATCHED_FAILURE_BASIS = "same-request-same-selected-parent-v1"


def _complete_hash(value: str, label: str) -> None:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{label} must be a complete lowercase SHA-256")


def _coordinate_value(coordinate: ArchiveCoordinate) -> JsonObject:
    return coordinate.to_value()


@dataclass(frozen=True, slots=True)
class ExperienceLineageStep:
    """One proposer-visible candidate/score transition in a successful lineage."""

    candidate_id: str
    ordered_parent_ids: tuple[str, ...]
    ast: BitExpr
    local_errors: int
    local_cases: int
    exact: bool
    ast_bits: int
    residual_bits: int
    archive_coordinate: ArchiveCoordinate

    def __post_init__(self) -> None:
        if not self.candidate_id or not isinstance(self.ast, BitExpr):
            raise ValueError("experience lineage step requires a typed candidate")
        if len(set(self.ordered_parent_ids)) != len(self.ordered_parent_ids):
            raise ValueError("experience lineage parents must be unique and ordered")
        if (
            min(self.local_errors, self.local_cases, self.ast_bits, self.residual_bits) < 0
            or self.local_errors > self.local_cases
            or self.local_cases < 1
        ):
            raise ValueError("experience lineage score fields are invalid")
        if self.exact != (
            self.local_errors == 0 and self.archive_coordinate.layer is ArchiveLayer.EXACT
        ):
            raise ValueError("experience lineage exactness and archive layer differ")
        if self.archive_coordinate.representation_family is not representation_family(
            self.ast
        ) or self.archive_coordinate.size_bin != size_bin(self.ast):
            raise ValueError("experience lineage AST and public archive coordinate differ")

    @classmethod
    def from_search_record(
        cls,
        *,
        candidate: Candidate,
        result: OracleResult,
        coordinate: ArchiveCoordinate,
    ) -> ExperienceLineageStep:
        if not isinstance(candidate.ast, BitExpr):
            raise TypeError("experience memory accepts only typed DSL candidates")
        return cls(
            candidate_id=candidate.candidate_id,
            ordered_parent_ids=candidate.parent_ids,
            ast=candidate.ast,
            local_errors=result.local_errors,
            local_cases=result.local_cases,
            exact=result.exact,
            ast_bits=result.ast_bits,
            residual_bits=result.residual_bits,
            archive_coordinate=coordinate,
        )

    def proposer_value(self) -> JsonObject:
        """Return only candidate syntax, public archive coordinates, and score-only feedback."""

        return {
            "candidate_id": self.candidate_id,
            "ordered_parent_ids": list(self.ordered_parent_ids),
            "ast": ast_to_value(self.ast),
            "score": {
                "local_errors": self.local_errors,
                "local_cases": self.local_cases,
                "exact": self.exact,
                "ast_bits": self.ast_bits,
                "residual_bits": self.residual_bits,
                "two_part_bits": self.ast_bits + self.residual_bits,
            },
            "archive_coordinate": _coordinate_value(self.archive_coordinate),
        }


def _validate_connected_lineage(steps: tuple[ExperienceLineageStep, ...], *, label: str) -> None:
    if not steps:
        raise ValueError(f"{label} is empty")
    seen: set[str] = set()
    for index, step in enumerate(steps):
        if step.candidate_id in seen:
            raise ValueError(f"{label} contains a duplicate candidate")
        if index and not set(step.ordered_parent_ids) & seen:
            raise ValueError(f"{label} step has no earlier recorded parent")
        if not index and step.ordered_parent_ids:
            raise ValueError(f"{label} root cannot cite an omitted parent")
        seen.add(step.candidate_id)


@dataclass(frozen=True, slots=True)
class MatchedUnsuccessfulLineage:
    """A failed sibling generated in the successful request from the same selected parent."""

    steps: tuple[ExperienceLineageStep, ...]
    match_basis: str = MATCHED_FAILURE_BASIS

    def __post_init__(self) -> None:
        if self.match_basis != MATCHED_FAILURE_BASIS:
            raise ValueError("unsupported unsuccessful-lineage matching basis")
        _validate_connected_lineage(self.steps, label="matched unsuccessful lineage")
        if self.steps[-1].exact:
            raise ValueError("matched unsuccessful lineage cannot have an exact terminal")

    @property
    def terminal(self) -> ExperienceLineageStep:
        return self.steps[-1]

    def evaluator_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "match_basis": self.match_basis,
                "ordered_unsuccessful_lineage": [step.proposer_value() for step in self.steps],
            },
        )

    def proposer_value(self) -> JsonObject:
        """Avoid repeating the successful lineage's shared prefix in the model prompt."""

        return {
            "match_basis": self.match_basis,
            "unsuccessful_revision_from_shared_selected_parent": self.terminal.proposer_value(),
        }


@dataclass(frozen=True, slots=True)
class SuccessfulLineageEvidence:
    """One successful lineage plus same-request failures, with evaluator-only provenance."""

    task_id: str
    generator_family_id: str
    role: SplitLabel
    source_task_split: SplitLabel
    search_seed: int
    consequential_request_index: int
    selected_parent_candidate_id: str
    steps: tuple[ExperienceLineageStep, ...]
    matched_unsuccessful_lineages: tuple[MatchedUnsuccessfulLineage, ...]
    run_hash: str
    artifact_hash: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.generator_family_id or not self.steps:
            raise ValueError("successful lineage evidence is incomplete")
        if self.role not in {SplitLabel.TRAINING, SplitLabel.DEVELOPMENT}:
            raise ValueError("sealed or unrelated evidence cannot enter experience memory")
        if self.source_task_split in {SplitLabel.TEST, SplitLabel.VALIDATION}:
            raise ValueError("validation or sealed source tasks cannot enter training memory")
        if self.search_seed < 0 or self.consequential_request_index < 0:
            raise ValueError("lineage request identity is invalid")
        if not self.selected_parent_candidate_id:
            raise ValueError("lineage evidence requires the consequential selected parent")
        _complete_hash(self.run_hash, "lineage run hash")
        _complete_hash(self.artifact_hash, "lineage artifact hash")
        _validate_connected_lineage(self.steps, label="successful lineage")
        terminal = self.steps[-1]
        if not terminal.exact or terminal.archive_coordinate.layer is not ArchiveLayer.EXACT:
            raise ValueError("experience evidence requires an exact terminal archive elite")
        if self.selected_parent_candidate_id not in terminal.ordered_parent_ids:
            raise ValueError("exact terminal was not generated from the selected parent")
        if not any(
            step.candidate_id == self.selected_parent_candidate_id for step in self.steps[:-1]
        ):
            raise ValueError("successful lineage omits the consequential selected parent")
        if not self.matched_unsuccessful_lineages:
            raise ValueError("successful evidence requires matched unsuccessful lineages")
        matched_terminals: set[str] = set()
        for matched in self.matched_unsuccessful_lineages:
            if self.selected_parent_candidate_id not in matched.terminal.ordered_parent_ids:
                raise ValueError("unsuccessful lineage is not matched to the selected parent")
            if matched.terminal.candidate_id in matched_terminals:
                raise ValueError("matched unsuccessful terminal is duplicated")
            matched_terminals.add(matched.terminal.candidate_id)

    @property
    def archive_representation_family(self) -> RepresentationFamily:
        """Family assignment from the selected parent cell, not the successful child."""

        for step in self.steps:
            if step.candidate_id == self.selected_parent_candidate_id:
                return step.archive_coordinate.representation_family
        raise AssertionError("validated selected parent is unavailable")

    def evaluator_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "lineage_version": EXPERIENCE_LINEAGE_VERSION,
                "task_id": self.task_id,
                "generator_family_id": self.generator_family_id,
                "role": self.role.value,
                "source_task_split": self.source_task_split.value,
                "search_seed": self.search_seed,
                "consequential_request_index": self.consequential_request_index,
                "selected_parent_candidate_id": self.selected_parent_candidate_id,
                "archive_representation_family": self.archive_representation_family.value,
                "steps": [step.proposer_value() for step in self.steps],
                "matched_unsuccessful_lineages": [
                    item.evaluator_value() for item in self.matched_unsuccessful_lineages
                ],
                "run_hash": self.run_hash,
                "artifact_hash": self.artifact_hash,
            },
        )

    @property
    def evidence_id(self) -> str:
        return sha256_json(self.evaluator_value())

    def proposer_value(self) -> JsonObject:
        """Sanitized packet for LLM lesson induction; hidden generator family is omitted."""

        return cast(
            JsonObject,
            {
                "contrast_version": EXPERIENCE_CONTRAST_VERSION,
                "evidence_id": self.evidence_id,
                "opaque_task_id": self.task_id,
                "archive_representation_family": self.archive_representation_family.value,
                "selected_parent_candidate_id": self.selected_parent_candidate_id,
                "ordered_successful_lineage": [step.proposer_value() for step in self.steps],
                "matched_unsuccessful_lineages": [
                    item.proposer_value() for item in self.matched_unsuccessful_lineages
                ],
            },
        )


@dataclass(frozen=True, slots=True)
class ExperienceLessonProposal:
    lesson_text: str
    archive_representation_family: RepresentationFamily
    source_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        encoded = self.lesson_text.encode("utf-8")
        if not self.lesson_text.strip() or len(encoded) > 1_200:
            raise ValueError("experience lesson text must be nonempty and at most 1,200 bytes")
        if tuple(sorted(set(self.source_evidence_ids))) != self.source_evidence_ids:
            raise ValueError("experience lesson evidence IDs must be unique and sorted")
        for evidence_id in self.source_evidence_ids:
            _complete_hash(evidence_id, "experience lesson evidence ID")
        forbidden = ("generator_family", "reference_ast", "semantic_hash", "oracle_handle")
        if any(value in self.lesson_text.casefold() for value in forbidden):
            raise ValueError("experience lesson contains evaluator-only vocabulary")

    def identity_value(self) -> JsonObject:
        return {
            "lesson_version": EXPERIENCE_LESSON_VERSION,
            "lesson_text": self.lesson_text,
            "archive_representation_family": self.archive_representation_family.value,
            "source_evidence_ids": list(self.source_evidence_ids),
        }

    @property
    def proposal_id(self) -> str:
        return sha256_json(self.identity_value())


class ValidationStage(StrEnum):
    INDIVIDUAL_LESSON = "individual-lesson-screen-v1"
    PROMOTED_BUNDLE = "promoted-bundle-confirmation-v1"


@dataclass(frozen=True, slots=True)
class LessonValidationPair:
    validation_stage: ValidationStage
    lesson_proposal_id: str
    task_id: str
    generator_family_id: str
    seed: int
    archive_representation_family: RepresentationFamily
    baseline_exact: bool
    treatment_exact: bool
    baseline_normalized_exact_auc_ppm: int
    treatment_normalized_exact_auc_ppm: int
    treatment_memory_applied: bool

    def __post_init__(self) -> None:
        if self.validation_stage is not ValidationStage.INDIVIDUAL_LESSON:
            raise ValueError("lesson validation pair must be an individual-lesson screen")
        _complete_hash(self.lesson_proposal_id, "lesson validation proposal ID")
        if not self.task_id or not self.generator_family_id or self.seed < 0:
            raise ValueError("lesson validation pair identity is invalid")
        for value in (
            self.baseline_normalized_exact_auc_ppm,
            self.treatment_normalized_exact_auc_ppm,
        ):
            if not 0 <= value <= 1_000_000:
                raise ValueError("normalized exact AUC must be in [0, 1_000,000] ppm")

    @property
    def difference_ppm(self) -> int:
        return self.treatment_normalized_exact_auc_ppm - self.baseline_normalized_exact_auc_ppm

    def to_value(self) -> JsonObject:
        return {
            "validation_stage": self.validation_stage.value,
            "lesson_proposal_id": self.lesson_proposal_id,
            "task_id": self.task_id,
            "generator_family_id": self.generator_family_id,
            "seed": self.seed,
            "archive_representation_family": self.archive_representation_family.value,
            "baseline_exact": self.baseline_exact,
            "treatment_exact": self.treatment_exact,
            "baseline_normalized_exact_auc_ppm": self.baseline_normalized_exact_auc_ppm,
            "treatment_normalized_exact_auc_ppm": self.treatment_normalized_exact_auc_ppm,
            "treatment_memory_applied": self.treatment_memory_applied,
        }


@dataclass(frozen=True, slots=True)
class LessonPromotionPolicy:
    """Frozen thresholds; v2 defaults permit the declared single source family."""

    minimum_training_tasks: int = 2
    minimum_training_generator_families: int = 1
    minimum_validation_tasks_with_exposure: int = 4
    minimum_validation_generator_families: int = 1

    def __post_init__(self) -> None:
        if (
            min(
                self.minimum_training_tasks,
                self.minimum_training_generator_families,
                self.minimum_validation_tasks_with_exposure,
                self.minimum_validation_generator_families,
            )
            < 1
        ):
            raise ValueError("lesson promotion thresholds must be positive")


DEFAULT_LESSON_PROMOTION_POLICY = LessonPromotionPolicy()


@dataclass(frozen=True, slots=True)
class PromotedExperienceLesson:
    proposal: ExperienceLessonProposal
    validation_pair_hashes: tuple[str, ...]
    validation_mean_gain_ppm: int

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.validation_pair_hashes))) != self.validation_pair_hashes:
            raise ValueError("validation pair hashes must be unique and sorted")
        for pair_hash in self.validation_pair_hashes:
            _complete_hash(pair_hash, "validation pair hash")

    @property
    def lesson_id(self) -> str:
        return sha256_json(
            {
                **self.proposal.identity_value(),
                "validation_pair_hashes": list(self.validation_pair_hashes),
                "validation_mean_gain_ppm": self.validation_mean_gain_ppm,
            }
        )

    def safe_value(self) -> JsonObject:
        return {
            "lesson_id": self.lesson_id,
            "lesson_text": self.proposal.lesson_text,
            "archive_representation_family": self.proposal.archive_representation_family.value,
        }


@dataclass(frozen=True, slots=True)
class LessonPromotionDecision:
    status: str
    reasons: tuple[str, ...]
    promoted_lesson: PromotedExperienceLesson | None


def evaluate_lesson_promotion(
    *,
    proposal: ExperienceLessonProposal,
    evidence_catalog: dict[str, SuccessfulLineageEvidence],
    validation_pairs: tuple[LessonValidationPair, ...],
    policy: LessonPromotionPolicy = DEFAULT_LESSON_PROMOTION_POLICY,
) -> LessonPromotionDecision:
    """Screen one sole lesson against a shared control on prospective tasks."""

    sources = [evidence_catalog.get(evidence_id) for evidence_id in proposal.source_evidence_ids]
    invalid: list[str] = []
    insufficient: list[str] = []
    negative: list[str] = []
    if any(source is None for source in sources):
        invalid.append("missing-source-lineage")
    typed_sources = [source for source in sources if source is not None]
    if any(source.role is not SplitLabel.TRAINING for source in typed_sources):
        invalid.append("source-lineage-not-training")
    if any(
        source.archive_representation_family is not proposal.archive_representation_family
        for source in typed_sources
    ):
        invalid.append("source-lineage-archive-family-mismatch")
    if len({source.task_id for source in typed_sources}) < policy.minimum_training_tasks:
        insufficient.append("insufficient-independent-training-tasks")
    if (
        len({source.generator_family_id for source in typed_sources})
        < policy.minimum_training_generator_families
    ):
        insufficient.append("insufficient-independent-training-generator-families")
    relevant = [
        pair
        for pair in validation_pairs
        if pair.archive_representation_family is proposal.archive_representation_family
        and pair.lesson_proposal_id == proposal.proposal_id
    ]
    applied = [pair for pair in relevant if pair.treatment_memory_applied]
    if len({pair.task_id for pair in applied}) < policy.minimum_validation_tasks_with_exposure:
        insufficient.append("insufficient-validation-tasks-with-memory-exposure")
    if (
        len({pair.generator_family_id for pair in applied})
        < policy.minimum_validation_generator_families
    ):
        insufficient.append("insufficient-validation-generator-families-with-memory-exposure")
    if any(pair.baseline_exact and not pair.treatment_exact for pair in applied):
        negative.append("validation-exact-regression")
    family_gains: dict[str, list[int]] = {}
    for pair in applied:
        family_gains.setdefault(pair.generator_family_id, []).append(pair.difference_ppm)
    mean_gain = sum(pair.difference_ppm for pair in applied) // max(1, len(applied))
    if not insufficient:
        if any(sum(values) <= 0 for values in family_gains.values()):
            negative.append("nonpositive-gain-in-a-validation-family")
        if mean_gain <= 0:
            negative.append("nonpositive-mean-validation-gain")
    if invalid or negative:
        return LessonPromotionDecision(
            "rejected", tuple(sorted(set((*invalid, *negative, *insufficient)))), None
        )
    if insufficient:
        return LessonPromotionDecision("inconclusive", tuple(sorted(set(insufficient))), None)
    pair_hashes = tuple(sorted(sha256_json(pair.to_value()) for pair in applied))
    promoted = PromotedExperienceLesson(proposal, pair_hashes, mean_gain)
    return LessonPromotionDecision("promoted", (), promoted)


@dataclass(frozen=True, slots=True)
class BundleValidationPair:
    """Fresh-task comparison of the jointly retrieved promoted lesson bundle."""

    task_id: str
    generator_family_id: str
    seed: int
    baseline_exact: bool
    treatment_exact: bool
    baseline_normalized_exact_auc_ppm: int
    treatment_normalized_exact_auc_ppm: int
    applied_lesson_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.generator_family_id or self.seed < 0:
            raise ValueError("bundle validation pair identity is invalid")
        if tuple(sorted(set(self.applied_lesson_ids))) != self.applied_lesson_ids:
            raise ValueError("applied bundle lesson IDs must be unique and sorted")
        for lesson_id in self.applied_lesson_ids:
            _complete_hash(lesson_id, "applied bundle lesson ID")
        for value in (
            self.baseline_normalized_exact_auc_ppm,
            self.treatment_normalized_exact_auc_ppm,
        ):
            if not 0 <= value <= 1_000_000:
                raise ValueError("normalized exact AUC must be in [0, 1,000,000] ppm")

    @property
    def difference_ppm(self) -> int:
        return self.treatment_normalized_exact_auc_ppm - self.baseline_normalized_exact_auc_ppm

    def to_value(self) -> JsonObject:
        return {
            "validation_stage": ValidationStage.PROMOTED_BUNDLE.value,
            "task_id": self.task_id,
            "generator_family_id": self.generator_family_id,
            "seed": self.seed,
            "baseline_exact": self.baseline_exact,
            "treatment_exact": self.treatment_exact,
            "baseline_normalized_exact_auc_ppm": self.baseline_normalized_exact_auc_ppm,
            "treatment_normalized_exact_auc_ppm": self.treatment_normalized_exact_auc_ppm,
            "applied_lesson_ids": list(self.applied_lesson_ids),
        }


@dataclass(frozen=True, slots=True)
class ExperienceMemorySnapshot:
    protocol_hash: str
    lessons: tuple[PromotedExperienceLesson, ...]
    bundle_validation_pair_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _complete_hash(self.protocol_hash, "experience-memory protocol hash")
        if tuple(sorted(self.lessons, key=lambda lesson: lesson.lesson_id)) != self.lessons:
            raise ValueError("experience lessons must be content ordered")
        if len({lesson.lesson_id for lesson in self.lessons}) != len(self.lessons):
            raise ValueError("experience snapshot contains duplicate lessons")
        if (
            tuple(sorted(set(self.bundle_validation_pair_hashes)))
            != self.bundle_validation_pair_hashes
        ):
            raise ValueError("bundle validation pair hashes must be unique and sorted")
        for pair_hash in self.bundle_validation_pair_hashes:
            _complete_hash(pair_hash, "bundle validation pair hash")

    def to_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "snapshot_version": EXPERIENCE_SNAPSHOT_VERSION,
                "protocol_hash": self.protocol_hash,
                "lessons": [lesson.safe_value() for lesson in self.lessons],
                "bundle_validation_pair_hashes": list(self.bundle_validation_pair_hashes),
            },
        )

    @property
    def snapshot_hash(self) -> str:
        return sha256_json(self.to_value())


@dataclass(frozen=True, slots=True)
class BundleValidationDecision:
    status: str
    reasons: tuple[str, ...]
    snapshot: ExperienceMemorySnapshot | None


def evaluate_bundle_validation(
    *,
    protocol_hash: str,
    promoted_lessons: tuple[PromotedExperienceLesson, ...],
    individual_screen_task_ids: frozenset[str],
    validation_pairs: tuple[BundleValidationPair, ...],
    minimum_validation_tasks_with_exposure: int = 4,
    minimum_validation_generator_families: int = 1,
    minimum_exposure_tasks_per_lesson: int = 2,
) -> BundleValidationDecision:
    """Confirm the promoted bundle on fresh tasks before freezing a snapshot."""

    _complete_hash(protocol_hash, "experience-memory protocol hash")
    if (
        min(
            minimum_validation_tasks_with_exposure,
            minimum_validation_generator_families,
            minimum_exposure_tasks_per_lesson,
        )
        < 1
    ):
        raise ValueError("bundle validation thresholds must be positive")
    if not promoted_lessons:
        return BundleValidationDecision("inconclusive", ("no-lessons-passed-screening",), None)
    if len({lesson.lesson_id for lesson in promoted_lessons}) != len(promoted_lessons):
        raise ValueError("bundle contains duplicate promoted lessons")
    expected_ids = {lesson.lesson_id for lesson in promoted_lessons}
    invalid: list[str] = []
    insufficient: list[str] = []
    negative: list[str] = []
    if any(pair.task_id in individual_screen_task_ids for pair in validation_pairs):
        invalid.append("bundle-validation-reuses-individual-screen-task")
    if any(set(pair.applied_lesson_ids) - expected_ids for pair in validation_pairs):
        invalid.append("bundle-validation-applied-unknown-lesson")
    applied = [pair for pair in validation_pairs if pair.applied_lesson_ids]
    if len({pair.task_id for pair in applied}) < minimum_validation_tasks_with_exposure:
        insufficient.append("insufficient-bundle-validation-tasks-with-memory-exposure")
    if len({pair.generator_family_id for pair in applied}) < minimum_validation_generator_families:
        insufficient.append("insufficient-bundle-validation-generator-families")
    for lesson_id in expected_ids:
        exposed_tasks = {pair.task_id for pair in applied if lesson_id in pair.applied_lesson_ids}
        if len(exposed_tasks) < minimum_exposure_tasks_per_lesson:
            insufficient.append("insufficient-per-lesson-bundle-exposure")
            break
    if any(pair.baseline_exact and not pair.treatment_exact for pair in applied):
        negative.append("bundle-validation-exact-regression")
    mean_gain = sum(pair.difference_ppm for pair in applied) // max(1, len(applied))
    family_gains: dict[str, list[int]] = {}
    for pair in applied:
        family_gains.setdefault(pair.generator_family_id, []).append(pair.difference_ppm)
    if not insufficient:
        if mean_gain <= 0:
            negative.append("nonpositive-mean-bundle-validation-gain")
        if any(sum(values) <= 0 for values in family_gains.values()):
            negative.append("nonpositive-bundle-gain-in-a-validation-family")
    if invalid or negative:
        return BundleValidationDecision(
            "rejected", tuple(sorted(set((*invalid, *negative, *insufficient)))), None
        )
    if insufficient:
        return BundleValidationDecision("inconclusive", tuple(sorted(set(insufficient))), None)
    pair_hashes = tuple(sorted(sha256_json(pair.to_value()) for pair in applied))
    lessons = tuple(sorted(promoted_lessons, key=lambda lesson: lesson.lesson_id))
    return BundleValidationDecision(
        "promoted",
        (),
        ExperienceMemorySnapshot(protocol_hash, lessons, pair_hashes),
    )


@dataclass(frozen=True, slots=True)
class ExperienceRetrievalRecord:
    query_id: str
    snapshot_hash: str
    selected_archive_coordinate: ArchiveCoordinate
    eligible_lesson_ids: tuple[str, ...]
    selected_lesson_ids: tuple[str, ...]
    exclusions: tuple[tuple[str, str], ...]
    rendered_memory: str
    rendered_bytes: int
    conservative_token_bound: int

    def to_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "retrieval_version": EXPERIENCE_RETRIEVAL_VERSION,
                "query_id": self.query_id,
                "snapshot_hash": self.snapshot_hash,
                "selected_archive_coordinate": self.selected_archive_coordinate.to_value(),
                "eligible_lesson_ids": list(self.eligible_lesson_ids),
                "selected_lesson_ids": list(self.selected_lesson_ids),
                "exclusions": [
                    {"lesson_id": lesson_id, "reason": reason}
                    for lesson_id, reason in self.exclusions
                ],
                "rendered_memory_utf8": self.rendered_memory,
                "rendered_bytes": self.rendered_bytes,
                "conservative_token_bound": self.conservative_token_bound,
            },
        )


def retrieve_experience_for_cell(
    *,
    snapshot: ExperienceMemorySnapshot,
    selected_cell: ArchiveCoordinate,
    max_items: int,
    max_bytes: int,
    max_tokens: int,
) -> ExperienceRetrievalRecord:
    """Retrieve only lessons scoped to the selected public archive representation family."""

    if min(max_items, max_bytes, max_tokens) < 0:
        raise ValueError("experience retrieval bounds must be nonnegative")
    family = selected_cell.representation_family
    query_id = sha256_json(
        {
            "retrieval_version": EXPERIENCE_RETRIEVAL_VERSION,
            "snapshot_hash": snapshot.snapshot_hash,
            "selected_archive_coordinate": selected_cell.to_value(),
            "bounds": {"items": max_items, "bytes": max_bytes, "tokens": max_tokens},
        }
    )

    def block(lessons: tuple[PromotedExperienceLesson, ...]) -> JsonObject:
        return cast(
            JsonObject,
            {
                "memory_block_version": EXPERIENCE_MEMORY_BLOCK_VERSION,
                "retrieval_query_id": query_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "selected_archive_coordinate": selected_cell.to_value(),
                "selected_archive_representation_family": family.value,
                "items": [lesson.safe_value() for lesson in lessons],
            },
        )

    eligible = [
        lesson
        for lesson in snapshot.lessons
        if lesson.proposal.archive_representation_family is family
    ]
    ranked = sorted(
        eligible, key=lambda lesson: (-lesson.validation_mean_gain_ppm, lesson.lesson_id)
    )
    exclusions: list[tuple[str, str]] = [
        (lesson.lesson_id, "archive-representation-family-mismatch")
        for lesson in snapshot.lessons
        if lesson.proposal.archive_representation_family is not family
    ]
    selected: list[PromotedExperienceLesson] = []
    for lesson in ranked:
        if len(selected) >= max_items:
            exclusions.append((lesson.lesson_id, "item-bound"))
            continue
        candidate = canonical_json(block((*selected, lesson)))
        size = len(candidate.encode("utf-8"))
        if size > max_bytes:
            exclusions.append((lesson.lesson_id, "byte-bound"))
            continue
        if size > max_tokens:
            exclusions.append((lesson.lesson_id, "conservative-token-bound"))
            continue
        selected.append(lesson)
    rendered = canonical_json(block(tuple(selected)))
    rendered_bytes = len(rendered.encode("utf-8"))
    if rendered_bytes > max_bytes or rendered_bytes > max_tokens:
        raise ValueError("retrieval bounds cannot hold the explicit experience-memory block")
    return ExperienceRetrievalRecord(
        query_id=query_id,
        snapshot_hash=snapshot.snapshot_hash,
        selected_archive_coordinate=selected_cell,
        eligible_lesson_ids=tuple(lesson.lesson_id for lesson in ranked),
        selected_lesson_ids=tuple(lesson.lesson_id for lesson in selected),
        exclusions=tuple(sorted(exclusions)),
        rendered_memory=rendered,
        rendered_bytes=rendered_bytes,
        conservative_token_bound=rendered_bytes,
    )
