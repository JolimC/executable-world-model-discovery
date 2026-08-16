"""Deterministic storage, similarity, retrieval, exposure, and rendering for memory v3."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from world_model_search.memory.contextual import (
    MEMORY_BLOCK_SCHEMA,
    MEMORY_BLOCK_SCHEMA_V2,
    RETRIEVAL_SCHEMA,
    ContextFeatures,
    ContextMode,
    ExperienceRecord,
    PromptProjection,
    RetrievalMode,
    SelectionPolicy,
)
from world_model_search.persistence.artifacts import read_text_artifact, write_content_artifact
from world_model_search.serialization import (
    JsonObject,
    JsonValue,
    canonical_json,
    parse_json_object,
    sha256_json,
)

SIMILARITY_SCALE = 1_000_000
SNAPSHOT_SCHEMA = "contextual-experience-snapshot-v1"
RUNTIME_SCHEMA = "contextual-experience-runtime-v1"


def _bounded_ppm(value: int) -> int:
    return max(0, min(SIMILARITY_SCALE, value))


@dataclass(frozen=True, slots=True)
class SimilarityWeights:
    representation_family: int = 400
    behavior_cluster: int = 150
    error_pattern: int = 0
    score_error_count: int = 150
    ast_constructor_histogram: int = 100
    structural_motifs: int = 100
    plateau: int = 50
    search_stage: int = 50

    def __post_init__(self) -> None:
        values = self.to_value()
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise ValueError("similarity weights must be nonnegative integers")
        if sum(cast(int, value) for value in values.values()) < 1:
            raise ValueError("at least one similarity weight must be positive")

    def to_value(self) -> JsonObject:
        return {
            "representation_family": self.representation_family,
            "behavior_cluster": self.behavior_cluster,
            "error_pattern": self.error_pattern,
            "score_error_count": self.score_error_count,
            "ast_constructor_histogram": self.ast_constructor_histogram,
            "structural_motifs": self.structural_motifs,
            "plateau": self.plateau,
            "search_stage": self.search_stage,
        }


@dataclass(frozen=True, slots=True)
class RandomizedExposureConfig:
    enabled: bool = False
    probability_numerator: int = 1
    probability_denominator: int = 1
    randomization_seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.probability_numerator, bool)
            or isinstance(self.probability_denominator, bool)
            or self.probability_denominator < 1
            or not 0 <= self.probability_numerator <= self.probability_denominator
        ):
            raise ValueError("exposure probability must be an exact fraction in [0,1]")

    def to_value(self) -> JsonObject:
        return {
            "enabled": self.enabled,
            "probability_numerator": self.probability_numerator,
            "probability_denominator": self.probability_denominator,
            "randomization_seed": self.randomization_seed,
        }


@dataclass(frozen=True, slots=True)
class ContextualMemoryConfig:
    retrieval_mode: RetrievalMode = RetrievalMode.CONTRASTIVE
    context_mode: ContextMode = ContextMode.RICH
    weights: SimilarityWeights = SimilarityWeights()
    minimum_similarity_ppm: int = 300_000
    max_memory_records: int = 3
    max_memory_bytes: int = 4096
    max_memory_tokens_conservative: int = 4096
    include_diverse_third: bool = False
    short_horizon_steps: int = 8
    exposure: RandomizedExposureConfig = RandomizedExposureConfig()
    prompt_projection: PromptProjection = PromptProjection.FULL_GREEDY_V1
    selection_policy: SelectionPolicy = SelectionPolicy.SIMILARITY_RECORD_ID_V1
    include_aggregate_summary: bool = False
    aggregate_max_signatures: int = 6

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_similarity_ppm <= SIMILARITY_SCALE:
            raise ValueError("minimum similarity must be a parts-per-million value")
        if (
            min(
                self.max_memory_records,
                self.max_memory_bytes,
                self.max_memory_tokens_conservative,
                self.short_horizon_steps,
                self.aggregate_max_signatures,
            )
            < 1
        ):
            raise ValueError("memory and horizon bounds must be positive")
        if (
            self.include_aggregate_summary
            and self.prompt_projection is not PromptProjection.COMPACT_ADAPTIVE_V2
        ):
            raise ValueError("aggregate evidence summaries require the compact-adaptive projection")
        if (
            self.include_diverse_third
            and self.selection_policy is SelectionPolicy.TASK_BALANCED_CONTRAST_V2
        ):
            raise ValueError("include_diverse_third applies only to the v1 selection policy")

    def to_value(self) -> JsonObject:
        return {
            "schema_version": "contextual-memory-config-v2",
            "retrieval_mode": self.retrieval_mode.value,
            "context_mode": self.context_mode.value,
            "weights": self.weights.to_value(),
            "minimum_similarity_ppm": self.minimum_similarity_ppm,
            "max_memory_records": self.max_memory_records,
            "max_memory_bytes": self.max_memory_bytes,
            "max_memory_tokens_conservative": self.max_memory_tokens_conservative,
            "include_diverse_third": self.include_diverse_third,
            "short_horizon_steps": self.short_horizon_steps,
            "exposure": self.exposure.to_value(),
            "prompt_projection": self.prompt_projection.value,
            "selection_policy": self.selection_policy.value,
            "include_aggregate_summary": self.include_aggregate_summary,
            "aggregate_max_signatures": self.aggregate_max_signatures,
        }


@dataclass(frozen=True, slots=True)
class ExperienceSnapshot:
    source_task_ids: tuple[str, ...]
    records: tuple[ExperienceRecord, ...]
    source_artifact_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.source_task_ids
            or tuple(sorted(set(self.source_task_ids))) != self.source_task_ids
        ):
            raise ValueError("snapshot needs sorted unique source task IDs")
        if len({record.record_id for record in self.records}) != len(self.records):
            raise ValueError("snapshot contains duplicate experience records")
        source_ids = set(self.source_task_ids)
        for record in self.records:
            if (
                record.provenance.task_id not in source_ids
                or not record.memory_metadata.training_eligible
                or record.memory_metadata.sealed_test
                or record.provenance.source_split not in {"training", "development"}
                or record.downstream_outcome.annotation_status != "complete"
            ):
                raise ValueError("snapshot includes ineligible, sealed, or unannotated experience")

    @property
    def snapshot_hash(self) -> str:
        return sha256_json(self.to_value(include_hash=False))

    def to_value(self, *, include_hash: bool = True) -> JsonObject:
        value: JsonObject = {
            "schema_version": SNAPSHOT_SCHEMA,
            "source_task_ids": list(self.source_task_ids),
            "source_artifact_hashes": list(self.source_artifact_hashes),
            "records": [record.to_value() for record in self.records],
        }
        if include_hash:
            value["snapshot_hash"] = self.snapshot_hash
        return value

    @classmethod
    def from_value(cls, value: object) -> ExperienceSnapshot:
        if not isinstance(value, dict) or value.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError("contextual experience snapshot schema is invalid")
        source_ids = value.get("source_task_ids")
        hashes = value.get("source_artifact_hashes")
        records = value.get("records")
        if (
            not isinstance(source_ids, list)
            or not isinstance(hashes, list)
            or not isinstance(records, list)
        ):
            raise ValueError("contextual experience snapshot fields are malformed")
        snapshot = cls(
            tuple(str(item) for item in source_ids),
            tuple(ExperienceRecord.from_value(item) for item in records),
            tuple(str(item) for item in hashes),
        )
        if snapshot.snapshot_hash != value.get("snapshot_hash"):
            raise ValueError("contextual experience snapshot hash mismatch")
        return snapshot

    def write(self, path: Path) -> str:
        return write_content_artifact(path, canonical_json(self.to_value()))

    @classmethod
    def read(cls, path: Path) -> ExperienceSnapshot:
        return cls.from_value(parse_json_object(read_text_artifact(path)))


@dataclass(frozen=True, slots=True)
class SimilarityComponent:
    raw_similarity_ppm: int
    configured_weight: int
    effective_weight: int
    weighted_contribution: int
    available: bool

    def to_value(self) -> JsonObject:
        return {
            "raw_similarity_ppm": self.raw_similarity_ppm,
            "configured_weight": self.configured_weight,
            "effective_weight": self.effective_weight,
            "weighted_contribution": self.weighted_contribution,
            "available": self.available,
        }


@dataclass(frozen=True, slots=True)
class ScoredExperience:
    record_id: str
    outcome_class: str
    total_similarity_ppm: int
    retrieved_rank: int
    components: JsonObject

    def to_value(self) -> JsonObject:
        return {
            "record_id": self.record_id,
            "outcome_class": self.outcome_class,
            "total_similarity_ppm": self.total_similarity_ppm,
            "retrieved_rank": self.retrieved_rank,
            "components": self.components,
        }


@dataclass(frozen=True, slots=True)
class ExposureDecision:
    eligible_record_id: str
    retrieved_rank: int
    inclusion_probability_numerator: int
    inclusion_probability_denominator: int
    shown: bool
    randomization_seed: int

    def to_value(self) -> JsonObject:
        return {
            "eligible_record_id": self.eligible_record_id,
            "retrieved_rank": self.retrieved_rank,
            "inclusion_probability_numerator": self.inclusion_probability_numerator,
            "inclusion_probability_denominator": self.inclusion_probability_denominator,
            "shown": self.shown,
            "randomization_seed": self.randomization_seed,
        }


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    current_task_id: str
    search_seed: int
    retrieval_index: int
    context: ContextFeatures
    config: ContextualMemoryConfig
    all_eligible_scores: tuple[ScoredExperience, ...]
    selected_record_ids: tuple[str, ...]
    exposure_decisions: tuple[ExposureDecision, ...]

    @property
    def shown_record_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.eligible_record_id for decision in self.exposure_decisions if decision.shown
        )

    def to_value(self, *, actually_shown_record_ids: tuple[str, ...] | None = None) -> JsonObject:
        shown_ids = (
            self.shown_record_ids
            if actually_shown_record_ids is None
            else actually_shown_record_ids
        )
        shown_set = set(shown_ids)
        exposure_values: list[JsonObject] = []
        for item in self.exposure_decisions:
            value = item.to_value()
            value["shown"] = item.eligible_record_id in shown_set
            exposure_values.append(value)
        return {
            "schema_version": RETRIEVAL_SCHEMA,
            "current_task_id": self.current_task_id,
            "search_seed": self.search_seed,
            "retrieval_index": self.retrieval_index,
            "context": self.context.to_value(),
            "config": self.config.to_value(),
            "all_eligible_scores": [item.to_value() for item in self.all_eligible_scores],
            "selected_record_ids": list(self.selected_record_ids),
            "exposure_decisions": cast(list[JsonValue], exposure_values),
            "shown_record_ids": list(shown_ids),
        }


class ExperienceStore:
    def __init__(self, snapshot: ExperienceSnapshot | None) -> None:
        self.snapshot = snapshot
        self._records = (
            {record.record_id: record for record in snapshot.records} if snapshot else {}
        )

    def get(self, record_id: str) -> ExperienceRecord:
        return self._records[record_id]

    def eligible(
        self,
        *,
        current_task_id: str,
        forbidden_task_ids: frozenset[str],
    ) -> tuple[ExperienceRecord, ...]:
        if self.snapshot is None:
            return ()
        if current_task_id in set(self.snapshot.source_task_ids):
            raise ValueError("current target task is present in the frozen memory source")
        return tuple(
            record
            for record in self.snapshot.records
            if record.provenance.task_id != current_task_id
            and record.provenance.task_id not in forbidden_task_ids
            and record.memory_metadata.training_eligible
            and not record.memory_metadata.sealed_test
            and record.provenance.source_split in {"training", "development"}
        )


def _jaccard(left: set[str], right: set[str]) -> int:
    union = left | right
    return SIMILARITY_SCALE if not union else len(left & right) * SIMILARITY_SCALE // len(union)


def _histogram(context: ContextFeatures) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, value in context.constructor_histogram.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("constructor histogram must contain nonnegative integers")
        result[name] = value
    return result


def _histogram_similarity(left: ContextFeatures, right: ContextFeatures) -> int:
    left_counts, right_counts = _histogram(left), _histogram(right)
    keys = set(left_counts) | set(right_counts)
    denominator = sum(max(left_counts.get(key, 0), right_counts.get(key, 0)) for key in keys)
    numerator = sum(min(left_counts.get(key, 0), right_counts.get(key, 0)) for key in keys)
    return SIMILARITY_SCALE if denominator == 0 else numerator * SIMILARITY_SCALE // denominator


def classify_outcome(record: ExperienceRecord) -> str:
    """Deterministic positive/neutral/negative label shared by retrieval and analysis."""

    outcome = record.immediate_outcome
    downstream = record.downstream_outcome
    if (
        outcome.score_delta > 0
        or outcome.exact_solution
        or downstream.eventually_had_exact_descendant
        or downstream.max_descendant_improvement > 0
    ):
        return "positive"
    if (
        outcome.score_delta < 0
        or outcome.exact_regression
        or outcome.archive_outcome in {"rejected", "duplicate"}
        or outcome.canonical_duplicate
        or outcome.semantic_duplicate
    ):
        return "negative"
    return "neutral"


def _component(
    raw: int,
    weight: int,
    *,
    available: bool,
    enabled: bool,
) -> SimilarityComponent:
    effective = weight if available and enabled else 0
    bounded = _bounded_ppm(raw) if available else 0
    return SimilarityComponent(bounded, weight, effective, bounded * effective, available)


def contextual_similarity(
    current: ContextFeatures,
    remembered: ContextFeatures,
    *,
    config: ContextualMemoryConfig,
) -> tuple[int, JsonObject]:
    rich = config.context_mode is ContextMode.RICH
    weights = config.weights
    pattern_available = (
        current.parent_error_pattern is not None and remembered.parent_error_pattern is not None
    )
    components = {
        "representation_family_match": _component(
            SIMILARITY_SCALE
            if current.representation_family == remembered.representation_family
            else 0,
            weights.representation_family,
            available=True,
            enabled=True,
        ),
        "behavior_cluster_match": _component(
            SIMILARITY_SCALE if current.behavior_cluster == remembered.behavior_cluster else 0,
            weights.behavior_cluster,
            available=True,
            enabled=rich,
        ),
        "error_pattern_similarity": _component(
            _jaccard(
                set(current.parent_error_pattern or ()), set(remembered.parent_error_pattern or ())
            ),
            weights.error_pattern,
            available=pattern_available,
            enabled=rich,
        ),
        "score_error_count_similarity": _component(
            SIMILARITY_SCALE
            - abs(current.parent_error_count - remembered.parent_error_count)
            * SIMILARITY_SCALE
            // 8,
            weights.score_error_count,
            available=True,
            enabled=rich,
        ),
        "ast_constructor_histogram_similarity": _component(
            _histogram_similarity(current, remembered),
            weights.ast_constructor_histogram,
            available=True,
            enabled=rich,
        ),
        "structural_motif_similarity": _component(
            _jaccard(set(current.structural_motifs), set(remembered.structural_motifs)),
            weights.structural_motifs,
            available=True,
            enabled=rich,
        ),
        "plateau_similarity": _component(
            SIMILARITY_SCALE - abs(current.plateau_length - remembered.plateau_length) * 250_000,
            weights.plateau,
            available=True,
            enabled=rich,
        ),
        "search_stage_similarity": _component(
            SIMILARITY_SCALE - abs(current.search_step - remembered.search_step) * 25_000,
            weights.search_stage,
            available=True,
            enabled=rich,
        ),
    }
    denominator = sum(item.effective_weight for item in components.values())
    total = (
        sum(item.weighted_contribution for item in components.values()) // denominator
        if denominator
        else 0
    )
    return total, {name: item.to_value() for name, item in components.items()}


def _action_signature(record: ExperienceRecord) -> tuple[str, ...]:
    return tuple(item.value for item in record.action.edit_classes)


def _select_task_balanced(
    relevant: tuple[tuple[ExperienceRecord, int, int], ...],
    config: ContextualMemoryConfig,
) -> list[tuple[ExperienceRecord, int]]:
    """Deterministically spread selected evidence across source tasks and action signatures."""

    remaining = list(relevant)
    selected: list[tuple[ExperienceRecord, int]] = []
    task_use: Counter[str] = Counter()
    signature_use: Counter[tuple[str, ...]] = Counter()

    def take(item: tuple[ExperienceRecord, int, int]) -> None:
        record, _score, rank = item
        selected.append((record, rank))
        remaining.remove(item)
        task_use[record.provenance.task_id] += 1
        signature_use[_action_signature(record)] += 1

    if config.retrieval_mode is RetrievalMode.CONTRASTIVE:
        for label in ("positive", "negative"):
            match = next(
                (item for item in remaining if classify_outcome(item[0]) == label),
                None,
            )
            if match is not None:
                take(match)
        allowed = {"positive", "negative"}
    else:
        allowed = {"positive"}
    while len(selected) < config.max_memory_records:
        candidates = [item for item in remaining if classify_outcome(item[0]) in allowed]
        if not candidates:
            break
        take(
            min(
                candidates,
                key=lambda item: (
                    task_use[item[0].provenance.task_id],
                    -item[1],
                    signature_use[_action_signature(item[0])],
                    item[0].record_id,
                ),
            )
        )
    return selected


class MemoryExposurePolicy:
    def decide(
        self,
        *,
        record_id: str,
        rank: int,
        current_task_id: str,
        search_seed: int,
        retrieval_index: int,
        config: RandomizedExposureConfig,
    ) -> ExposureDecision:
        numerator = config.probability_numerator if config.enabled else 1
        denominator = config.probability_denominator if config.enabled else 1
        draw = int(
            sha256_json(
                {
                    "policy": "reproducible-memory-exposure-v1",
                    "record_id": record_id,
                    "current_task_id": current_task_id,
                    "search_seed": search_seed,
                    "retrieval_index": retrieval_index,
                    "randomization_seed": config.randomization_seed,
                }
            ),
            16,
        )
        shown = draw % denominator < numerator
        return ExposureDecision(
            record_id,
            rank,
            numerator,
            denominator,
            shown,
            config.randomization_seed,
        )


class ExperienceRetriever:
    def retrieve(
        self,
        *,
        store: ExperienceStore,
        current_context: ContextFeatures,
        current_task_id: str,
        forbidden_task_ids: frozenset[str],
        search_seed: int,
        retrieval_index: int,
        config: ContextualMemoryConfig,
    ) -> RetrievalDecision:
        eligible = store.eligible(
            current_task_id=current_task_id,
            forbidden_task_ids=forbidden_task_ids,
        )
        ranked_values: list[tuple[ExperienceRecord, int, JsonObject]] = []
        for record in eligible:
            similarity, components = contextual_similarity(
                current_context,
                record.context,
                config=config,
            )
            ranked_values.append((record, similarity, components))
        ranked_values.sort(key=lambda item: (-item[1], item[0].record_id))
        scored = tuple(
            ScoredExperience(record.record_id, classify_outcome(record), score, rank, components)
            for rank, (record, score, components) in enumerate(ranked_values, start=1)
        )
        relevant = tuple(
            (record, score, rank)
            for rank, (record, score, _components) in enumerate(ranked_values, start=1)
            if score >= config.minimum_similarity_ppm
        )
        selected: list[tuple[ExperienceRecord, int]] = []
        if (
            config.retrieval_mode is not RetrievalMode.DISABLED
            and config.selection_policy is SelectionPolicy.TASK_BALANCED_CONTRAST_V2
        ):
            selected = _select_task_balanced(relevant, config)
        elif config.retrieval_mode is RetrievalMode.POSITIVE_ONLY:
            selected.extend(
                (record, rank)
                for record, _score, rank in relevant
                if classify_outcome(record) == "positive"
            )
            selected = selected[:1]
        elif config.retrieval_mode is RetrievalMode.CONTRASTIVE:
            for label in ("positive", "negative"):
                match = next(
                    (
                        (record, rank)
                        for record, _score, rank in relevant
                        if classify_outcome(record) == label
                    ),
                    None,
                )
                if match is not None:
                    selected.append(match)
            if config.include_diverse_third and len(selected) < config.max_memory_records:
                used_ids = {record.record_id for record, _rank in selected}
                used_edits = {
                    edit.value for record, _rank in selected for edit in record.action.edit_classes
                }
                diverse = next(
                    (
                        (record, rank)
                        for record, _score, rank in relevant
                        if record.record_id not in used_ids
                        and any(edit.value not in used_edits for edit in record.action.edit_classes)
                    ),
                    None,
                )
                if diverse is not None:
                    selected.append(diverse)
        selected = selected[: config.max_memory_records]
        exposures = tuple(
            MemoryExposurePolicy().decide(
                record_id=record.record_id,
                rank=rank,
                current_task_id=current_task_id,
                search_seed=search_seed,
                retrieval_index=retrieval_index,
                config=config.exposure,
            )
            for record, rank in selected
        )
        return RetrievalDecision(
            current_task_id,
            search_seed,
            retrieval_index,
            current_context,
            config,
            scored,
            tuple(record.record_id for record, _rank in selected),
            exposures,
        )


@dataclass(frozen=True, slots=True)
class RenderedMemoryBlock:
    canonical_json_block: str
    shown_record_ids: tuple[str, ...]
    byte_count: int
    conservative_token_count: int
    projection: str = PromptProjection.FULL_GREEDY_V1.value
    detail_level: str = "full"
    dropped_record_ids: tuple[str, ...] = ()
    includes_aggregate: bool = False

    @property
    def value(self) -> JsonObject:
        return parse_json_object(self.canonical_json_block)


def _proposer_safe_record(
    record: ExperienceRecord,
    score: ScoredExperience,
) -> JsonObject:
    return {
        "record_kind": score.outcome_class,
        "context_similarity_ppm": score.total_similarity_ppm,
        "context": {
            "representation_family": record.context.representation_family,
            "parent_score": record.context.parent_score,
            "parent_error_count": record.context.parent_error_count,
            "parent_error_pattern": (
                list(record.context.parent_error_pattern)
                if record.context.parent_error_pattern is not None
                else None
            ),
            "public_probe_behavior": record.context.public_probe_behavior,
            "behavior_cluster": record.context.behavior_cluster,
            "canonical_size": record.context.parent_canonical_size,
            "constructor_histogram": record.context.constructor_histogram,
            "structural_motifs": list(record.context.structural_motifs),
            "plateau_length": record.context.plateau_length,
            "search_step": record.context.search_step,
            "recent_edit_classes": list(record.context.recent_edit_classes),
        },
        "action": record.action.to_value(),
        "outcome": {
            "immediate": record.immediate_outcome.to_value(),
            "downstream_descriptive_not_causal": record.downstream_outcome.to_value(),
        },
    }


def _compact_record(
    record: ExperienceRecord,
    score: ScoredExperience,
) -> JsonObject:
    action = record.action
    return {
        "record_kind": score.outcome_class,
        "context_similarity_ppm": score.total_similarity_ppm,
        "context": {
            "representation_family": record.context.representation_family,
            "parent_score": record.context.parent_score,
            "parent_error_count": record.context.parent_error_count,
            "public_probe_behavior": record.context.public_probe_behavior,
            "behavior_cluster": record.context.behavior_cluster,
            "canonical_size": record.context.parent_canonical_size,
            "constructor_histogram": record.context.constructor_histogram,
            "plateau_length": record.context.plateau_length,
            "search_step": record.context.search_step,
        },
        "action": {
            "normalized_edit_classes": [item.value for item in action.edit_classes],
            "constructors_added": list(action.constructors_added),
            "constructors_removed": list(action.constructors_removed),
            "operators_replaced": [list(pair) for pair in action.operators_replaced],
            "representation_family_before": action.representation_family_before,
            "representation_family_after": action.representation_family_after,
            "canonical_size_delta": action.size_delta,
            "affected_subtree_path_count": len(action.affected_paths),
            "affected_subtree_paths_prefix": [list(path) for path in action.affected_paths[:3]],
        },
        "outcome": {
            "score_delta": record.immediate_outcome.score_delta,
            "child_score": record.immediate_outcome.child_score,
            "exact_solution": record.immediate_outcome.exact_solution,
            "archive_outcome": record.immediate_outcome.archive_outcome,
            "duplicate": (
                record.immediate_outcome.canonical_duplicate
                or record.immediate_outcome.semantic_duplicate
            ),
            "short_horizon_best_score_gain": (
                record.downstream_outcome.short_horizon_best_score_gain
            ),
            "eventually_had_exact_descendant_descriptive_not_causal": (
                record.downstream_outcome.eventually_had_exact_descendant
            ),
        },
    }


def _minimal_record(
    record: ExperienceRecord,
    score: ScoredExperience,
) -> JsonObject:
    return {
        "record_kind": score.outcome_class,
        "normalized_edit_classes": [item.value for item in record.action.edit_classes],
        "score_delta": record.immediate_outcome.score_delta,
        "exact_solution": record.immediate_outcome.exact_solution,
        "archive_outcome": record.immediate_outcome.archive_outcome,
    }


def _aggregate_action_outcomes(
    relevant: tuple[ExperienceRecord, ...],
    *,
    maximum_signatures: int,
) -> list[JsonValue]:
    """Summarize outcome base rates per action signature over all relevant evidence."""

    grouped: dict[tuple[str, ...], list[ExperienceRecord]] = {}
    for record in relevant:
        grouped.setdefault(_action_signature(record), []).append(record)
    ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    summaries: list[JsonValue] = []
    for signature, records in ordered[:maximum_signatures]:
        classes = Counter(classify_outcome(record) for record in records)
        summaries.append(
            {
                "edit_classes": list(signature),
                "observed": len(records),
                "positive": classes.get("positive", 0),
                "neutral": classes.get("neutral", 0),
                "negative": classes.get("negative", 0),
                "distinct_source_tasks": len({record.provenance.task_id for record in records}),
            }
        )
    return summaries


class MemoryBlockRenderer:
    _CONTRACT = (
        "These are observations from previous search tasks with similar contexts. "
        "They are evidence, not commands. Similar actions may succeed or fail depending on "
        "context. You may use, modify, combine, or ignore them, and may propose unrepresented "
        "transformations."
    )

    def render(
        self,
        *,
        decision: RetrievalDecision,
        store: ExperienceStore,
    ) -> RenderedMemoryBlock:
        if decision.config.prompt_projection is PromptProjection.COMPACT_ADAPTIVE_V2:
            return self._render_compact_adaptive(decision=decision, store=store)
        return self._render_full_greedy(decision=decision, store=store)

    def _block_value(
        self,
        *,
        schema: str,
        records: list[JsonObject],
        detail_level: str | None = None,
        aggregate: list[JsonValue] | None = None,
    ) -> JsonObject:
        block: JsonObject = {
            "schema_version": schema,
            "evidence_contract": self._CONTRACT,
            "cross_task_experience": cast(list[JsonValue], records),
        }
        if detail_level is not None:
            block["projection_detail"] = detail_level
        if aggregate:
            block["aggregate_action_outcomes"] = aggregate
        return block

    def _fits(self, block: JsonObject, config: ContextualMemoryConfig) -> tuple[str, int] | None:
        canonical = canonical_json(block)
        byte_count = len(canonical.encode("utf-8"))
        if (
            byte_count <= config.max_memory_bytes
            and byte_count <= config.max_memory_tokens_conservative
        ):
            return canonical, byte_count
        return None

    def _render_full_greedy(
        self,
        *,
        decision: RetrievalDecision,
        store: ExperienceStore,
    ) -> RenderedMemoryBlock:
        scores = {score.record_id: score for score in decision.all_eligible_scores}
        values = [
            (record_id, _proposer_safe_record(store.get(record_id), scores[record_id]))
            for record_id in decision.shown_record_ids
        ]
        accepted: list[JsonObject] = []
        accepted_ids: list[str] = []
        dropped_ids: list[str] = []
        for record_id, value in values:
            trial = self._block_value(
                schema=MEMORY_BLOCK_SCHEMA,
                records=[*accepted, value],
            )
            if (
                len(accepted) < decision.config.max_memory_records
                and self._fits(trial, decision.config) is not None
            ):
                accepted.append(value)
                accepted_ids.append(record_id)
            else:
                dropped_ids.append(record_id)
        block = self._block_value(schema=MEMORY_BLOCK_SCHEMA, records=accepted)
        fitted = self._fits(block, decision.config)
        if fitted is None:
            raise ValueError("memory bounds are too small for the canonical empty block")
        canonical, byte_count = fitted
        return RenderedMemoryBlock(
            canonical,
            tuple(accepted_ids),
            byte_count,
            byte_count,
            projection=PromptProjection.FULL_GREEDY_V1.value,
            detail_level="full",
            dropped_record_ids=tuple(dropped_ids),
            includes_aggregate=False,
        )

    def _render_compact_adaptive(
        self,
        *,
        decision: RetrievalDecision,
        store: ExperienceStore,
    ) -> RenderedMemoryBlock:
        config = decision.config
        scores = {score.record_id: score for score in decision.all_eligible_scores}
        shown = [(record_id, store.get(record_id)) for record_id in decision.shown_record_ids]
        relevant = tuple(
            store.get(score.record_id)
            for score in decision.all_eligible_scores
            if score.total_similarity_ppm >= config.minimum_similarity_ppm
        )
        aggregate = (
            _aggregate_action_outcomes(relevant, maximum_signatures=config.aggregate_max_signatures)
            if config.include_aggregate_summary
            else []
        )
        renderers = {"compact": _compact_record, "minimal": _minimal_record}
        attempts: list[tuple[str, bool]] = []
        if aggregate:
            attempts.append(("compact", True))
        attempts.extend((("compact", False), ("minimal", False)))
        for detail_level, with_aggregate in attempts:
            records = [
                renderers[detail_level](record, scores[record_id]) for record_id, record in shown
            ]
            block = self._block_value(
                schema=MEMORY_BLOCK_SCHEMA_V2,
                records=records,
                detail_level=detail_level,
                aggregate=aggregate if with_aggregate else None,
            )
            fitted = self._fits(block, config)
            if fitted is not None:
                canonical, byte_count = fitted
                return RenderedMemoryBlock(
                    canonical,
                    tuple(record_id for record_id, _record in shown),
                    byte_count,
                    byte_count,
                    projection=PromptProjection.COMPACT_ADAPTIVE_V2.value,
                    detail_level=detail_level,
                    dropped_record_ids=(),
                    includes_aggregate=with_aggregate,
                )
        accepted: list[JsonObject] = []
        accepted_ids: list[str] = []
        dropped_ids: list[str] = []
        for record_id, record in shown:
            trial = self._block_value(
                schema=MEMORY_BLOCK_SCHEMA_V2,
                records=[*accepted, _minimal_record(record, scores[record_id])],
                detail_level="minimal",
            )
            if len(accepted) < config.max_memory_records and self._fits(trial, config) is not None:
                accepted.append(_minimal_record(record, scores[record_id]))
                accepted_ids.append(record_id)
            else:
                dropped_ids.append(record_id)
        block = self._block_value(
            schema=MEMORY_BLOCK_SCHEMA_V2, records=accepted, detail_level="minimal"
        )
        fitted = self._fits(block, config)
        if fitted is None:
            raise ValueError("memory bounds are too small for the canonical empty block")
        canonical, byte_count = fitted
        return RenderedMemoryBlock(
            canonical,
            tuple(accepted_ids),
            byte_count,
            byte_count,
            projection=PromptProjection.COMPACT_ADAPTIVE_V2.value,
            detail_level="minimal",
            dropped_record_ids=tuple(dropped_ids),
            includes_aggregate=False,
        )


@dataclass(frozen=True, slots=True)
class ContextualExperienceRuntime:
    arm_id: str
    snapshot: ExperienceSnapshot | None
    config: ContextualMemoryConfig
    frozen_target_task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.arm_id
            or tuple(sorted(set(self.frozen_target_task_ids))) != self.frozen_target_task_ids
        ):
            raise ValueError("runtime requires an arm and sorted unique target task IDs")
        if self.snapshot is not None and set(self.snapshot.source_task_ids) & set(
            self.frozen_target_task_ids
        ):
            raise ValueError("memory source and frozen target tasks overlap")

    @property
    def runtime_hash(self) -> str:
        return sha256_json(self.to_value())

    def to_value(self) -> JsonObject:
        return {
            "schema_version": RUNTIME_SCHEMA,
            "arm_id": self.arm_id,
            "snapshot_hash": self.snapshot.snapshot_hash if self.snapshot else None,
            "config": self.config.to_value(),
            "frozen_target_task_ids": list(self.frozen_target_task_ids),
        }
