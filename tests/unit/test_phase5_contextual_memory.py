from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from world_model_search.domain.types import (
    CandidateSummary,
    OracleFeedback,
    OracleResponseMode,
    OracleResult,
    ProposalRole,
    PublicDemonstration,
    PublicTask,
    PublicWorldSpec,
    SplitLabel,
)
from world_model_search.dsl.ast import And, At, BitExpr, Not, Or, Xor
from world_model_search.errors import ConfigurationError
from world_model_search.evaluation.phase5_contextual import (
    FROZEN_ARMS,
    ContextualExperimentRegistry,
    _uptake_counts,
    enforce_arm_distinctness,
    preflight_arm_distinctness,
)
from world_model_search.memory.contextual import (
    PENDING_DOWNSTREAM,
    ContextFeatureExtractor,
    ContextFeatures,
    ContextMode,
    DownstreamOutcome,
    DownstreamOutcomeAnnotator,
    EditClass,
    ExperienceMetadata,
    ExperienceProvenance,
    ExperienceRecord,
    ImmediateOutcome,
    PromptProjection,
    RetrievalMode,
    SelectionPolicy,
    extract_ast_delta,
)
from world_model_search.memory.contextual_retrieval import (
    ContextualMemoryConfig,
    ExperienceRetriever,
    ExperienceSnapshot,
    ExperienceStore,
    MemoryBlockRenderer,
    RandomizedExposureConfig,
    RenderedMemoryBlock,
    _aggregate_action_outcomes,
)
from world_model_search.model.contextual_prompts import (
    assert_contextual_prompt_isolation,
    inject_contextual_memory,
)
from world_model_search.model.prompts import ParentScoreFeedback, render_prompt
from world_model_search.search.archive import descriptor
from world_model_search.serialization import canonical_json


def _task(task_id: str = "task-current") -> PublicTask:
    return PublicTask(
        task_id=task_id,
        public_world_spec=PublicWorldSpec(
            "test-world-v1",
            "binary ring",
            "binary ring",
            "typed AST",
        ),
        split=SplitLabel.TRAINING,
        demonstrations=(PublicDemonstration("0101", "1010"),),
        active_queries_enabled=False,
        query_budget=0,
    )


def _result(errors: int) -> OracleResult:
    return OracleResult(
        type_valid=True,
        total=True,
        local_errors=errors,
        local_cases=8,
        rollout_pass=errors == 0,
        exact=errors == 0,
        ast_bits=12,
        residual_bits=errors * 2,
        runtime_ns=1,
        response=OracleFeedback(OracleResponseMode.SCORE_ONLY),
    )


def _context() -> ContextFeatures:
    task = _task()
    ast = At(0)
    result = _result(2)
    return ContextFeatureExtractor().extract(
        parent=ast,
        result=result,
        task=task,
        coordinate=descriptor(ast, result, task),
        search_step=3,
        lineage_depth=2,
        plateau_length=1,
        recent_edit_classes=(EditClass.ADD_NEGATION.value,),
    )


def _record(child_id: str = "child") -> ExperienceRecord:
    context = _context()
    return ExperienceRecord(
        ExperienceProvenance(
            "elementary-ca-v1",
            "task-source",
            SplitLabel.TRAINING.value,
            "run-a",
            7,
            3,
            0,
            "parent",
            child_id,
            "lineage",
            8,
        ),
        context,
        extract_ast_delta(At(0), Not(At(0))),
        ImmediateOutcome(
            7, 1, 1, -1, False, False, True, False, "inserted", {}, False, False, None
        ),
        PENDING_DOWNSTREAM,
        ExperienceMetadata(True, False, "prospective"),
    )


def _annotated_record(
    task_id: str,
    *,
    score_delta: int,
    archive_outcome: str = "inserted",
    sealed: bool = False,
) -> ExperienceRecord:
    record = _record(f"child-{task_id}")
    return replace(
        record,
        provenance=replace(
            record.provenance,
            task_id=task_id,
            child_candidate_id=f"child-{task_id}",
        ),
        immediate_outcome=replace(
            record.immediate_outcome,
            score_delta=score_delta,
            error_delta=-score_delta,
            archive_outcome=archive_outcome,
            archive_inserted=archive_outcome in {"inserted", "replaced", "reserved"},
        ),
        downstream_outcome=DownstreamOutcome(
            "complete",
            False,
            None,
            7 + max(score_delta, 0),
            max(score_delta, 0),
            0,
            8,
            max(score_delta, 0),
            False,
        ),
        memory_metadata=ExperienceMetadata(not sealed, sealed, "prospective"),
    )


def _snapshot(*records: ExperienceRecord) -> ExperienceSnapshot:
    return ExperienceSnapshot(
        tuple(sorted(record.provenance.task_id for record in records)),
        records,
        tuple(f"hash-{index}" for index, _record_value in enumerate(records)),
    )


def test_deterministic_ast_diff_and_multi_label_classification() -> None:
    first = extract_ast_delta(At(0), And(At(0), Not(At(1))))
    second = extract_ast_delta(At(0), And(At(0), Not(At(1))))
    assert first == second
    assert first.changes
    assert EditClass.ADD_POSITION_REFERENCE in first.edit_classes
    assert EditClass.ADD_NEGATION in first.edit_classes
    assert EditClass.COMPOSE_BOOLEAN_SUBTREES in first.edit_classes
    assert EditClass.EXPAND_SUBTREE in first.edit_classes
    assert first.size_delta > 0


def test_context_features_are_deterministic_and_public_score_only() -> None:
    first = _context()
    second = _context()
    assert first == second
    assert first.parent_score == 6
    assert first.parent_error_count == 2
    assert first.parent_error_pattern is None
    assert first.public_probe_behavior
    assert first.constructor_histogram == {"At": 1}


def test_record_round_trip_preserves_identity_and_rejects_sealed_reuse() -> None:
    record = _record()
    assert ExperienceRecord.from_value(record.to_value()) == record
    sealed = replace(
        record,
        memory_metadata=ExperienceMetadata(False, True, "prospective"),
    )
    assert sealed.memory_metadata.training_eligible is False
    assert sealed.memory_metadata.sealed_test is True


def test_downstream_annotation_is_descriptive_and_deterministic() -> None:
    record = _record("child")
    annotated = DownstreamOutcomeAnnotator().annotate(
        (record,),
        parent_ids={"grandchild": ("child",), "exact": ("grandchild",)},
        score_by_candidate={"child": 7, "grandchild": 7, "exact": 8},
        exact_by_candidate={"child": False, "grandchild": False, "exact": True},
        evaluation_by_candidate={"child": 8, "grandchild": 10, "exact": 12},
    )[0]
    assert annotated.downstream_outcome.annotation_status == "complete"
    assert annotated.downstream_outcome.eventually_had_exact_descendant
    assert annotated.downstream_outcome.evaluations_until_exact_descendant == 4
    assert annotated.downstream_outcome.best_descendant_score == 8
    assert annotated.downstream_outcome.max_descendant_improvement == 1
    assert annotated.downstream_outcome.lineage_survival_length == 2


def test_downstream_annotation_excludes_lineage_events_before_transition() -> None:
    record = replace(
        _record("child"),
        provenance=replace(_record("child").provenance, sequence_index=12),
    )
    annotated = DownstreamOutcomeAnnotator().annotate(
        (record,),
        parent_ids={"old-descendant": ("child",), "later": ("old-descendant",)},
        score_by_candidate={"child": 7, "old-descendant": 7, "later": 8},
        exact_by_candidate={"child": False, "old-descendant": False, "later": True},
        evaluation_by_candidate={"child": 8, "old-descendant": 10, "later": 13},
    )[0]
    assert annotated.downstream_outcome.eventually_had_exact_descendant is False
    assert annotated.downstream_outcome.lineage_survival_length == 0


def test_retrieval_ranking_and_contrast_selection_are_deterministic() -> None:
    positive = _annotated_record("source-positive", score_delta=1)
    negative = _annotated_record("source-negative", score_delta=-1, archive_outcome="rejected")
    store = ExperienceStore(_snapshot(positive, negative))
    config = ContextualMemoryConfig(
        retrieval_mode=RetrievalMode.CONTRASTIVE,
        minimum_similarity_ppm=0,
        max_memory_bytes=20_000,
        max_memory_tokens_conservative=20_000,
    )
    arguments = {
        "store": store,
        "current_context": _context(),
        "current_task_id": "target",
        "forbidden_task_ids": frozenset({"target"}),
        "search_seed": 17,
        "retrieval_index": 2,
        "config": config,
    }
    decision = ExperienceRetriever().retrieve(**arguments)
    assert {item.outcome_class for item in decision.all_eligible_scores} == {
        "positive",
        "negative",
    }
    assert decision.selected_record_ids == (positive.record_id, negative.record_id)
    assert all(item.components for item in decision.all_eligible_scores)
    assert decision == ExperienceRetriever().retrieve(**arguments)


def test_family_only_ablation_disables_rich_similarity_components() -> None:
    record = _annotated_record("source", score_delta=1)
    decision = ExperienceRetriever().retrieve(
        store=ExperienceStore(_snapshot(record)),
        current_context=_context(),
        current_task_id="target",
        forbidden_task_ids=frozenset(),
        search_seed=1,
        retrieval_index=0,
        config=ContextualMemoryConfig(
            context_mode=ContextMode.FAMILY_ONLY,
            retrieval_mode=RetrievalMode.POSITIVE_ONLY,
            minimum_similarity_ppm=0,
        ),
    )
    components = decision.all_eligible_scores[0].components
    assert components["representation_family_match"]["effective_weight"] > 0
    assert components["structural_motif_similarity"]["effective_weight"] == 0


def test_store_excludes_forbidden_tasks_and_snapshot_rejects_sealed_records() -> None:
    allowed = _annotated_record("allowed", score_delta=1)
    forbidden = _annotated_record("forbidden", score_delta=1)
    store = ExperienceStore(_snapshot(allowed, forbidden))
    assert store.eligible(
        current_task_id="target",
        forbidden_task_ids=frozenset({"forbidden"}),
    ) == (allowed,)
    sealed = _annotated_record("sealed", score_delta=1, sealed=True)
    try:
        _snapshot(sealed)
    except ValueError as exc:
        assert "ineligible, sealed" in str(exc)
    else:
        raise AssertionError("sealed test experience entered a reusable snapshot")


def test_memory_renderer_is_canonical_bounded_and_contains_no_provenance() -> None:
    positive = _annotated_record("source-positive", score_delta=1)
    negative = _annotated_record("source-negative", score_delta=-1, archive_outcome="rejected")
    store = ExperienceStore(_snapshot(positive, negative))
    config = ContextualMemoryConfig(
        retrieval_mode=RetrievalMode.CONTRASTIVE,
        minimum_similarity_ppm=0,
        max_memory_bytes=20_000,
        max_memory_tokens_conservative=20_000,
    )
    decision = ExperienceRetriever().retrieve(
        store=store,
        current_context=_context(),
        current_task_id="target",
        forbidden_task_ids=frozenset(),
        search_seed=1,
        retrieval_index=0,
        config=config,
    )
    block = MemoryBlockRenderer().render(decision=decision, store=store)
    assert block.canonical_json_block == canonical_json(block.value)
    assert block.byte_count <= config.max_memory_bytes
    assert "task_id" not in block.canonical_json_block
    assert "run_id" not in block.canonical_json_block
    assert len(block.shown_record_ids) == 2


def test_prompt_budget_drops_whole_records_and_logs_actual_exposure() -> None:
    record = _annotated_record("source", score_delta=1)
    store = ExperienceStore(_snapshot(record))
    config = ContextualMemoryConfig(
        retrieval_mode=RetrievalMode.POSITIVE_ONLY,
        minimum_similarity_ppm=0,
        max_memory_bytes=500,
        max_memory_tokens_conservative=500,
    )
    decision = ExperienceRetriever().retrieve(
        store=store,
        current_context=_context(),
        current_task_id="target",
        forbidden_task_ids=frozenset(),
        search_seed=1,
        retrieval_index=0,
        config=config,
    )
    block = MemoryBlockRenderer().render(decision=decision, store=store)
    assert block.value["cross_task_experience"] == []
    assert block.shown_record_ids == ()
    audit = decision.to_value(actually_shown_record_ids=block.shown_record_ids)
    assert audit["exposure_decisions"][0]["shown"] is False


def test_no_relevant_memory_returns_canonical_empty_block() -> None:
    record = _annotated_record("source", score_delta=1)
    store = ExperienceStore(_snapshot(record))
    decision = ExperienceRetriever().retrieve(
        store=store,
        current_context=replace(_context(), representation_family="parity"),
        current_task_id="target",
        forbidden_task_ids=frozenset(),
        search_seed=1,
        retrieval_index=0,
        config=ContextualMemoryConfig(
            context_mode=ContextMode.FAMILY_ONLY,
            retrieval_mode=RetrievalMode.POSITIVE_ONLY,
            minimum_similarity_ppm=1,
        ),
    )
    block = MemoryBlockRenderer().render(decision=decision, store=store)
    assert decision.selected_record_ids == ()
    assert block.value["cross_task_experience"] == []


def test_randomized_exposure_is_reproducible_and_logged() -> None:
    record = _annotated_record("source", score_delta=1)
    store = ExperienceStore(_snapshot(record))
    config = ContextualMemoryConfig(
        retrieval_mode=RetrievalMode.POSITIVE_ONLY,
        minimum_similarity_ppm=0,
        exposure=RandomizedExposureConfig(True, 1, 2, 99),
    )
    arguments = {
        "store": store,
        "current_context": _context(),
        "current_task_id": "target",
        "forbidden_task_ids": frozenset(),
        "search_seed": 3,
        "retrieval_index": 4,
        "config": config,
    }
    first = ExperienceRetriever().retrieve(**arguments)
    second = ExperienceRetriever().retrieve(**arguments)
    assert first.exposure_decisions == second.exposure_decisions
    exposure = first.exposure_decisions[0]
    assert exposure.inclusion_probability_numerator == 1
    assert exposure.inclusion_probability_denominator == 2
    assert exposure.randomization_seed == 99


def test_memoryless_and_treatment_prompts_differ_only_in_memory_block() -> None:
    positive = _annotated_record("source", score_delta=1)
    treatment_store = ExperienceStore(_snapshot(positive))
    config = ContextualMemoryConfig(
        retrieval_mode=RetrievalMode.POSITIVE_ONLY,
        minimum_similarity_ppm=0,
        max_memory_bytes=20_000,
        max_memory_tokens_conservative=20_000,
    )

    def block(store: ExperienceStore, mode: RetrievalMode) -> RenderedMemoryBlock:
        decision = ExperienceRetriever().retrieve(
            store=store,
            current_context=_context(),
            current_task_id="target",
            forbidden_task_ids=frozenset(),
            search_seed=1,
            retrieval_index=0,
            config=replace(config, retrieval_mode=mode),
        )
        return MemoryBlockRenderer().render(decision=decision, store=store)

    candidate_id = "a" * 64
    _template, _version, base = render_prompt(
        task=_task("target"),
        role=ProposalRole.EXPLOIT,
        requested_batch_size=1,
        parent=CandidateSummary(candidate_id, At(0)),
        feedback=ParentScoreFeedback(candidate_id, True, True, 2, 8, False, 12, 4, 16),
    )
    control = inject_contextual_memory(
        base_prompt=base,
        memory_block=block(ExperienceStore(None), RetrievalMode.DISABLED),
    )
    treatment = inject_contextual_memory(
        base_prompt=base,
        memory_block=block(treatment_store, RetrievalMode.POSITIVE_ONLY),
    )
    assert_contextual_prompt_isolation(control, treatment)


def _deep_ast(depth: int) -> BitExpr:
    positions = (-1, 0, 1)
    expr: BitExpr = At(0)
    for index in range(depth):
        expr = And(
            Xor(At(positions[index % 3]), expr),
            Or(At(positions[(index + 1) % 3]), Not(At(positions[(index + 2) % 3]))),
        )
    return expr


def _bulky_record(
    task_id: str,
    *,
    score_delta: int,
    archive_outcome: str = "inserted",
) -> ExperienceRecord:
    base = _annotated_record(task_id, score_delta=score_delta, archive_outcome=archive_outcome)
    return replace(base, action=extract_ast_delta(_deep_ast(20), Xor(_deep_ast(20), At(1))))


def _compact_config() -> ContextualMemoryConfig:
    return ContextualMemoryConfig(
        retrieval_mode=RetrievalMode.CONTRASTIVE,
        minimum_similarity_ppm=0,
        prompt_projection=PromptProjection.COMPACT_ADAPTIVE_V2,
        selection_policy=SelectionPolicy.TASK_BALANCED_CONTRAST_V2,
        include_aggregate_summary=True,
    )


def _rendered(store: ExperienceStore, config: ContextualMemoryConfig) -> RenderedMemoryBlock:
    decision = ExperienceRetriever().retrieve(
        store=store,
        current_context=_context(),
        current_task_id="target",
        forbidden_task_ids=frozenset(),
        search_seed=1,
        retrieval_index=0,
        config=config,
    )
    return MemoryBlockRenderer().render(decision=decision, store=store)


def test_default_config_preserves_v1_projection_and_selection() -> None:
    config = ContextualMemoryConfig()
    assert config.prompt_projection is PromptProjection.FULL_GREEDY_V1
    assert config.selection_policy is SelectionPolicy.SIMILARITY_RECORD_ID_V1
    assert config.include_aggregate_summary is False
    with pytest.raises(ValueError):
        ContextualMemoryConfig(include_aggregate_summary=True)
    with pytest.raises(ValueError):
        ContextualMemoryConfig(
            include_diverse_third=True,
            selection_policy=SelectionPolicy.TASK_BALANCED_CONTRAST_V2,
        )


def test_compact_adaptive_projection_preserves_contrast_within_budget() -> None:
    positive = _bulky_record("source-positive", score_delta=1)
    negative = _bulky_record("source-negative", score_delta=-1, archive_outcome="rejected")
    store = ExperienceStore(_snapshot(positive, negative))
    legacy = ContextualMemoryConfig(
        retrieval_mode=RetrievalMode.CONTRASTIVE,
        minimum_similarity_ppm=0,
    )
    legacy_block = _rendered(store, legacy)
    compact_block = _rendered(store, _compact_config())
    assert len(legacy_block.shown_record_ids) < 2
    assert legacy_block.dropped_record_ids
    assert len(compact_block.shown_record_ids) == 2
    assert compact_block.byte_count <= legacy.max_memory_bytes
    assert compact_block.projection == PromptProjection.COMPACT_ADAPTIVE_V2.value
    assert compact_block.value["schema_version"] == "cross-task-experience-block-v2"
    experiences = compact_block.value["cross_task_experience"]
    assert isinstance(experiences, list)
    kinds = {item["record_kind"] for item in experiences if isinstance(item, dict)}
    assert kinds == {"positive", "negative"}
    assert "task_id" not in compact_block.canonical_json_block
    assert "run_id" not in compact_block.canonical_json_block


def test_aggregate_summary_reports_base_rates_and_caps_signatures() -> None:
    positive = _bulky_record("source-positive", score_delta=1)
    negative = _bulky_record("source-negative", score_delta=-1, archive_outcome="rejected")
    neutral = _annotated_record("source-neutral", score_delta=0)
    entries = _aggregate_action_outcomes((positive, negative, neutral), maximum_signatures=6)
    assert len(entries) == 2
    first = entries[0]
    assert isinstance(first, dict)
    assert first["observed"] == 2
    assert first["positive"] == 1
    assert first["negative"] == 1
    assert first["distinct_source_tasks"] == 2
    assert len(_aggregate_action_outcomes((positive, negative, neutral), maximum_signatures=1)) == 1
    block = _rendered(ExperienceStore(_snapshot(positive, negative)), _compact_config())
    assert block.includes_aggregate
    aggregate = block.value["aggregate_action_outcomes"]
    assert isinstance(aggregate, list) and aggregate


def test_task_balanced_selection_spreads_source_tasks() -> None:
    def positive(task_id: str, child: str) -> ExperienceRecord:
        record = _annotated_record(task_id, score_delta=1)
        return replace(
            record,
            provenance=replace(record.provenance, child_candidate_id=child),
        )

    crowded = tuple(positive("task-a", f"child-{index}") for index in range(3))
    lone = positive("task-b", "child-lone")
    snapshot = ExperienceSnapshot(
        ("task-a", "task-b"),
        (*crowded, lone),
        ("h0", "h1", "h2", "h3"),
    )
    store = ExperienceStore(snapshot)

    def selected(config: ContextualMemoryConfig) -> tuple[str, ...]:
        return (
            ExperienceRetriever()
            .retrieve(
                store=store,
                current_context=_context(),
                current_task_id="target",
                forbidden_task_ids=frozenset(),
                search_seed=1,
                retrieval_index=0,
                config=config,
            )
            .selected_record_ids
        )

    v1 = selected(
        ContextualMemoryConfig(retrieval_mode=RetrievalMode.POSITIVE_ONLY, minimum_similarity_ppm=0)
    )
    assert len(v1) == 1
    v2 = selected(
        ContextualMemoryConfig(
            retrieval_mode=RetrievalMode.POSITIVE_ONLY,
            minimum_similarity_ppm=0,
            selection_policy=SelectionPolicy.TASK_BALANCED_CONTRAST_V2,
        )
    )
    assert len(v2) == 3
    assert lone.record_id in v2
    tasks = {store.get(record_id).provenance.task_id for record_id in v2}
    assert tasks == {"task-a", "task-b"}


def test_isolation_accepts_v2_blocks_and_rejects_schema_mismatch() -> None:
    positive = _bulky_record("source-positive", score_delta=1)
    negative = _bulky_record("source-negative", score_delta=-1, archive_outcome="rejected")
    store = ExperienceStore(_snapshot(positive, negative))
    compact = _compact_config()
    candidate_id = "a" * 64
    _template, _version, base = render_prompt(
        task=_task("target"),
        role=ProposalRole.EXPLOIT,
        requested_batch_size=1,
        parent=CandidateSummary(candidate_id, At(0)),
        feedback=ParentScoreFeedback(candidate_id, True, True, 2, 8, False, 12, 4, 16),
    )
    control = inject_contextual_memory(
        base_prompt=base,
        memory_block=_rendered(
            ExperienceStore(None), replace(compact, retrieval_mode=RetrievalMode.DISABLED)
        ),
    )
    treatment = inject_contextual_memory(
        base_prompt=base,
        memory_block=_rendered(store, compact),
    )
    assert_contextual_prompt_isolation(control, treatment)
    v1_control = inject_contextual_memory(
        base_prompt=base,
        memory_block=_rendered(
            ExperienceStore(None),
            ContextualMemoryConfig(retrieval_mode=RetrievalMode.DISABLED),
        ),
    )
    with pytest.raises(ValueError, match="schema differs"):
        assert_contextual_prompt_isolation(v1_control, treatment)


def test_uptake_counts_measure_shown_edit_overlap() -> None:
    counts = _uptake_counts(
        shown_edit_classes_by_logical={
            0: frozenset({"ADD_NEGATION"}),
            1: frozenset({"INTRODUCE_COUNT"}),
            2: frozenset(),
        },
        requests_by_logical={0: (0,), 1: (1, 2)},
        child_edit_classes_by_request={
            0: (frozenset({"ADD_NEGATION", "EXPAND_SUBTREE"}), frozenset({"OTHER"})),
            2: (frozenset({"INTRODUCE_COUNT"}),),
        },
    )
    assert counts == {
        "memory_shown_request_count": 2,
        "children_after_shown_memory": 3,
        "children_matching_shown_edit_classes": 2,
    }


def _preflight_registry(
    memory_config: ContextualMemoryConfig,
    *,
    status: str = "controlled-development",
) -> ContextualExperimentRegistry:
    return ContextualExperimentRegistry(
        experiment_id="unit-preflight",
        status=status,
        base_config=Path("configs/phase4-fake-smoke.yaml"),
        source_run_root=Path("artifacts/none"),
        source_run_suffix="-C",
        expected_source_task_ids=("source-negative", "source-positive"),
        snapshot_path=Path("artifacts/none/memory-snapshot.json"),
        target_split=SplitLabel.DEVELOPMENT,
        target_task_ids=("target",),
        search_seeds=(1,),
        arms=FROZEN_ARMS,
        output_root=Path("artifacts/none/out"),
        runs_root=Path("artifacts/none/runs"),
        memory_config=memory_config,
        raw={},
    )


def test_preflight_blocks_indistinguishable_contrast_and_passes_compact_repair() -> None:
    positive = _bulky_record("source-positive", score_delta=1)
    negative = _bulky_record("source-negative", score_delta=-1, archive_outcome="rejected")
    snapshot = _snapshot(positive, negative)
    legacy = ContextualMemoryConfig(retrieval_mode=RetrievalMode.CONTRASTIVE)
    legacy_registry = _preflight_registry(legacy)
    legacy_report = preflight_arm_distinctness(registry=legacy_registry, snapshot=snapshot)
    assert legacy_report.snapshot_has_negative_records
    with pytest.raises(ConfigurationError):
        enforce_arm_distinctness(registry=legacy_registry, report=legacy_report)
    smoke_registry = _preflight_registry(legacy, status="provider-free-smoke")
    enforce_arm_distinctness(registry=smoke_registry, report=legacy_report)
    compact_registry = _preflight_registry(_compact_config())
    compact_report = preflight_arm_distinctness(registry=compact_registry, snapshot=snapshot)
    assert compact_report.c_differs_from_b_context_count >= 1
    assert compact_report.b_differs_from_empty_context_count >= 1
    assert (
        compact_report.b_matches_d_rich_context_count == compact_report.representative_context_count
    )
    enforce_arm_distinctness(registry=compact_registry, report=compact_report)
