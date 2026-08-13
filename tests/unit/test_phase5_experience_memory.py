from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from world_model_search.domain.types import SplitLabel
from world_model_search.dsl.ast import At, Count, Eq, If, IntConst, Not
from world_model_search.memory.experience import (
    BundleValidationPair,
    ExperienceLessonProposal,
    ExperienceLineageStep,
    ExperienceMemorySnapshot,
    LessonPromotionPolicy,
    LessonValidationPair,
    MatchedUnsuccessfulLineage,
    PromotedExperienceLesson,
    SuccessfulLineageEvidence,
    ValidationStage,
    evaluate_bundle_validation,
    evaluate_lesson_promotion,
    retrieve_experience_for_cell,
)
from world_model_search.model.phase5_experience_prompts import (
    assert_experience_prompt_isolation,
    inject_experience_memory,
    lesson_induction_json_schema,
    parse_lesson_induction_response,
    render_lesson_induction_prompt,
)
from world_model_search.search.archive import (
    ArchiveCoordinate,
    ArchiveLayer,
    RepresentationFamily,
)
from world_model_search.serialization import canonical_json, sha256_json


def _coordinate(family: RepresentationFamily, *, exact: bool) -> ArchiveCoordinate:
    return ArchiveCoordinate(
        size_bin=("b0" if family is RepresentationFamily.POSITION_SPECIFIC else "b2"),
        representation_family=family,
        error_signature_cluster="p2-00" if exact else "p2-10",
        layer=ArchiveLayer.EXACT if exact else ArchiveLayer.PARTIAL,
    )


def _lineage(task: str, generator_family: str) -> SuccessfulLineageEvidence:
    family = RepresentationFamily.CONDITIONAL
    root = ExperienceLineageStep(
        candidate_id=f"{task}-root",
        ordered_parent_ids=(),
        ast=If(Eq(Count((-1, 0, 1)), IntConst(1)), At(-1), At(0)),
        local_errors=2,
        local_cases=8,
        exact=False,
        ast_bits=36,
        residual_bits=6,
        archive_coordinate=_coordinate(family, exact=False),
    )
    terminal = ExperienceLineageStep(
        candidate_id=f"{task}-exact",
        ordered_parent_ids=(root.candidate_id,),
        ast=If(Eq(Count((-1, 0, 1)), IntConst(1)), At(-1), At(1)),
        local_errors=0,
        local_cases=8,
        exact=True,
        ast_bits=36,
        residual_bits=0,
        archive_coordinate=_coordinate(family, exact=True),
    )
    failed = ExperienceLineageStep(
        candidate_id=f"{task}-failed",
        ordered_parent_ids=(root.candidate_id,),
        ast=If(Eq(Count((-1, 0, 1)), IntConst(1)), At(-1), Not(At(0))),
        local_errors=1,
        local_cases=8,
        exact=False,
        ast_bits=40,
        residual_bits=4,
        archive_coordinate=_coordinate(family, exact=False),
    )
    return SuccessfulLineageEvidence(
        task_id=task,
        generator_family_id=generator_family,
        role=SplitLabel.TRAINING,
        source_task_split=SplitLabel.DEVELOPMENT,
        search_seed=7,
        consequential_request_index=3,
        selected_parent_candidate_id=root.candidate_id,
        steps=(root, terminal),
        matched_unsuccessful_lineages=(MatchedUnsuccessfulLineage((root, failed)),),
        run_hash=sha256_json({"run": task}),
        artifact_hash=sha256_json({"artifact": task}),
    )


def _proposal(*evidence: SuccessfulLineageEvidence) -> ExperienceLessonProposal:
    return ExperienceLessonProposal(
        lesson_text=(
            "When revising a conditional archive branch, preserve the condition and test small "
            "typed changes to its branches before replacing the whole expression."
        ),
        archive_representation_family=RepresentationFamily.CONDITIONAL,
        source_evidence_ids=tuple(sorted(item.evidence_id for item in evidence)),
    )


def _validation(
    proposal: ExperienceLessonProposal,
    task: str,
    generator_family: str,
    *,
    gain: int = 100_000,
) -> LessonValidationPair:
    return LessonValidationPair(
        validation_stage=ValidationStage.INDIVIDUAL_LESSON,
        lesson_proposal_id=proposal.proposal_id,
        task_id=task,
        generator_family_id=generator_family,
        seed=7,
        archive_representation_family=RepresentationFamily.CONDITIONAL,
        baseline_exact=False,
        treatment_exact=True,
        baseline_normalized_exact_auc_ppm=100_000,
        treatment_normalized_exact_auc_ppm=100_000 + gain,
        treatment_memory_applied=True,
    )


def test_successful_lineage_prompt_is_experience_based_and_hides_generator_family() -> None:
    first = _lineage("task-a", "hidden-source-a")
    second = _lineage("task-b", "hidden-source-b")
    prompt = render_lesson_induction_prompt(
        evidence=(first, second),
        requested_lessons=1,
        representation_family=RepresentationFamily.CONDITIONAL,
    )
    value = json.loads(prompt)
    assert len(value["matched_lineage_contrasts"]) == 2
    assert "ordered_successful_lineage" in value["matched_lineage_contrasts"][0]
    assert "matched_unsuccessful_lineages" in value["matched_lineage_contrasts"][0]
    assert "generator_family_id" not in prompt
    assert "reference_ast" not in prompt
    assert (
        lesson_induction_json_schema(
            requested_lessons=1,
            representation_family=RepresentationFamily.CONDITIONAL,
        )["type"]
        == "object"
    )


def test_lesson_induction_requires_cited_lineages_from_one_archive_family() -> None:
    first = _lineage("task-a", "hidden-source-a")
    second = _lineage("task-b", "hidden-source-b")
    catalog = {item.evidence_id: item for item in (first, second)}
    response = canonical_json(
        {
            "lesson_batch_version": 1,
            "lessons": [
                {
                    "lesson_text": _proposal(first, second).lesson_text,
                    "archive_representation_family": "conditional",
                    "source_evidence_ids": sorted(catalog),
                }
            ],
        }
    )
    parsed = parse_lesson_induction_response(
        response,
        evidence_catalog=catalog,
        requested_lessons=1,
        representation_family=RepresentationFamily.CONDITIONAL,
    )
    assert parsed[0].source_evidence_ids == tuple(sorted(catalog))

    crossed = json.loads(response)
    crossed["lessons"][0]["archive_representation_family"] = "count-based"
    with pytest.raises(ValueError, match="changed the request's representation family"):
        parse_lesson_induction_response(
            canonical_json(crossed),
            evidence_catalog=catalog,
            requested_lessons=1,
            representation_family=RepresentationFamily.CONDITIONAL,
        )


def test_promotion_accepts_declared_single_source_family_and_four_prospective_tasks() -> None:
    first = _lineage("task-a", "single-source-family")
    second = _lineage("task-b", "single-source-family")
    catalog = {item.evidence_id: item for item in (first, second)}
    proposal = _proposal(first, second)
    positive = tuple(
        _validation(proposal, f"validation-{index}", "single-validation-family")
        for index in range(4)
    )
    passed = evaluate_lesson_promotion(
        proposal=proposal,
        evidence_catalog=catalog,
        validation_pairs=positive,
    )
    assert passed.status == "promoted"
    assert passed.promoted_lesson is not None

    failed = evaluate_lesson_promotion(
        proposal=proposal,
        evidence_catalog=catalog,
        validation_pairs=tuple(
            _validation(
                proposal,
                f"validation-{index}",
                "single-validation-family",
                gain=-50_000,
            )
            for index in range(4)
        ),
    )
    assert failed.status == "rejected"
    assert "nonpositive-gain-in-a-validation-family" in failed.reasons


def test_individual_screen_is_inconclusive_until_exposure_threshold() -> None:
    first = _lineage("task-a", "single-source-family")
    second = _lineage("task-b", "single-source-family")
    proposal = _proposal(first, second)
    pairs = tuple(
        _validation(proposal, f"validation-{index}", "single-validation-family")
        for index in range(2)
    )
    decision = evaluate_lesson_promotion(
        proposal=proposal,
        evidence_catalog={item.evidence_id: item for item in (first, second)},
        validation_pairs=pairs,
    )
    assert decision.status == "inconclusive"
    assert "insufficient-validation-tasks-with-memory-exposure" in decision.reasons

    relaxed = evaluate_lesson_promotion(
        proposal=proposal,
        evidence_catalog={item.evidence_id: item for item in (first, second)},
        validation_pairs=pairs,
        policy=LessonPromotionPolicy(minimum_validation_tasks_with_exposure=2),
    )
    assert relaxed.status == "promoted"


def test_bundle_confirmation_requires_fresh_tasks_and_per_lesson_exposure() -> None:
    first = _lineage("task-a", "single-source-family")
    second = _lineage("task-b", "single-source-family")
    proposal = _proposal(first, second)
    screening = tuple(
        _validation(proposal, f"screen-{index}", "single-validation-family") for index in range(4)
    )
    promoted = evaluate_lesson_promotion(
        proposal=proposal,
        evidence_catalog={item.evidence_id: item for item in (first, second)},
        validation_pairs=screening,
    ).promoted_lesson
    assert promoted is not None

    def pair(task: str) -> BundleValidationPair:
        return BundleValidationPair(
            task_id=task,
            generator_family_id="single-validation-family",
            seed=11,
            baseline_exact=False,
            treatment_exact=True,
            baseline_normalized_exact_auc_ppm=100_000,
            treatment_normalized_exact_auc_ppm=200_000,
            applied_lesson_ids=(promoted.lesson_id,),
        )

    passed = evaluate_bundle_validation(
        protocol_hash=sha256_json({"protocol": "bundle"}),
        promoted_lessons=(promoted,),
        individual_screen_task_ids=frozenset(pair.task_id for pair in screening),
        validation_pairs=tuple(pair(f"bundle-{index}") for index in range(4)),
    )
    assert passed.status == "promoted"
    assert passed.snapshot is not None
    assert len(passed.snapshot.bundle_validation_pair_hashes) == 4

    reused = evaluate_bundle_validation(
        protocol_hash=sha256_json({"protocol": "bundle"}),
        promoted_lessons=(promoted,),
        individual_screen_task_ids=frozenset(pair.task_id for pair in screening),
        validation_pairs=(pair("screen-0"), *(pair(f"bundle-{index}") for index in range(3))),
    )
    assert reused.status == "rejected"
    assert "bundle-validation-reuses-individual-screen-task" in reused.reasons


def test_cell_conditioned_retrieval_passes_only_matching_family_and_is_bounded() -> None:
    first = _lineage("task-a", "hidden-source-a")
    second = _lineage("task-b", "hidden-source-b")
    decision = evaluate_lesson_promotion(
        proposal=(proposal := _proposal(first, second)),
        evidence_catalog={item.evidence_id: item for item in (first, second)},
        validation_pairs=tuple(
            _validation(proposal, f"validation-{index}", "hidden-validation") for index in range(4)
        ),
    )
    assert decision.promoted_lesson is not None
    unrelated_proposal = ExperienceLessonProposal(
        lesson_text="For count-based branches, try small threshold changes before changing form.",
        archive_representation_family=RepresentationFamily.COUNT_BASED,
        source_evidence_ids=decision.promoted_lesson.proposal.source_evidence_ids,
    )
    unrelated = PromotedExperienceLesson(
        unrelated_proposal,
        decision.promoted_lesson.validation_pair_hashes,
        50_000,
    )
    snapshot = ExperienceMemorySnapshot(
        protocol_hash=sha256_json({"protocol": 2}),
        lessons=tuple(
            sorted((decision.promoted_lesson, unrelated), key=lambda item: item.lesson_id)
        ),
    )
    retrieval = retrieve_experience_for_cell(
        snapshot=snapshot,
        selected_cell=_coordinate(RepresentationFamily.CONDITIONAL, exact=True),
        max_items=1,
        max_bytes=4_096,
        max_tokens=4_096,
    )
    block = json.loads(retrieval.rendered_memory)
    assert block["selected_archive_representation_family"] == "conditional"
    assert len(block["items"]) == 1
    assert block["items"][0]["lesson_id"] == decision.promoted_lesson.lesson_id
    assert unrelated.lesson_id not in retrieval.selected_lesson_ids
    assert (unrelated.lesson_id, "archive-representation-family-mismatch") in retrieval.exclusions
    assert retrieval.rendered_bytes <= 4_096


def test_memory_off_on_prompts_differ_only_in_cell_conditioned_block() -> None:
    first = _lineage("task-a", "hidden-source-a")
    second = _lineage("task-b", "hidden-source-b")
    decision = evaluate_lesson_promotion(
        proposal=(proposal := _proposal(first, second)),
        evidence_catalog={item.evidence_id: item for item in (first, second)},
        validation_pairs=tuple(
            _validation(proposal, f"validation-{index}", "hidden-validation") for index in range(4)
        ),
    )
    assert decision.promoted_lesson is not None
    protocol_hash = sha256_json({"protocol": 2})
    on_snapshot = ExperienceMemorySnapshot(protocol_hash, (decision.promoted_lesson,))
    off_snapshot = ExperienceMemorySnapshot(protocol_hash, ())
    cell = _coordinate(RepresentationFamily.CONDITIONAL, exact=True)
    bounds = {"max_items": 4, "max_bytes": 4_096, "max_tokens": 4_096}
    on = retrieve_experience_for_cell(snapshot=on_snapshot, selected_cell=cell, **bounds)
    off = retrieve_experience_for_cell(snapshot=off_snapshot, selected_cell=cell, **bounds)
    base = canonical_json({"public_task": {"task_id": "opaque"}, "selected_parent": {}})
    on_prompt = inject_experience_memory(base_prompt=base, retrieval=on)
    off_prompt = inject_experience_memory(base_prompt=base, retrieval=off)
    assert_experience_prompt_isolation(off_prompt, on_prompt)


def test_nonexact_or_disconnected_lineage_cannot_become_evidence() -> None:
    with pytest.raises(ValueError, match="exact terminal archive elite"):
        SuccessfulLineageEvidence(
            task_id="task-a",
            generator_family_id="hidden-a",
            role=SplitLabel.TRAINING,
            source_task_split=SplitLabel.DEVELOPMENT,
            search_seed=7,
            consequential_request_index=0,
            selected_parent_candidate_id="root",
            steps=(
                ExperienceLineageStep(
                    candidate_id="root",
                    ordered_parent_ids=(),
                    ast=At(0),
                    local_errors=1,
                    local_cases=8,
                    exact=False,
                    ast_bits=5,
                    residual_bits=4,
                    archive_coordinate=_coordinate(
                        RepresentationFamily.POSITION_SPECIFIC, exact=False
                    ),
                ),
            ),
            matched_unsuccessful_lineages=(),
            run_hash="0" * 64,
            artifact_hash="1" * 64,
        )


def test_experience_v2_paid_protocol_is_fail_closed() -> None:
    protocol = yaml.safe_load(
        Path("experiments/phase5-experience-v2.pending.yaml").read_text(encoding="utf-8")
    )
    assert protocol["status"].startswith("design-frozen-pending")
    source = protocol["experience_source"]
    assert source["canonical_reference_ast_available_to_lesson_induction"] is False
    assert source["designation"] == "retrospective-single-source-family-training"
    assert source["source_condition"] == "uniform-diverse-archive-v1"
    assert source["minimum_training_generator_families"] == 1
    assert protocol["lesson_induction"]["prepared_request_count"] == 3
    assert protocol["lesson_induction"]["provider_requests_executed"] == 0
    assert (
        protocol["promotion_validation"]["stage_1_individual_lesson_screen"]["treatment_arms"]
        == "exactly-one-lesson-per-arm"
    )
    assert protocol["promotion_validation"]["stage_2_promoted_bundle_confirmation"][
        "tasks_fresh_from_stage_1"
    ]
    assert protocol["evidence_scopes"]["retrieval_key"] == (
        "public-archive-coordinate-representation-family-v1"
    )
    assert not any(protocol["authorization"].values())
    assert protocol["fail_closed"] is True
