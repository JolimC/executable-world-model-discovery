"""Phase 5 v2 deep prospective lesson and bundle validation."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from world_model_search.config import AppConfig, load_config
from world_model_search.domain.types import SplitLabel
from world_model_search.errors import ConfigurationError
from world_model_search.evaluation.phase5_experience import (
    SOURCE_GENERATOR_FAMILY_ID,
    extract_phase4_condition_c_corpus,
)
from world_model_search.evaluation.phase5_experience_live import (
    BASE_CONFIG_PATH,
    SOURCE_PHASE4_TASKS,
    _lessons,
    _metrics,
    _read,
    _run_arm,
    _selected_lesson_ids,
)
from world_model_search.memory.experience import (
    BundleValidationPair,
    ExperienceLessonProposal,
    ExperienceMemorySnapshot,
    PromotedExperienceLesson,
    evaluate_bundle_validation,
)
from world_model_search.model.phase5_experience_prompts import (
    assert_experience_prompt_isolation,
)
from world_model_search.persistence.artifacts import write_content_artifact
from world_model_search.search.phase4 import Phase4Outcome
from world_model_search.serialization import JsonObject, canonical_json, sha256_json

DEEP_OUTPUT_ROOT = Path("artifacts/phase5-experience-v2/deep-validation-v1")
DEEP_RUNS_ROOT = Path("artifacts/phase5-experience-v2/deep-runs-v1")
SHALLOW_REGISTRY_PATH = Path("artifacts/phase5-experience-v2/prospective-v1/task-registry.json")
DEEP_PROTOCOL: JsonObject = {
    "protocol_version": "phase5-experience-deep-validation-v1",
    "source_family_count": 1,
    "search_distribution": "phase4-condition-c-matched-depth-and-batch-v1",
    "retrieval_key": "selected-public-archive-representation-family-v1",
    "lesson_may_cross_representation_family": False,
    "stage1_minimum_exposure_tasks": 4,
    "stage1_minimum_matching_request_exposures_total": 8,
    "stage1_primary_endpoint": "paired-normalized-exact-solve-auc-v1",
    "stage1_screening_endpoint": "paired-best-so-far-local-accuracy-auc-v1",
    "stage1_promotion_rule": (
        "no-exact-regression-and-positive-mean-exact-auc-gain-or-"
        "nonnegative-mean-exact-auc-gain-with-positive-mean-local-accuracy-auc-gain-v1"
    ),
    "stage2_minimum_any_exposure_tasks": 4,
    "stage2_minimum_exposure_tasks_per_lesson": 2,
    "stage2_requires_positive_mean_exact_auc_gain": True,
    "exact_regression_allowed": False,
    "canary_authorized": False,
}
DEEP_PROTOCOL_HASH = sha256_json(DEEP_PROTOCOL)
STAGE1_SEED = 61_001
STAGE2_SEED = 62_001
PAIR_COUNT_PER_STAGE = 8
LOGICAL_REQUESTS_PER_CHILD = 63
MODEL_REQUEST_ATTEMPT_CAP = 126
PROPOSAL_ITEM_CAP = 249
ORACLE_CALL_CAP = 256
CHILD_CAP_NANO_USD = 150_000_000
MAX_PARALLEL_CHILDREN = 4


@dataclass(frozen=True, slots=True)
class DeepChild:
    task_id: str
    seed: int
    arm: str
    outcome: Phase4Outcome
    result: JsonObject


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{label} must be an integer")
    return value


def _registry(repository_root: Path) -> JsonObject:
    shallow = _read(repository_root / SHALLOW_REGISTRY_PATH)
    excluded = set(SOURCE_PHASE4_TASKS)
    source_tasks = shallow.get("source_training_task_ids")
    if not isinstance(source_tasks, list):
        raise ConfigurationError("shallow source task registry is malformed")
    excluded.update(str(value) for value in source_tasks)
    for stage in ("stage1", "stage2", "canary", "development"):
        block = shallow.get(stage)
        if isinstance(block, dict) and isinstance(block.get("task_ids"), list):
            task_ids = cast(list[object], block["task_ids"])
            excluded.update(str(value) for value in task_ids)
    available: list[str] = []
    for path in (repository_root / "artifacts/phase2-benchmark/public").glob("*.json"):
        value = _read(path)
        if value.get("split") == SplitLabel.DEVELOPMENT.value and path.stem not in excluded:
            available.append(path.stem)
    selected = sorted(available)
    needed = PAIR_COUNT_PER_STAGE * 2
    if len(selected) < needed:
        raise ConfigurationError("insufficient untouched tasks for deep validation")
    registry = cast(
        JsonObject,
        {
            "registry_version": "phase5-experience-deep-task-registry-v1",
            "generator_family_id": SOURCE_GENERATOR_FAMILY_ID,
            "excluded_prior_task_ids": sorted(excluded),
            "stage1": {
                "task_ids": selected[:PAIR_COUNT_PER_STAGE],
                "search_seeds": [STAGE1_SEED],
            },
            "stage2": {
                "task_ids": selected[PAIR_COUNT_PER_STAGE:needed],
                "search_seeds": [STAGE2_SEED],
            },
            "disjointness": (
                "stage1-and-stage2-mutually-disjoint-and-disjoint-from-all-prior-v2-"
                "training-validation-canary-and-development-registry-tasks-v1"
            ),
        },
    )
    return registry


def freeze_deep_validation_plan(*, repository_root: Path) -> JsonObject:
    output = repository_root / DEEP_OUTPUT_ROOT
    registry = _registry(repository_root)
    registry_hash = write_content_artifact(output / "task-registry.json", canonical_json(registry))
    stage1_children = PAIR_COUNT_PER_STAGE * 4
    stage2_children = PAIR_COUNT_PER_STAGE * 2
    plan = cast(
        JsonObject,
        {
            "plan_version": "phase5-experience-deep-analysis-and-exposure-plan-v1",
            "protocol": DEEP_PROTOCOL,
            "protocol_hash": DEEP_PROTOCOL_HASH,
            "task_registry_hash": registry_hash,
            "model": "gpt-5-mini-2025-08-07",
            "search": {
                "condition": "uniform-diverse-archive-v1",
                "batch_size": 4,
                "logical_requests_per_child": LOGICAL_REQUESTS_PER_CHILD,
                "model_request_attempt_cap": MODEL_REQUEST_ATTEMPT_CAP,
                "proposal_item_cap": PROPOSAL_ITEM_CAP,
                "oracle_call_cap": ORACLE_CALL_CAP,
                "continue_after_first_exact": True,
                "retrieval_bounds": {
                    "max_items": 4,
                    "max_bytes": 4096,
                    "max_tokens": 4096,
                },
            },
            "exposure_gate": {
                "minimum_distinct_tasks": 4,
                "minimum_matching_requests_total": 8,
                "mismatched_family_retrieval_allowed": False,
                "insufficient_status": "inconclusive-not-rejected",
            },
            "arm_counts": {
                "stage1_pairs_per_lesson": PAIR_COUNT_PER_STAGE,
                "stage1_shared_control_children": PAIR_COUNT_PER_STAGE,
                "stage1_sole_lesson_treatment_children": PAIR_COUNT_PER_STAGE * 3,
                "stage1_children": stage1_children,
                "stage2_pairs": PAIR_COUNT_PER_STAGE,
                "stage2_children_maximum": stage2_children,
            },
            "fail_closed_exposure_nano_usd": {
                "child": CHILD_CAP_NANO_USD,
                "stage1_maximum": stage1_children * CHILD_CAP_NANO_USD,
                "stage2_maximum": stage2_children * CHILD_CAP_NANO_USD,
                "combined_maximum": (stage1_children + stage2_children) * CHILD_CAP_NANO_USD,
            },
            "execution": {
                "maximum_parallel_children": MAX_PARALLEL_CHILDREN,
                "stop_before_canary": True,
            },
        },
    )
    plan_hash = write_content_artifact(output / "analysis-plan.json", canonical_json(plan))
    return {"task_registry_hash": registry_hash, "analysis_plan_hash": plan_hash}


def _deep_run(
    *,
    repository_root: Path,
    base: AppConfig,
    task_id: str,
    seed: int,
    arm: str,
    snapshot: ExperienceMemorySnapshot,
) -> DeepChild:
    run_id = f"P5V2D-{task_id[:8].upper()}-{seed}-{arm.upper()}"
    outcome, result = _run_arm(
        repository_root=repository_root,
        base=base,
        run_id=run_id,
        task_id=task_id,
        seed=seed,
        snapshot=snapshot,
        arm_id=arm,
        requests=LOGICAL_REQUESTS_PER_CHILD,
        child_cap=CHILD_CAP_NANO_USD,
        runs_root=DEEP_RUNS_ROOT,
        batch_size=4,
        proposal_item_cap=PROPOSAL_ITEM_CAP,
        oracle_call_cap=ORACLE_CALL_CAP,
        model_request_cap=MODEL_REQUEST_ATTEMPT_CAP,
        input_token_cap=2_000_000,
        cache_namespace="phase5-experience-v2-deep-validation-v1",
    )
    if outcome.status != "completed":
        raise ConfigurationError(f"deep validation child did not complete: {run_id}")
    return DeepChild(task_id, seed, arm, outcome, result)


def _run_jobs(
    *,
    repository_root: Path,
    base: AppConfig,
    jobs: list[tuple[str, int, str, ExperienceMemorySnapshot]],
) -> dict[tuple[str, int, str], DeepChild]:
    children: dict[tuple[str, int, str], DeepChild] = {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHILDREN) as executor:
        futures = {
            executor.submit(
                _deep_run,
                repository_root=repository_root,
                base=base,
                task_id=task_id,
                seed=seed,
                arm=arm,
                snapshot=snapshot,
            ): (task_id, seed, arm)
            for task_id, seed, arm, snapshot in jobs
        }
        for future in as_completed(futures):
            identity = futures[future]
            children[identity] = future.result()
    return children


def _local_accuracy_auc_ppm(run_directory: Path) -> int:
    connection = sqlite3.connect(f"file:{run_directory / 'run.sqlite3'}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT evaluation_index,result_json FROM evaluation ORDER BY evaluation_index"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != ORACLE_CALL_CAP:
        raise ConfigurationError("deep child did not consume the frozen oracle horizon")
    best_errors = 8
    numerator = 0
    for _index, result_text in rows:
        value = json.loads(str(result_text))
        errors = value.get("local_errors")
        cases = value.get("local_cases")
        if not isinstance(errors, int) or cases != 8:
            raise ConfigurationError("deep child local-error record is malformed")
        best_errors = min(best_errors, errors)
        numerator += 8 - best_errors
    return numerator * 1_000_000 // (8 * ORACLE_CALL_CAP)


def _exposure_count(run_directory: Path, lesson_id: str) -> int:
    count = 0
    for path in sorted((run_directory / "retrieval").glob("*.json")):
        selected = _read(path).get("selected_lesson_ids")
        if isinstance(selected, list) and lesson_id in selected:
            count += 1
    return count


def _stage1_pair(
    *,
    proposal: ExperienceLessonProposal,
    sole: PromotedExperienceLesson,
    control: DeepChild,
    treatment: DeepChild,
) -> JsonObject:
    baseline_exact, baseline_exact_auc = _metrics(control.result)
    treatment_exact, treatment_exact_auc = _metrics(treatment.result)
    exposure_count = _exposure_count(treatment.outcome.run_directory, sole.lesson_id)
    return {
        "pair_version": "phase5-experience-deep-individual-pair-v1",
        "lesson_proposal_id": proposal.proposal_id,
        "archive_representation_family": proposal.archive_representation_family.value,
        "task_id": control.task_id,
        "generator_family_id": SOURCE_GENERATOR_FAMILY_ID,
        "seed": control.seed,
        "baseline_exact": baseline_exact,
        "treatment_exact": treatment_exact,
        "baseline_normalized_exact_auc_ppm": baseline_exact_auc,
        "treatment_normalized_exact_auc_ppm": treatment_exact_auc,
        "exact_auc_gain_ppm": treatment_exact_auc - baseline_exact_auc,
        "baseline_local_accuracy_auc_ppm": _local_accuracy_auc_ppm(control.outcome.run_directory),
        "treatment_local_accuracy_auc_ppm": _local_accuracy_auc_ppm(
            treatment.outcome.run_directory
        ),
        "local_accuracy_auc_gain_ppm": _local_accuracy_auc_ppm(treatment.outcome.run_directory)
        - _local_accuracy_auc_ppm(control.outcome.run_directory),
        "matching_request_exposure_count": exposure_count,
        "treatment_memory_applied": exposure_count > 0,
        "control_run_id": control.outcome.run_id,
        "treatment_run_id": treatment.outcome.run_id,
    }


def _promote(
    *, proposal: ExperienceLessonProposal, pairs: list[JsonObject]
) -> tuple[str, list[str], PromotedExperienceLesson | None, JsonObject]:
    applied = [pair for pair in pairs if pair["treatment_memory_applied"] is True]
    exposed_tasks = {str(pair["task_id"]) for pair in applied}
    total_exposures = sum(
        _integer(pair["matching_request_exposure_count"], "matching exposure") for pair in applied
    )
    exact_gains = [_integer(pair["exact_auc_gain_ppm"], "exact AUC gain") for pair in applied]
    local_gains = [
        _integer(pair["local_accuracy_auc_gain_ppm"], "local AUC gain") for pair in applied
    ]
    exact_mean = sum(exact_gains) // max(1, len(exact_gains))
    local_mean = sum(local_gains) // max(1, len(local_gains))
    summary: JsonObject = {
        "exposed_task_count": len(exposed_tasks),
        "matching_request_exposure_count": total_exposures,
        "mean_exact_auc_gain_ppm": exact_mean,
        "mean_local_accuracy_auc_gain_ppm": local_mean,
    }
    insufficient: list[str] = []
    if len(exposed_tasks) < 4:
        insufficient.append("fewer-than-four-exposed-tasks")
    if total_exposures < 8:
        insufficient.append("fewer-than-eight-matching-request-exposures")
    if insufficient:
        return "inconclusive", insufficient, None, summary
    if any(pair["baseline_exact"] is True and pair["treatment_exact"] is False for pair in applied):
        return "rejected", ["exact-solve-regression"], None, summary
    if exact_mean < 0:
        return "rejected", ["negative-mean-exact-auc-gain"], None, summary
    if exact_mean == 0 and local_mean <= 0:
        return "rejected", ["nonpositive-exact-and-local-screening-gain"], None, summary
    validation_hashes = tuple(sorted(sha256_json(pair) for pair in applied))
    gain = exact_mean if exact_mean > 0 else local_mean
    return (
        "promoted",
        [],
        PromotedExperienceLesson(proposal, validation_hashes, gain),
        summary,
    )


def _audit_prompt_isolation(*, control: DeepChild, treatment: DeepChild) -> None:
    control_prompt = (control.outcome.run_directory / "prompts/request-00000.json").read_text(
        encoding="utf-8"
    )
    treatment_prompt = (treatment.outcome.run_directory / "prompts/request-00000.json").read_text(
        encoding="utf-8"
    )
    assert_experience_prompt_isolation(control_prompt, treatment_prompt)


def run_deep_validation(*, repository_root: Path) -> JsonObject:
    """Run both validation stages and stop after snapshot freeze, before canary."""

    frozen = freeze_deep_validation_plan(repository_root=repository_root)
    registry = _registry(repository_root)
    proposals = _lessons(repository_root)
    corpus = extract_phase4_condition_c_corpus(repository_root=repository_root)
    catalog = {item.evidence_id: item for item in corpus.evidence}
    if any(
        any(evidence_id not in catalog for evidence_id in proposal.source_evidence_ids)
        for proposal in proposals
    ):
        raise ConfigurationError("deep validation lesson evidence is unavailable")
    base = load_config(repository_root / BASE_CONFIG_PATH)
    empty = ExperienceMemorySnapshot(DEEP_PROTOCOL_HASH, ())
    stage1 = cast(dict[str, list[object]], registry["stage1"])
    stage1_tasks = cast(list[str], stage1["task_ids"])
    stage1_seed = cast(list[int], stage1["search_seeds"])[0]
    sole_snapshots: dict[str, tuple[PromotedExperienceLesson, ExperienceMemorySnapshot]] = {}
    for proposal in proposals:
        sole = PromotedExperienceLesson(proposal, (), 0)
        sole_snapshots[proposal.proposal_id] = (
            sole,
            ExperienceMemorySnapshot(DEEP_PROTOCOL_HASH, (sole,)),
        )
    stage1_jobs: list[tuple[str, int, str, ExperienceMemorySnapshot]] = []
    for task_id in stage1_tasks:
        stage1_jobs.append((task_id, stage1_seed, "control", empty))
        for proposal in proposals:
            _sole, snapshot = sole_snapshots[proposal.proposal_id]
            stage1_jobs.append(
                (
                    task_id,
                    stage1_seed,
                    proposal.archive_representation_family.value,
                    snapshot,
                )
            )
    children = _run_jobs(repository_root=repository_root, base=base, jobs=stage1_jobs)
    promoted: list[PromotedExperienceLesson] = []
    stage1_lessons: list[JsonObject] = []
    for proposal in proposals:
        sole, _snapshot = sole_snapshots[proposal.proposal_id]
        pairs: list[JsonObject] = []
        arm = proposal.archive_representation_family.value
        for task_id in stage1_tasks:
            control = children[(task_id, stage1_seed, "control")]
            treatment = children[(task_id, stage1_seed, arm)]
            _audit_prompt_isolation(control=control, treatment=treatment)
            pairs.append(
                _stage1_pair(
                    proposal=proposal,
                    sole=sole,
                    control=control,
                    treatment=treatment,
                )
            )
        status, reasons, promoted_lesson, summary = _promote(proposal=proposal, pairs=pairs)
        if promoted_lesson is not None:
            promoted.append(promoted_lesson)
        stage1_lessons.append(
            cast(
                JsonObject,
                {
                    "proposal_id": proposal.proposal_id,
                    "family": proposal.archive_representation_family.value,
                    "status": status,
                    "reasons": reasons,
                    "summary": summary,
                    "pairs": pairs,
                },
            )
        )
    output = repository_root / DEEP_OUTPUT_ROOT
    stage1_value = cast(
        JsonObject,
        {
            "stage": "deep-individual-lesson-screen-v1",
            "lessons": stage1_lessons,
            "promoted_lesson_ids": sorted(lesson.lesson_id for lesson in promoted),
        },
    )
    stage1_hash = write_content_artifact(
        output / "stage1-individual-screen.json", canonical_json(stage1_value)
    )
    if not promoted:
        stage2_value: JsonObject = {
            "stage": "deep-promoted-bundle-confirmation-v1",
            "status": "inconclusive",
            "reasons": ["no-lessons-passed-deep-screening"],
            "provider_requests_required": 0,
            "pairs": [],
        }
        stage2_hash = write_content_artifact(
            output / "stage2-bundle-validation.json", canonical_json(stage2_value)
        )
        return {
            "status": "stopped-before-canary-no-validated-snapshot",
            "stage1_artifact_hash": stage1_hash,
            "stage2_artifact_hash": stage2_hash,
            "promoted_lesson_count": 0,
            **frozen,
        }
    bundle = ExperienceMemorySnapshot(
        DEEP_PROTOCOL_HASH, tuple(sorted(promoted, key=lambda item: item.lesson_id))
    )
    stage2 = cast(dict[str, list[object]], registry["stage2"])
    stage2_tasks = cast(list[str], stage2["task_ids"])
    stage2_seed = cast(list[int], stage2["search_seeds"])[0]
    stage2_jobs: list[tuple[str, int, str, ExperienceMemorySnapshot]] = []
    for task_id in stage2_tasks:
        stage2_jobs.extend(
            (
                (task_id, stage2_seed, "control", empty),
                (task_id, stage2_seed, "bundle", bundle),
            )
        )
    stage2_children = _run_jobs(repository_root=repository_root, base=base, jobs=stage2_jobs)
    stage2_pairs: list[BundleValidationPair] = []
    for task_id in stage2_tasks:
        control = stage2_children[(task_id, stage2_seed, "control")]
        treatment = stage2_children[(task_id, stage2_seed, "bundle")]
        _audit_prompt_isolation(control=control, treatment=treatment)
        baseline_exact, baseline_auc = _metrics(control.result)
        treatment_exact, treatment_auc = _metrics(treatment.result)
        stage2_pairs.append(
            BundleValidationPair(
                task_id,
                SOURCE_GENERATOR_FAMILY_ID,
                stage2_seed,
                baseline_exact,
                treatment_exact,
                baseline_auc,
                treatment_auc,
                _selected_lesson_ids(treatment.outcome.run_directory),
            )
        )
    decision = evaluate_bundle_validation(
        protocol_hash=DEEP_PROTOCOL_HASH,
        promoted_lessons=tuple(promoted),
        individual_screen_task_ids=frozenset(stage1_tasks),
        validation_pairs=tuple(stage2_pairs),
        minimum_validation_tasks_with_exposure=4,
        minimum_validation_generator_families=1,
        minimum_exposure_tasks_per_lesson=2,
    )
    stage2_value = {
        "stage": "deep-promoted-bundle-confirmation-v1",
        "status": decision.status,
        "reasons": list(decision.reasons),
        "pairs": [pair.to_value() for pair in stage2_pairs],
    }
    stage2_hash = write_content_artifact(
        output / "stage2-bundle-validation.json", canonical_json(stage2_value)
    )
    if decision.snapshot is None:
        return {
            "status": "stopped-before-canary-no-validated-snapshot",
            "stage1_artifact_hash": stage1_hash,
            "stage2_artifact_hash": stage2_hash,
            "stage2_status": decision.status,
            "stage2_reasons": list(decision.reasons),
            "promoted_lesson_count": len(promoted),
            **frozen,
        }
    snapshot_hash = write_content_artifact(
        output / "frozen-memory-snapshot.json",
        canonical_json(decision.snapshot.to_value()),
    )
    completion: JsonObject = {
        "status": "validated-snapshot-frozen-stopped-before-canary",
        "stage1_artifact_hash": stage1_hash,
        "stage2_artifact_hash": stage2_hash,
        "memory_snapshot_artifact_hash": snapshot_hash,
        "memory_snapshot_hash": decision.snapshot.snapshot_hash,
        "promoted_lesson_count": len(promoted),
        "canary_executed": False,
        **frozen,
    }
    write_content_artifact(output / "completion.json", canonical_json(completion))
    return completion
