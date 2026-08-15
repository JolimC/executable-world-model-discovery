"""Prospective confirmation of the Phase 5 position-specific experience lesson."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from world_model_search.config import AppConfig, load_config
from world_model_search.errors import ConfigurationError
from world_model_search.evaluation.phase5_experience_deep import (
    CHILD_CAP_NANO_USD,
    DEEP_OUTPUT_ROOT,
    LOGICAL_REQUESTS_PER_CHILD,
    MAX_PARALLEL_CHILDREN,
    MODEL_REQUEST_ATTEMPT_CAP,
    ORACLE_CALL_CAP,
    PROPOSAL_ITEM_CAP,
    DeepChild,
    _audit_prompt_isolation,
    _exposure_count,
    _integer,
    _local_accuracy_auc_ppm,
)
from world_model_search.evaluation.phase5_experience_live import (
    BASE_CONFIG_PATH,
    _lessons,
    _metrics,
    _read,
    _run_arm,
)
from world_model_search.memory.experience import (
    ExperienceMemorySnapshot,
    PromotedExperienceLesson,
)
from world_model_search.persistence.artifacts import write_content_artifact
from world_model_search.serialization import JsonObject, canonical_json, sha256_json

OUTPUT_ROOT = Path("artifacts/phase5-experience-v2/position-confirmatory-v1")
RUNS_ROOT = Path("artifacts/phase5-experience-v2/position-confirmatory-runs-v1")
POSITION_FAMILY = "position-specific"
STAGE2_SEED = 62_001
PAIR_COUNT = 8
INPUT_TOKEN_CAP = 2_000_000
PROTOCOL: JsonObject = {
    "protocol_version": "phase5-position-specific-confirmatory-v1",
    "candidate_selection": "highest-aggregate-signal-from-deep-stage1-v1",
    "analysis_population": "all-eight-frozen-pairs-intent-to-treat-v1",
    "minimum_exposure_tasks": 4,
    "minimum_matching_request_exposures_total": 8,
    "insufficient_exposure_status": "inconclusive-not-failed",
    "maximum_exact_solve_regressions": 1,
    "requires_positive_net_exact_solves": True,
    "requires_positive_mean_normalized_exact_auc_gain": True,
    "requires_nonnegative_mean_local_accuracy_auc_gain": True,
    "lesson_may_cross_representation_family": False,
    "canary_authorized": False,
}
PROTOCOL_HASH = sha256_json(PROTOCOL)


@dataclass(frozen=True, slots=True)
class ConfirmatoryDecision:
    status: str
    reasons: tuple[str, ...]
    summary: JsonObject


def _candidate(repository_root: Path) -> PromotedExperienceLesson:
    proposals = tuple(
        proposal
        for proposal in _lessons(repository_root)
        if proposal.archive_representation_family.value == POSITION_FAMILY
    )
    if len(proposals) != 1:
        raise ConfigurationError("position-specific confirmatory candidate is not unique")
    return PromotedExperienceLesson(proposals[0], (), 0)


def _task_binding(repository_root: Path) -> JsonObject:
    deep_registry = _read(repository_root / DEEP_OUTPUT_ROOT / "task-registry.json")
    stage2 = deep_registry.get("stage2")
    if not isinstance(stage2, dict):
        raise ConfigurationError("deep Stage 2 task binding is malformed")
    task_ids = stage2.get("task_ids")
    seeds = stage2.get("search_seeds")
    if (
        not isinstance(task_ids, list)
        or len(task_ids) != PAIR_COUNT
        or not all(isinstance(value, str) for value in task_ids)
        or seeds != [STAGE2_SEED]
    ):
        raise ConfigurationError("deep Stage 2 task binding differs from the frozen contract")
    runs = repository_root / RUNS_ROOT
    frozen_binding = repository_root / OUTPUT_ROOT / "task-binding.json"
    if runs.exists() and any(runs.iterdir()) and not frozen_binding.is_file():
        raise ConfigurationError("confirmatory run root is not unused")
    return cast(
        JsonObject,
        {
            "binding_version": "phase5-position-specific-confirmatory-task-binding-v1",
            "source_registry_hash": sha256_json(deep_registry),
            "task_ids": cast(list[str], task_ids),
            "search_seed": STAGE2_SEED,
            "prior_provider_children_on_these_bindings": 0,
        },
    )


def freeze_confirmatory_plan(*, repository_root: Path) -> JsonObject:
    output = repository_root / OUTPUT_ROOT
    candidate = _candidate(repository_root)
    binding = _task_binding(repository_root)
    binding_hash = write_content_artifact(output / "task-binding.json", canonical_json(binding))
    plan = cast(
        JsonObject,
        {
            "plan_version": "phase5-position-specific-confirmatory-analysis-plan-v1",
            "protocol": PROTOCOL,
            "protocol_hash": PROTOCOL_HASH,
            "task_binding_hash": binding_hash,
            "candidate_lesson_id": candidate.lesson_id,
            "candidate_proposal_id": candidate.proposal.proposal_id,
            "candidate_family": POSITION_FAMILY,
            "candidate_source_stage1_hash": (
                "f6e6ed89cea50adca0eb6e55df5bc3d585a3081dab5a1441d53b074044aff935"
            ),
            "model": "gpt-5-mini-2025-08-07",
            "search": {
                "condition": "uniform-diverse-archive-v1",
                "batch_size": 4,
                "logical_requests_per_child": LOGICAL_REQUESTS_PER_CHILD,
                "model_request_attempt_cap": MODEL_REQUEST_ATTEMPT_CAP,
                "proposal_item_cap": PROPOSAL_ITEM_CAP,
                "oracle_call_cap": ORACLE_CALL_CAP,
                "continue_after_first_exact": True,
            },
            "execution": {
                "paired_tasks": PAIR_COUNT,
                "control_children": PAIR_COUNT,
                "sole_lesson_treatment_children": PAIR_COUNT,
                "maximum_parallel_children": MAX_PARALLEL_CHILDREN,
                "stop_before_canary": True,
            },
            "fail_closed_exposure_nano_usd": {
                "child": CHILD_CAP_NANO_USD,
                "combined_maximum": 2 * PAIR_COUNT * CHILD_CAP_NANO_USD,
            },
        },
    )
    plan_hash = write_content_artifact(output / "analysis-plan.json", canonical_json(plan))
    return {"task_binding_hash": binding_hash, "analysis_plan_hash": plan_hash}


def _run_child(
    *,
    repository_root: Path,
    base: AppConfig,
    task_id: str,
    arm: str,
    snapshot: ExperienceMemorySnapshot,
) -> DeepChild:
    run_id = f"P5V2C-{task_id[:8].upper()}-{STAGE2_SEED}-{arm.upper()}"
    outcome, result = _run_arm(
        repository_root=repository_root,
        base=base,
        run_id=run_id,
        task_id=task_id,
        seed=STAGE2_SEED,
        snapshot=snapshot,
        arm_id=arm,
        stage="development",
        requests=LOGICAL_REQUESTS_PER_CHILD,
        child_cap=CHILD_CAP_NANO_USD,
        runs_root=RUNS_ROOT,
        batch_size=4,
        proposal_item_cap=PROPOSAL_ITEM_CAP,
        oracle_call_cap=ORACLE_CALL_CAP,
        model_request_cap=MODEL_REQUEST_ATTEMPT_CAP,
        input_token_cap=INPUT_TOKEN_CAP,
        cache_namespace="phase5-position-specific-confirmatory-v1",
    )
    if outcome.status != "completed":
        raise ConfigurationError(f"confirmatory child did not complete: {run_id}")
    return DeepChild(task_id, STAGE2_SEED, arm, outcome, result)


def _run_jobs(
    *,
    repository_root: Path,
    base: AppConfig,
    jobs: list[tuple[str, str, ExperienceMemorySnapshot]],
) -> dict[tuple[str, str], DeepChild]:
    children: dict[tuple[str, str], DeepChild] = {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHILDREN) as executor:
        futures = {
            executor.submit(
                _run_child,
                repository_root=repository_root,
                base=base,
                task_id=task_id,
                arm=arm,
                snapshot=snapshot,
            ): (task_id, arm)
            for task_id, arm, snapshot in jobs
        }
        for future in as_completed(futures):
            identity = futures[future]
            children[identity] = future.result()
    return children


def _pair(
    *, control: DeepChild, treatment: DeepChild, candidate: PromotedExperienceLesson
) -> JsonObject:
    baseline_exact, baseline_exact_auc = _metrics(control.result)
    treatment_exact, treatment_exact_auc = _metrics(treatment.result)
    baseline_local = _local_accuracy_auc_ppm(control.outcome.run_directory)
    treatment_local = _local_accuracy_auc_ppm(treatment.outcome.run_directory)
    exposure = _exposure_count(treatment.outcome.run_directory, candidate.lesson_id)
    return {
        "pair_version": "phase5-position-specific-confirmatory-pair-v1",
        "task_id": control.task_id,
        "seed": STAGE2_SEED,
        "baseline_exact": baseline_exact,
        "treatment_exact": treatment_exact,
        "exact_auc_gain_ppm": treatment_exact_auc - baseline_exact_auc,
        "local_accuracy_auc_gain_ppm": treatment_local - baseline_local,
        "matching_request_exposure_count": exposure,
        "treatment_memory_applied": exposure > 0,
        "control_run_id": control.outcome.run_id,
        "treatment_run_id": treatment.outcome.run_id,
    }


def evaluate_confirmatory(pairs: list[JsonObject]) -> ConfirmatoryDecision:
    exposed = [pair for pair in pairs if pair["treatment_memory_applied"] is True]
    exposure_total = sum(
        _integer(pair["matching_request_exposure_count"], "matching exposure") for pair in exposed
    )
    baseline_exact_count = sum(pair["baseline_exact"] is True for pair in pairs)
    treatment_exact_count = sum(pair["treatment_exact"] is True for pair in pairs)
    regression_count = sum(
        pair["baseline_exact"] is True and pair["treatment_exact"] is False for pair in pairs
    )
    gain_count = sum(
        pair["baseline_exact"] is False and pair["treatment_exact"] is True for pair in pairs
    )
    exact_mean = sum(
        _integer(pair["exact_auc_gain_ppm"], "exact AUC gain") for pair in pairs
    ) // len(pairs)
    local_mean = sum(
        _integer(pair["local_accuracy_auc_gain_ppm"], "local AUC gain") for pair in pairs
    ) // len(pairs)
    summary: JsonObject = {
        "pair_count": len(pairs),
        "exposed_task_count": len(exposed),
        "matching_request_exposure_count": exposure_total,
        "baseline_exact_count": baseline_exact_count,
        "treatment_exact_count": treatment_exact_count,
        "exact_gain_task_count": gain_count,
        "exact_regression_task_count": regression_count,
        "net_exact_solve_gain": treatment_exact_count - baseline_exact_count,
        "mean_exact_auc_gain_ppm": exact_mean,
        "mean_local_accuracy_auc_gain_ppm": local_mean,
    }
    insufficient: list[str] = []
    if len(exposed) < 4:
        insufficient.append("fewer-than-four-exposed-tasks")
    if exposure_total < 8:
        insufficient.append("fewer-than-eight-matching-request-exposures")
    if insufficient:
        return ConfirmatoryDecision("inconclusive", tuple(insufficient), summary)
    failures: list[str] = []
    if regression_count > 1:
        failures.append("more-than-one-exact-solve-regression")
    if treatment_exact_count <= baseline_exact_count:
        failures.append("nonpositive-net-exact-solves")
    if exact_mean <= 0:
        failures.append("nonpositive-mean-exact-auc-gain")
    if local_mean < 0:
        failures.append("negative-mean-local-accuracy-auc-gain")
    return ConfirmatoryDecision("failed" if failures else "passed", tuple(failures), summary)


def run_confirmatory_validation(*, repository_root: Path) -> JsonObject:
    """Run the frozen confirmation and stop before the canary."""

    frozen = freeze_confirmatory_plan(repository_root=repository_root)
    binding = _read(repository_root / OUTPUT_ROOT / "task-binding.json")
    task_ids = cast(list[str], binding["task_ids"])
    candidate = _candidate(repository_root)
    empty = ExperienceMemorySnapshot(PROTOCOL_HASH, ())
    treatment_snapshot = ExperienceMemorySnapshot(PROTOCOL_HASH, (candidate,))
    jobs = [
        job
        for task_id in task_ids
        for job in (
            (task_id, "control", empty),
            (task_id, POSITION_FAMILY, treatment_snapshot),
        )
    ]
    base = load_config(repository_root / BASE_CONFIG_PATH)
    children = _run_jobs(repository_root=repository_root, base=base, jobs=jobs)
    pairs: list[JsonObject] = []
    for task_id in task_ids:
        control = children[(task_id, "control")]
        treatment = children[(task_id, POSITION_FAMILY)]
        _audit_prompt_isolation(control=control, treatment=treatment)
        pairs.append(_pair(control=control, treatment=treatment, candidate=candidate))
    decision = evaluate_confirmatory(pairs)
    output = repository_root / OUTPUT_ROOT
    result = cast(
        JsonObject,
        {
            "result_version": "phase5-position-specific-confirmatory-result-v1",
            "protocol_hash": PROTOCOL_HASH,
            "status": decision.status,
            "reasons": list(decision.reasons),
            "summary": decision.summary,
            "pairs": pairs,
            "canary_executed": False,
            **frozen,
        },
    )
    result_hash = write_content_artifact(output / "result.json", canonical_json(result))
    completion: JsonObject = {
        "status": f"{decision.status}-stopped-before-canary",
        "result_artifact_hash": result_hash,
        "canary_executed": False,
        **frozen,
    }
    if decision.status == "passed":
        validated = PromotedExperienceLesson(
            candidate.proposal,
            tuple(sorted(sha256_json(pair) for pair in pairs)),
            _integer(decision.summary["mean_exact_auc_gain_ppm"], "mean exact AUC gain"),
        )
        snapshot = ExperienceMemorySnapshot(PROTOCOL_HASH, (validated,))
        completion["memory_snapshot_artifact_hash"] = write_content_artifact(
            output / "frozen-memory-snapshot.json", canonical_json(snapshot.to_value())
        )
        completion["memory_snapshot_hash"] = snapshot.snapshot_hash
    write_content_artifact(output / "completion.json", canonical_json(completion))
    return completion
