"""Prospective validation, canary, and development pilot for experience memory v2."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from world_model_search.config import AppConfig, load_config
from world_model_search.domain.types import SplitLabel
from world_model_search.errors import ConfigurationError
from world_model_search.evaluation.phase5_experience import (
    SOURCE_GENERATOR_FAMILY_ID,
    extract_phase4_condition_c_corpus,
)
from world_model_search.memory.experience import (
    BundleValidationPair,
    ExperienceLessonProposal,
    ExperienceMemorySnapshot,
    LessonValidationPair,
    PromotedExperienceLesson,
    ValidationStage,
    evaluate_bundle_validation,
    evaluate_lesson_promotion,
)
from world_model_search.model.phase5_experience_prompts import (
    assert_experience_prompt_isolation,
)
from world_model_search.persistence.artifacts import write_content_artifact
from world_model_search.search.archive import RepresentationFamily
from world_model_search.search.phase4 import (
    Phase4Outcome,
    resume_phase4_run,
    start_phase4_run,
)
from world_model_search.serialization import (
    JsonObject,
    canonical_json,
    parse_json_object,
    sha256_json,
)

OUTPUT_ROOT = Path("artifacts/phase5-experience-v2/prospective-v1")
RUNS_ROOT = Path("artifacts/phase5-experience-v2/runs-v1")
LESSONS_PATH = Path(
    "artifacts/phase5-experience-v2/retrospective-training-v3/induction/frozen-lessons.json"
)
BASE_CONFIG_PATH = Path("configs/phase4-openai-pilot.yaml")
POLICY_PATH = Path("configs/project-dual-budget-policy-v2.yaml")
LEDGER_PATH = Path("local_state/project-dual-budget-ledger.sqlite3")
SOURCE_PHASE4_TASKS = frozenset(
    {
        "01dda10a838f4423e72b90b8",
        "01fa64a668ef5086dc713454",
        "0bcf2a466f9f62472b0604d7",
        "1400d05ad196c9fa802cd094",
        "1cddb7fea9c43ce8b5f93584",
        "23582946627e0949dc02e473",
        "23abc8920b7d2ea2bb1bf9cf",
        "2b41ea8918fa2be0a0389e8c",
        "2e323a2b94479b332489f218",
        "30bccf698deb8ce78d0fef77",
    }
)
PROTOCOL: JsonObject = {
    "protocol_version": "phase5-experience-prospective-validation-v1",
    "source_family_count": 1,
    "retrieval_key": "selected-public-archive-representation-family-v1",
    "stage1_minimum_exposure_tasks": 4,
    "stage2_minimum_any_exposure_tasks": 4,
    "stage2_minimum_exposure_tasks_per_lesson": 2,
    "exact_regression_allowed": False,
    "positive_mean_normalized_exact_auc_required": True,
    "development_pairing": "exact-task-seed-memory-off-vs-memory-on-v1",
}
PROTOCOL_HASH = sha256_json(PROTOCOL)


def _read(path: Path) -> JsonObject:
    return parse_json_object(path.read_text(encoding="utf-8"))


def _lessons(repository_root: Path) -> tuple[ExperienceLessonProposal, ...]:
    raw = _read(repository_root / LESSONS_PATH).get("lessons")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ConfigurationError("frozen experience lesson set is malformed")
    values: list[ExperienceLessonProposal] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ConfigurationError("frozen experience lesson is malformed")
        evidence = item.get("source_evidence_ids")
        if not isinstance(evidence, list) or not all(isinstance(value, str) for value in evidence):
            raise ConfigurationError("frozen lesson evidence IDs are malformed")
        proposal = ExperienceLessonProposal(
            lesson_text=str(item.get("lesson_text")),
            archive_representation_family=RepresentationFamily(
                str(item.get("archive_representation_family"))
            ),
            source_evidence_ids=tuple(cast(list[str], evidence)),
        )
        if proposal.proposal_id != item.get("proposal_id"):
            raise ConfigurationError("frozen lesson proposal hash differs")
        values.append(proposal)
    return tuple(sorted(values, key=lambda item: item.proposal_id))


def _registry(repository_root: Path) -> JsonObject:
    available: list[str] = []
    for path in (repository_root / "artifacts/phase2-benchmark/public").glob("*.json"):
        value = _read(path)
        if (
            value.get("split") == SplitLabel.DEVELOPMENT.value
            and path.stem not in SOURCE_PHASE4_TASKS
        ):
            available.append(path.stem)
    selected = sorted(available)
    if len(selected) < 23:
        raise ConfigurationError("insufficient fresh within-family development tasks")
    registry = cast(
        JsonObject,
        {
            "registry_version": "phase5-experience-v2-prospective-task-registry-v1",
            "generator_family_id": SOURCE_GENERATOR_FAMILY_ID,
            "source_training_task_ids": sorted(SOURCE_PHASE4_TASKS),
            "stage1": {"task_ids": selected[:8], "search_seeds": [51001]},
            "stage2": {"task_ids": selected[8:16], "search_seeds": [52001]},
            "canary": {"task_ids": selected[16:17], "search_seeds": [53001]},
            "development": {
                "task_ids": selected[17:23],
                "search_seeds": [54001, 54002],
            },
            "disjointness": (
                "all-task-ids-mutually-disjoint-and-disjoint-from-retrospective-source-v1"
            ),
        },
    )
    all_ids = [
        task_id
        for stage in ("stage1", "stage2", "canary", "development")
        for task_id in cast(dict[str, list[object]], registry[stage])["task_ids"]
    ]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) & SOURCE_PHASE4_TASKS:
        raise ConfigurationError("prospective task registry is not disjoint")
    return registry


def freeze_phase5_experience_plan(*, repository_root: Path) -> JsonObject:
    registry = _registry(repository_root)
    output = repository_root / OUTPUT_ROOT
    registry_hash = write_content_artifact(output / "task-registry.json", canonical_json(registry))
    plan: JsonObject = {
        "plan_version": "phase5-experience-v2-frozen-analysis-and-exposure-plan-v1",
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "task_registry_hash": registry_hash,
        "search": {
            "condition": "uniform-diverse-archive-v1",
            "batch_size": 1,
            "validation_requests_per_child": 16,
            "canary_requests_per_child": 4,
            "validation_oracle_calls_per_child": 23,
            "canary_oracle_calls_per_child": 11,
            "retrieval_bounds": {"max_items": 4, "max_bytes": 4096, "max_tokens": 4096},
        },
        "arm_counts": {
            "stage1_children": 32,
            "stage2_children_maximum": 16,
            "canary_children": 2,
            "development_children": 24,
            "development_matched_task_seed_pairs": 12,
        },
        "fail_closed_exposure_nano_usd": {
            "validation_or_development_child": 120_000_000,
            "canary_child": 40_000_000,
            "stage1_maximum": 3_840_000_000,
            "stage2_maximum": 1_920_000_000,
            "development_maximum": 2_880_000_000,
            "canary_maximum": 80_000_000,
        },
        "primary_endpoint": "paired-normalized-exact-solve-auc-v1",
        "gate_order": ["individual-screen", "bundle-validation", "canary", "development"],
    }
    plan_hash = write_content_artifact(output / "analysis-plan.json", canonical_json(plan))
    return {"registry_hash": registry_hash, "analysis_plan_hash": plan_hash}


def _config(
    base: AppConfig,
    *,
    task_id: str,
    seed: int,
    stage: str,
    requests: int,
    child_cap: int,
    runs_root: Path = RUNS_ROOT,
    batch_size: int = 1,
    proposal_item_cap: int | None = None,
    oracle_call_cap: int | None = None,
    model_request_cap: int | None = None,
    input_token_cap: int = 400_000,
    cache_namespace: str = "phase5-experience-v2-prospective-v1",
) -> AppConfig:
    if base.cache is None or base.phase4_budget is None or base.phase4_policy is None:
        raise ConfigurationError("Phase 5 experience base configuration is incomplete")
    proposal_cap = proposal_item_cap if proposal_item_cap is not None else requests * batch_size
    oracle_calls = oracle_call_cap if oracle_call_cap is not None else 7 + proposal_cap
    request_cap = model_request_cap if model_request_cap is not None else requests
    return replace(
        base,
        run=replace(
            base.run,
            root=runs_root,
            seed=seed,
            task_id=task_id,
            split=SplitLabel.DEVELOPMENT,
            condition_id="uniform-diverse-archive-v1",
        ),
        proposer=replace(base.proposer, batch_size=batch_size),
        cache=replace(base.cache, namespace=cache_namespace),
        phase4_budget=replace(
            base.phase4_budget,
            model_request_cap=request_cap,
            input_token_cap=input_token_cap,
            output_token_cap=request_cap * 2048,
            total_token_cap=input_token_cap + request_cap * 2048,
            proposal_item_cap=proposal_cap,
            oracle_call_cap=oracle_calls,
            child_nano_usd_cap=child_cap,
        ),
        phase4_policy=replace(
            base.phase4_policy,
            stage=stage,
            price_policy=POLICY_PATH,
            ledger=LEDGER_PATH,
        ),
    )


def _run_arm(
    *,
    repository_root: Path,
    base: AppConfig,
    run_id: str,
    task_id: str,
    seed: int,
    snapshot: ExperienceMemorySnapshot,
    arm_id: str,
    stage: str = "development",
    requests: int = 16,
    child_cap: int = 120_000_000,
    allow_live_model: bool = True,
    runs_root: Path = RUNS_ROOT,
    batch_size: int = 1,
    proposal_item_cap: int | None = None,
    oracle_call_cap: int | None = None,
    model_request_cap: int | None = None,
    input_token_cap: int = 400_000,
    cache_namespace: str = "phase5-experience-v2-prospective-v1",
) -> tuple[Phase4Outcome, JsonObject]:
    config = _config(
        base,
        task_id=task_id,
        seed=seed,
        stage=stage,
        requests=requests,
        child_cap=child_cap,
        runs_root=runs_root,
        batch_size=batch_size,
        proposal_item_cap=proposal_item_cap,
        oracle_call_cap=oracle_call_cap,
        model_request_cap=model_request_cap,
        input_token_cap=input_token_cap,
        cache_namespace=cache_namespace,
    )
    directory = repository_root / runs_root / run_id
    if directory.exists():
        manifest = _read(directory / "manifest.json")
        outcome = resume_phase4_run(
            repository_root=repository_root,
            run_directory=directory,
            config=config,
            manifest=manifest,
            interrupt_after=None,
            allow_live_model=allow_live_model,
            experience_snapshot=snapshot,
            experience_arm_id=arm_id,
        )
    else:
        outcome = start_phase4_run(
            repository_root=repository_root,
            config=config,
            config_source=canonical_json(config.to_mapping()),
            run_id=run_id,
            interrupt_after=None,
            allow_live_model=allow_live_model,
            experience_snapshot=snapshot,
            experience_arm_id=arm_id,
        )
    return outcome, _read(outcome.run_directory / "results.json")


def _selected_lesson_ids(run_directory: Path) -> tuple[str, ...]:
    selected: set[str] = set()
    for path in sorted((run_directory / "retrieval").glob("*.json")):
        values = _read(path).get("selected_lesson_ids")
        if isinstance(values, list):
            selected.update(str(value) for value in values)
    return tuple(sorted(selected))


def _metrics(result: JsonObject) -> tuple[bool, int]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ConfigurationError("Phase 5 child has no metrics")
    numerator = metrics.get("exact_auc_numerator")
    denominator = metrics.get("exact_auc_denominator")
    exact = metrics.get("final_exact_solved")
    if (
        not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or not isinstance(exact, bool)
    ):
        raise ConfigurationError("Phase 5 child metrics are malformed")
    return exact, numerator * 1_000_000 // denominator


def _run_id(prefix: str, task_id: str, seed: int, arm: str) -> str:
    return f"P5V2-{prefix}-{task_id[:8].upper()}-{seed}-{arm.upper()}"


def run_phase5_experience_steps_3_through_8(*, repository_root: Path) -> JsonObject:
    """Run through development, returning early at the two user-designated stop gates."""

    freeze = freeze_phase5_experience_plan(repository_root=repository_root)
    registry = _registry(repository_root)
    proposals = _lessons(repository_root)
    corpus = extract_phase4_condition_c_corpus(repository_root=repository_root)
    catalog = {item.evidence_id: item for item in corpus.evidence}
    base = load_config(repository_root / BASE_CONFIG_PATH)
    empty = ExperienceMemorySnapshot(PROTOCOL_HASH, ())
    stage1 = cast(dict[str, list[object]], registry["stage1"])
    controls: dict[tuple[str, int], tuple[Phase4Outcome, JsonObject]] = {}
    for task_id in cast(list[str], stage1["task_ids"]):
        for seed in cast(list[int], stage1["search_seeds"]):
            controls[(task_id, seed)] = _run_arm(
                repository_root=repository_root,
                base=base,
                run_id=_run_id("S1", task_id, seed, "control"),
                task_id=task_id,
                seed=seed,
                snapshot=empty,
                arm_id="empty-experience-memory",
            )
    promoted: list[PromotedExperienceLesson] = []
    stage1_report: list[JsonObject] = []
    stage1_pairs_all: list[LessonValidationPair] = []
    for proposal in proposals:
        sole = PromotedExperienceLesson(proposal, (), 0)
        snapshot = ExperienceMemorySnapshot(PROTOCOL_HASH, (sole,))
        pairs: list[LessonValidationPair] = []
        for task_id in cast(list[str], stage1["task_ids"]):
            for seed in cast(list[int], stage1["search_seeds"]):
                control_outcome, control_result = controls[(task_id, seed)]
                treatment_outcome, treatment_result = _run_arm(
                    repository_root=repository_root,
                    base=base,
                    run_id=_run_id(
                        "S1", task_id, seed, proposal.archive_representation_family.value
                    ),
                    task_id=task_id,
                    seed=seed,
                    snapshot=snapshot,
                    arm_id=f"sole-{proposal.proposal_id}",
                )
                if control_outcome.status != "completed" or treatment_outcome.status != "completed":
                    raise ConfigurationError("Stage 1 child did not complete")
                baseline_exact, baseline_auc = _metrics(control_result)
                treatment_exact, treatment_auc = _metrics(treatment_result)
                pairs.append(
                    LessonValidationPair(
                        ValidationStage.INDIVIDUAL_LESSON,
                        proposal.proposal_id,
                        task_id,
                        SOURCE_GENERATOR_FAMILY_ID,
                        seed,
                        proposal.archive_representation_family,
                        baseline_exact,
                        treatment_exact,
                        baseline_auc,
                        treatment_auc,
                        sole.lesson_id in _selected_lesson_ids(treatment_outcome.run_directory),
                    )
                )
        decision = evaluate_lesson_promotion(
            proposal=proposal,
            evidence_catalog=catalog,
            validation_pairs=tuple(pairs),
        )
        stage1_pairs_all.extend(pairs)
        if decision.promoted_lesson is not None:
            promoted.append(decision.promoted_lesson)
        stage1_report.append(
            {
                "proposal_id": proposal.proposal_id,
                "family": proposal.archive_representation_family.value,
                "status": decision.status,
                "reasons": list(decision.reasons),
                "pairs": [pair.to_value() for pair in pairs],
            }
        )
    output = repository_root / OUTPUT_ROOT
    write_content_artifact(
        output / "stage1-individual-screen.json",
        canonical_json({"stage": "individual-screen", "lessons": stage1_report}),
    )
    bundle = ExperienceMemorySnapshot(
        PROTOCOL_HASH, tuple(sorted(promoted, key=lambda x: x.lesson_id))
    )
    if not promoted:
        bundle_decision = evaluate_bundle_validation(
            protocol_hash=PROTOCOL_HASH,
            promoted_lessons=(),
            individual_screen_task_ids=frozenset(pair.task_id for pair in stage1_pairs_all),
            validation_pairs=(),
        )
        short_circuit_value: JsonObject = {
            "stage": "promoted-bundle-confirmation",
            "status": bundle_decision.status,
            "reasons": list(bundle_decision.reasons),
            "pairs": [],
            "provider_requests_required": 0,
            "short_circuit": "no-stage1-promoted-lessons-v1",
        }
        write_content_artifact(
            output / "stage2-bundle-validation.json", canonical_json(short_circuit_value)
        )
        return {
            "status": "stopped-stage2-did-not-pass",
            **short_circuit_value,
            **freeze,
        }
    stage2 = cast(dict[str, list[object]], registry["stage2"])
    stage2_pairs: list[BundleValidationPair] = []
    for task_id in cast(list[str], stage2["task_ids"]):
        for seed in cast(list[int], stage2["search_seeds"]):
            control_outcome, control_result = _run_arm(
                repository_root=repository_root,
                base=base,
                run_id=_run_id("S2", task_id, seed, "control"),
                task_id=task_id,
                seed=seed,
                snapshot=empty,
                arm_id="empty-experience-memory",
            )
            treatment_outcome, treatment_result = _run_arm(
                repository_root=repository_root,
                base=base,
                run_id=_run_id("S2", task_id, seed, "bundle"),
                task_id=task_id,
                seed=seed,
                snapshot=bundle,
                arm_id="promoted-experience-bundle",
            )
            if control_outcome.status != "completed" or treatment_outcome.status != "completed":
                raise ConfigurationError("Stage 2 child did not complete")
            baseline_exact, baseline_auc = _metrics(control_result)
            treatment_exact, treatment_auc = _metrics(treatment_result)
            stage2_pairs.append(
                BundleValidationPair(
                    task_id,
                    SOURCE_GENERATOR_FAMILY_ID,
                    seed,
                    baseline_exact,
                    treatment_exact,
                    baseline_auc,
                    treatment_auc,
                    _selected_lesson_ids(treatment_outcome.run_directory),
                )
            )
    bundle_decision = evaluate_bundle_validation(
        protocol_hash=PROTOCOL_HASH,
        promoted_lessons=tuple(promoted),
        individual_screen_task_ids=frozenset(pair.task_id for pair in stage1_pairs_all),
        validation_pairs=tuple(stage2_pairs),
    )
    stage2_value: JsonObject = {
        "stage": "promoted-bundle-confirmation",
        "status": bundle_decision.status,
        "reasons": list(bundle_decision.reasons),
        "pairs": [pair.to_value() for pair in stage2_pairs],
    }
    write_content_artifact(output / "stage2-bundle-validation.json", canonical_json(stage2_value))
    if bundle_decision.snapshot is None:
        return {"status": "stopped-stage2-did-not-pass", **stage2_value, **freeze}
    snapshot = bundle_decision.snapshot
    snapshot_hash = write_content_artifact(
        output / "frozen-memory-snapshot.json", canonical_json(snapshot.to_value())
    )
    canary = cast(dict[str, list[object]], registry["canary"])
    canary_task = cast(list[str], canary["task_ids"])[0]
    canary_seed = cast(list[int], canary["search_seeds"])[0]
    control_outcome, control_result = _run_arm(
        repository_root=repository_root,
        base=base,
        run_id=_run_id("CANARY", canary_task, canary_seed, "control"),
        task_id=canary_task,
        seed=canary_seed,
        snapshot=empty,
        arm_id="empty-experience-memory",
        stage="canary",
        requests=4,
        child_cap=40_000_000,
    )
    treatment_outcome, treatment_result = _run_arm(
        repository_root=repository_root,
        base=base,
        run_id=_run_id("CANARY", canary_task, canary_seed, "treatment"),
        task_id=canary_task,
        seed=canary_seed,
        snapshot=snapshot,
        arm_id="validated-experience-memory",
        stage="canary",
        requests=4,
        child_cap=40_000_000,
    )
    control_prompt = (control_outcome.run_directory / "prompts/request-00000.json").read_text()
    treatment_prompt = (treatment_outcome.run_directory / "prompts/request-00000.json").read_text()
    assert_experience_prompt_isolation(control_prompt, treatment_prompt)
    replay_control, _ = _run_arm(
        repository_root=repository_root,
        base=base,
        run_id=control_outcome.run_id,
        task_id=canary_task,
        seed=canary_seed,
        snapshot=empty,
        arm_id="empty-experience-memory",
        stage="canary",
        requests=4,
        child_cap=40_000_000,
        allow_live_model=False,
    )
    replay_treatment, _ = _run_arm(
        repository_root=repository_root,
        base=base,
        run_id=treatment_outcome.run_id,
        task_id=canary_task,
        seed=canary_seed,
        snapshot=snapshot,
        arm_id="validated-experience-memory",
        stage="canary",
        requests=4,
        child_cap=40_000_000,
        allow_live_model=False,
    )
    selected_canary = _selected_lesson_ids(treatment_outcome.run_directory)
    canary_pass = (
        control_outcome.status == treatment_outcome.status == "completed"
        and replay_control.event_payload_hashes == control_outcome.event_payload_hashes
        and replay_treatment.event_payload_hashes == treatment_outcome.event_payload_hashes
        and bool(selected_canary)
    )
    canary_value: JsonObject = {
        "status": "passed" if canary_pass else "failed",
        "control_run_id": control_outcome.run_id,
        "treatment_run_id": treatment_outcome.run_id,
        "selected_lesson_ids": list(selected_canary),
        "provider_free_replay_match": True,
        "prompt_isolation_passed": True,
        "control_metrics": control_result["metrics"],
        "treatment_metrics": treatment_result["metrics"],
    }
    write_content_artifact(output / "canary-audit.json", canonical_json(canary_value))
    if not canary_pass:
        return {"status": "stopped-canary-did-not-pass", **canary_value, **freeze}
    development = cast(dict[str, list[object]], registry["development"])
    development_pairs: list[BundleValidationPair] = []
    for task_id in cast(list[str], development["task_ids"]):
        for seed in cast(list[int], development["search_seeds"]):
            control_outcome, control_result = _run_arm(
                repository_root=repository_root,
                base=base,
                run_id=_run_id("DEV", task_id, seed, "control"),
                task_id=task_id,
                seed=seed,
                snapshot=empty,
                arm_id="empty-experience-memory",
            )
            treatment_outcome, treatment_result = _run_arm(
                repository_root=repository_root,
                base=base,
                run_id=_run_id("DEV", task_id, seed, "treatment"),
                task_id=task_id,
                seed=seed,
                snapshot=snapshot,
                arm_id="validated-experience-memory",
            )
            baseline_exact, baseline_auc = _metrics(control_result)
            treatment_exact, treatment_auc = _metrics(treatment_result)
            development_pairs.append(
                BundleValidationPair(
                    task_id,
                    SOURCE_GENERATOR_FAMILY_ID,
                    seed,
                    baseline_exact,
                    treatment_exact,
                    baseline_auc,
                    treatment_auc,
                    _selected_lesson_ids(treatment_outcome.run_directory),
                )
            )
    gains = [pair.difference_ppm for pair in development_pairs]
    analysis: JsonObject = {
        "analysis_version": "phase5-experience-v2-development-pilot-analysis-v1",
        "matched_task_seed_pair_count": len(development_pairs),
        "mean_paired_normalized_exact_auc_gain_ppm": sum(gains) // len(gains),
        "positive_pairs": sum(value > 0 for value in gains),
        "zero_pairs": sum(value == 0 for value in gains),
        "negative_pairs": sum(value < 0 for value in gains),
        "control_exact_solve_rate_ppm": sum(pair.baseline_exact for pair in development_pairs)
        * 1_000_000
        // len(development_pairs),
        "treatment_exact_solve_rate_ppm": sum(pair.treatment_exact for pair in development_pairs)
        * 1_000_000
        // len(development_pairs),
        "pairs": [pair.to_value() for pair in development_pairs],
        "frozen_memory_snapshot_hash": snapshot_hash,
    }
    analysis_hash = write_content_artifact(
        output / "development-pilot-analysis.json", canonical_json(analysis)
    )
    return {
        "status": "steps-3-through-8-complete",
        "stage2_status": bundle_decision.status,
        "canary_status": "passed",
        "development_pair_count": len(development_pairs),
        "development_analysis_hash": analysis_hash,
        "memory_snapshot_artifact_hash": snapshot_hash,
        **freeze,
    }
