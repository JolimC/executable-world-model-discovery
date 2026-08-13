"""Provider-disabled Phase 5 development analysis and final freeze."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import cast

from world_model_search.dsl.primitives import PrimitiveRegistry, load_primitive_registry
from world_model_search.errors import ConfigurationError, PersistenceError
from world_model_search.evaluation.phase5_live import (
    CONDITION_C,
    CONDITION_D,
    load_phase5_live_experiment,
    replay_phase5_live_experiment,
)
from world_model_search.memory.types import load_memory_snapshot
from world_model_search.persistence.artifacts import read_text_artifact, write_content_artifact
from world_model_search.serialization import (
    JsonObject,
    JsonValue,
    canonical_json,
    sha256_json,
    sha256_text,
)


def _object(path: Path, label: str) -> JsonObject:
    try:
        value: object = json.loads(read_text_artifact(path))
    except json.JSONDecodeError as exc:
        raise PersistenceError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PersistenceError(f"{label} is not a JSON object")
    return cast(JsonObject, value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersistenceError(f"{label} must be an integer")
    return value


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[int(probability * (len(ordered) - 1))]


def _family_stratified_samples(
    values: dict[str, dict[str, list[float]]], *, seed: int, replicates: int
) -> list[float]:
    if not values or any(
        not tasks or any(not rows for rows in tasks.values()) for tasks in values.values()
    ):
        raise PersistenceError("family-stratified bootstrap has an empty task cluster")
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected: list[float] = []
        for family in sorted(values):
            tasks = values[family]
            task_ids = sorted(tasks)
            for _task in task_ids:
                selected.extend(tasks[task_ids[rng.randrange(len(task_ids))]])
        samples.append(mean(selected))
    return samples


def _comparison(
    pairs: list[JsonObject],
    *,
    field: str,
    seed: int,
    replicates: int,
    higher_is_better: bool,
) -> tuple[JsonObject, float]:
    clustered: dict[str, dict[str, list[float]]] = {}
    values: list[float] = []
    for pair in pairs:
        family = str(pair["target_family"])
        task_id = str(pair["task_id"])
        raw = pair[field]
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise PersistenceError(f"paired field is not numeric: {field}")
        value = float(raw)
        clustered.setdefault(family, {}).setdefault(task_id, []).append(value)
        values.append(value)
    point = mean(values)
    samples = _family_stratified_samples(clustered, seed=seed, replicates=replicates)
    centered = {
        family: {
            task_id: [value - point for value in task_values]
            for task_id, task_values in tasks.items()
        }
        for family, tasks in clustered.items()
    }
    null_samples = _family_stratified_samples(centered, seed=seed + 100_000, replicates=replicates)
    extreme = sum(abs(sample) >= abs(point) for sample in null_samples)
    p_value = (extreme + 1) / (replicates + 1)
    family_effects: JsonObject = {
        family: mean(value for rows in tasks.values() for value in rows)
        for family, tasks in sorted(clustered.items())
    }
    return (
        {
            "difference_direction": "condition-d-minus-condition-c",
            "higher_is_better": higher_is_better,
            "point_estimate": point,
            "family_stratified_task_cluster_bootstrap_95_interval": [
                _percentile(samples, 0.025),
                _percentile(samples, 0.975),
            ],
            "family_effects": family_effects,
            "task_count": sum(len(tasks) for tasks in clustered.values()),
            "family_count": len(clustered),
            "pair_count": len(values),
            "two_sided_p_value": p_value,
            "holm_adjusted_p_value": None,
            "holm_reject_0_05": False,
            "condition_d_superiority_established": False,
        },
        p_value,
    )


def _holm(p_values: dict[str, float]) -> dict[str, tuple[float, bool]]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = running
    rejected: dict[str, bool] = {}
    continue_rejecting = True
    for rank, name in enumerate(ordered):
        reject = continue_rejecting and p_values[name] <= 0.05 / (len(ordered) - rank)
        rejected[name] = reject
        continue_rejecting = reject
    return {name: (adjusted[name], rejected[name]) for name in p_values}


def _child_latency(root: Path, child: JsonObject, request_count: int) -> int:
    child_id = str(child["child_id"])
    total = 0
    for request_index in range(request_count):
        response = _object(
            root / "children" / child_id / "responses" / f"request-{request_index:05d}.json",
            "Phase 5 response artifact",
        )
        diagnostics = response.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise PersistenceError("Phase 5 response diagnostics are malformed")
        total += _integer(diagnostics.get("provider_latency_ns"), "provider latency")
    return total


def _usage(child: JsonObject, key: str) -> int:
    usage = child.get("usage")
    if not isinstance(usage, dict):
        raise PersistenceError("Phase 5 child usage is malformed")
    return _integer(usage.get(key), f"child usage {key}")


def _condition_totals(children: list[JsonObject]) -> JsonObject:
    totals: JsonObject = {}
    fields: dict[str, Callable[[JsonObject], int]] = {
        "request_attempts": lambda child: _integer(child.get("request_attempts"), "attempts"),
        "physical_provider_calls": lambda child: _integer(
            child.get("physical_provider_calls"), "provider calls"
        ),
        "valid_candidates": lambda child: _integer(child.get("valid_candidates"), "valid"),
        "invalid_candidates": lambda child: _integer(child.get("invalid_candidates"), "invalid"),
        "oracle_calls": lambda child: _integer(child.get("oracle_calls"), "oracle calls"),
        "published_rate_nano_usd": lambda child: _integer(
            child.get("published_rate_nano_usd"), "published cost"
        ),
        "input_tokens": lambda child: _usage(child, "input_tokens"),
        "output_tokens": lambda child: _usage(child, "output_tokens"),
        "total_tokens": lambda child: _usage(child, "total_tokens"),
    }
    for condition in (CONDITION_C, CONDITION_D):
        selected = [child for child in children if child.get("condition") == condition]
        score_sum = sum(_integer(child.get("best_score"), "best score") for child in selected)
        published = sum(fields["published_rate_nano_usd"](child) for child in selected)
        condition_totals: JsonObject = {
            key: sum(accessor(child) for child in selected) for key, accessor in fields.items()
        }
        condition_totals.update(
            {
                "child_count": len(selected),
                "exact_solved_children": sum(
                    child.get("exact_solved") is True for child in selected
                ),
                "best_score_sum": score_sum,
                "mean_best_score": score_sum / len(selected),
                "mean_best_score_per_oracle_call": score_sum
                / max(1, cast(int, condition_totals["oracle_calls"])),
                "best_score_sum_per_published_usd": score_sum / (published / 1_000_000_000),
                "retrieval_hits": sum(child.get("retrieval_hit") is True for child in selected),
            }
        )
        totals[condition] = condition_totals
    return totals


def finalize_phase5_development(
    *, repository_root: Path, registry_path: Path, freeze_root: Path
) -> JsonObject:
    """Analyze validated development artifacts and freeze an unchanged final mechanism."""

    experiment = load_phase5_live_experiment(registry_path)
    if experiment.stage != "development-pilot":
        raise ConfigurationError("Phase 5 final freeze requires a development-pilot registry")
    replay = replay_phase5_live_experiment(
        repository_root=repository_root, registry_path=registry_path
    )
    root = repository_root / experiment.output_root
    summary = _object(root / "summary.json", "Phase 5 development summary")
    children_value = _object(root / "children.json", "Phase 5 development children")
    raw_children = children_value.get("children")
    if not isinstance(raw_children, list) or any(not isinstance(row, dict) for row in raw_children):
        raise PersistenceError("Phase 5 development child index is malformed")
    children = cast(list[JsonObject], raw_children)
    indexed = {
        (str(child["task_id"]), _integer(child.get("seed"), "seed"), str(child["condition"])): child
        for child in children
    }
    pairs: list[JsonObject] = []
    for task_id in experiment.task_ids:
        for seed in experiment.search_seeds:
            off = indexed[(task_id, seed, CONDITION_C)]
            on = indexed[(task_id, seed, CONDITION_D)]
            off_latency = _child_latency(root, off, experiment.requests_per_child)
            on_latency = _child_latency(root, on, experiment.requests_per_child)
            pairs.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "target_family": str(off["target_family"]),
                    "exact_solve_difference": int(on["exact_solved"] is True)
                    - int(off["exact_solved"] is True),
                    "best_score_difference": _integer(on.get("best_score"), "D score")
                    - _integer(off.get("best_score"), "C score"),
                    "input_token_difference": _usage(on, "input_tokens")
                    - _usage(off, "input_tokens"),
                    "output_token_difference": _usage(on, "output_tokens")
                    - _usage(off, "output_tokens"),
                    "total_token_difference": _usage(on, "total_tokens")
                    - _usage(off, "total_tokens"),
                    "published_rate_difference_nano_usd": _integer(
                        on.get("published_rate_nano_usd"), "D cost"
                    )
                    - _integer(off.get("published_rate_nano_usd"), "C cost"),
                    "provider_latency_difference_ns": on_latency - off_latency,
                }
            )
    comparison_fields = {
        "matched_search_quality_best_score": ("best_score_difference", True),
        "model_total_tokens": ("total_token_difference", False),
        "published_rate_cost_nano_usd": ("published_rate_difference_nano_usd", False),
        "provider_latency_ns": ("provider_latency_difference_ns", False),
    }
    comparisons: JsonObject = {}
    p_values: dict[str, float] = {}
    for offset, (name, (field, higher_is_better)) in enumerate(comparison_fields.items()):
        result, p_value = _comparison(
            pairs,
            field=field,
            seed=56_001 + offset,
            replicates=10_000,
            higher_is_better=higher_is_better,
        )
        comparisons[name] = result
        p_values[name] = p_value
    for name, (adjusted, rejected) in _holm(p_values).items():
        stored_result = comparisons[name]
        if not isinstance(stored_result, dict):
            raise AssertionError("Phase 5 comparison result is malformed")
        stored_result["holm_adjusted_p_value"] = adjusted
        stored_result["holm_reject_0_05"] = rejected
        point = stored_result["point_estimate"]
        higher = stored_result["higher_is_better"]
        assert isinstance(point, int | float) and isinstance(higher, bool)
        stored_result["condition_d_superiority_established"] = rejected and (
            point > 0 if higher else point < 0
        )
    exact_result, _ = _comparison(
        pairs,
        field="exact_solve_difference",
        seed=66_001,
        replicates=10_000,
        higher_is_better=True,
    )
    exact_result.pop("two_sided_p_value")
    exact_result.pop("holm_adjusted_p_value")
    exact_result.pop("holm_reject_0_05")
    exact_result.pop("condition_d_superiority_established")
    source_analysis = _object(root / "analysis.json", "Phase 5 source analysis")
    condition_totals = _condition_totals(children)
    cash = cast(
        dict[str, JsonValue], cast(dict[str, JsonValue], summary["ledger_status"])["cash_budget"]
    )
    final_analysis: JsonObject = {
        "analysis_version": "phase5-live-development-final-analysis-v1",
        "source_experiment_hash": experiment.source_hash,
        "source_summary_hash": sha256_json(summary),
        "source_analysis_hash": sha256_text(read_text_artifact(root / "analysis.json")),
        "replay_summary_hash": replay["summary_hash"],
        "data_role": "development-pilot",
        "confirmatory": False,
        "h3_confirmed": False,
        "h3_development_supported": False,
        "primary_endpoint": "net-held-out-two-part-code-length-gain",
        "primary_endpoint_status": "failed-correctness-no-comparable-exact-pairs",
        "primary_accounting_floor_bits": source_analysis["aggregate_net_two_part_gain_bits"],
        "primary_accounting_floor_interpretable_as_transfer_gain": False,
        "library_definition_cost_bits_charged_once": source_analysis[
            "library_definition_cost_bits_charged_once"
        ],
        "exact_solve_comparison": exact_result,
        "paired_rows": cast(list[JsonValue], pairs),
        "condition_totals": condition_totals,
        "secondary_comparisons": comparisons,
        "multiplicity": "holm-two-sided-across-four-predeclared-paired-secondary-comparisons-v1",
        "retrieval_precision": source_analysis["retrieval_precision"],
        "primitive_transfer_matrix": source_analysis["transfer_matrix"],
        "primitive_transfer_gain_status": "not-estimable-no-comparable-exact-pairs",
        "cash_accounting": {
            "latest_reconciled_cash_nano_usd": cash["reconciled_cash_nano_usd"],
            "unreconciled_published_exposure_nano_usd": cash["unreconciled_actual_nano_usd"],
            "cash_upper_bound_nano_usd": cash["cash_upper_bound_nano_usd"],
            "active_reserved_nano_usd": cash["active_reserved_nano_usd"],
            "uncertain_nano_usd": cash["uncertain_nano_usd"],
            "arm_allocable_actual_cash": False,
        },
        "sealed_test_accesses": summary["sealed_test_accesses"],
        "conclusion": (
            "Condition D did not improve exact solving and had a negative mean "
            "best-score difference. "
            "The live development pilot does not support H3; no mechanism was refit."
        ),
    }
    final_analysis_hash = write_content_artifact(
        root / "analysis-final.json", canonical_json(final_analysis)
    )

    analysis_plan: JsonObject = {
        "analysis_plan_version": "phase5-final-sealed-analysis-plan-v1",
        "frozen_after_development_experiment_hash": experiment.source_hash,
        "transfer_registry_hash": experiment.transfer_registry_hash,
        "data_role": "confirmatory-test",
        "primary_endpoint": "net-held-out-two-part-code-length-gain",
        "correctness_precedence": "all-compared-programs-must-be-exact",
        "missing_exact_pair_rule": "gate-fails-without-imputation",
        "library_definition_cost": "charge-complete-frozen-library-once",
        "secondary_endpoints": [
            "matched-search-quality-best-score",
            "retrieval-precision",
            "primitive-transfer-gain-by-target-family",
            "provider-runtime",
            "model-token-usage",
            "published-rate-equivalent-cost",
            "reconciled-actual-cash",
        ],
        "pairing": "exact-task-id-and-search-seed-v1",
        "uncertainty": "family-stratified-task-cluster-bootstrap-v1",
        "bootstrap_seed": 56_001,
        "bootstrap_replicates": 10_000,
        "confidence_level": 95,
        "multiplicity": "holm-two-sided-across-four-paired-secondary-comparisons-v1",
        "paired_secondary_comparisons": [
            "matched-search-quality-best-score",
            "model-total-tokens",
            "published-rate-cost",
            "provider-latency",
        ],
        "promotion_rule": "strictly-positive-net-gain-and-exact-correctness",
        "memory_or_primitive_refit_after_development": False,
        "sealed_test_authorized": False,
    }
    analysis_plan_hash = sha256_json(analysis_plan)
    source_snapshot = load_memory_snapshot(repository_root / experiment.memory_snapshot)
    source_registry = load_primitive_registry(repository_root / experiment.primitive_registry)
    final_registry = PrimitiveRegistry(
        split_registry_hash=source_registry.split_registry_hash,
        analysis_plan_hash=analysis_plan_hash,
        source_evidence_ids=source_registry.source_evidence_ids,
        definitions=source_registry.definitions,
    )
    snapshot_value: JsonObject = {
        **source_snapshot.to_value(),
        "snapshot_hash": source_snapshot.snapshot_hash,
    }
    registry_value: JsonObject = {
        **final_registry.to_value(),
        "registry_hash": final_registry.registry_hash,
    }
    destination = repository_root / freeze_root
    plan_artifact_hash = write_content_artifact(
        destination / "analysis-plan.json", canonical_json(analysis_plan)
    )
    snapshot_artifact_hash = write_content_artifact(
        destination / "memory-snapshot.json", canonical_json(snapshot_value)
    )
    registry_artifact_hash = write_content_artifact(
        destination / "primitive-registry.json", canonical_json(registry_value)
    )
    manifest: JsonObject = {
        "freeze_manifest_version": "phase5-final-freeze-manifest-v1",
        "source_experiment_hash": experiment.source_hash,
        "source_summary_hash": sha256_json(summary),
        "provider_disabled_replay_hash": replay["summary_hash"],
        "development_analysis_hash": final_analysis_hash,
        "analysis_plan_hash": analysis_plan_hash,
        "analysis_plan_artifact_hash": plan_artifact_hash,
        "memory_snapshot_hash": source_snapshot.snapshot_hash,
        "memory_snapshot_artifact_hash": snapshot_artifact_hash,
        "primitive_registry_hash": final_registry.registry_hash,
        "primitive_registry_artifact_hash": registry_artifact_hash,
        "memory_content_unchanged_after_development": True,
        "primitive_definitions_unchanged_after_development": True,
        "development_refit_performed": False,
        "sealed_test_accesses": 0,
        "sealed_test_authorized": False,
        "scientific_disposition": "development-does-not-support-h3",
    }
    manifest_hash = write_content_artifact(destination / "manifest.json", canonical_json(manifest))
    return {
        **manifest,
        "freeze_manifest_hash": manifest_hash,
        "status": "frozen-provider-disabled",
        "provider_calls": 0,
        "oracle_accesses": 0,
    }
