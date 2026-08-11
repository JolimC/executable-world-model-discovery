"""Strict Phase 4 A/B/C registries, no-cost forecasts, and recorded analysis."""

from __future__ import annotations

import csv
import io
import json
import random
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any, cast

import yaml

from world_model_search.config import AppConfig, load_config
from world_model_search.domain.types import SplitLabel
from world_model_search.errors import ConfigurationError, PersistenceError
from world_model_search.model.policy import PricePolicy, load_price_policy
from world_model_search.persistence.artifacts import read_text_artifact, write_content_artifact
from world_model_search.search.loop import load_manifest
from world_model_search.search.phase4 import (
    Phase4Authority,
    resume_phase4_run,
    start_phase4_run,
)
from world_model_search.search.phase4_types import Phase4Condition
from world_model_search.serialization import (
    JsonObject,
    JsonValue,
    canonical_json,
    parse_json_object,
    sha256_json,
    sha256_text,
)

PHASE4_EXPERIMENT_SCHEMA_VERSION = 2
CONDITIONS = (
    Phase4Condition.DIRECT,
    Phase4Condition.INCUMBENT,
    Phase4Condition.DIVERSE,
)


def _record_counter(counters: object, name: str) -> int:
    value = counters.get(name) if isinstance(counters, dict) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersistenceError(f"Phase 4 child counter {name} is malformed")
    return value


@dataclass(frozen=True, slots=True)
class Phase4ExperimentRegistry:
    experiment_id: str
    status: str
    base_config: Path
    output_root: Path
    child_runs_root: Path
    report_root: Path
    prerequisite_canary: JsonObject | None
    split: SplitLabel
    task_ids: tuple[str, ...]
    search_seeds: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_replicates: int
    raw: JsonObject

    @property
    def content_hash(self) -> str:
        return sha256_json(self.raw)


def is_phase4_registry(path: Path) -> bool:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(raw, dict) and raw.get("experiment_schema_version") == 2


def _mapping(value: object, expected: set[str], location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping")
    if set(value) != expected:
        raise ConfigurationError(f"{location} has missing or unknown fields")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{location} must be a nonempty string")
    return value


def _integer(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{location} must be an integer >= {minimum}")
    return value


def _path(value: object, location: str) -> Path:
    path = Path(_string(value, location))
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ConfigurationError(f"{location} must be a specific repository-relative path")
    return path


def load_phase4_experiment_registry(path: Path) -> Phase4ExperimentRegistry:
    """Validate all matched conditions and predeclared inference without writing."""

    try:
        loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read Phase 4 experiment registry: {exc}") from exc
    root = _mapping(
        loaded,
        {
            "experiment_schema_version",
            "experiment_id",
            "status",
            "base_config",
            "output_root",
            "child_runs_root",
            "report_root",
            "prerequisite_canary",
            "conditions",
            "condition_execution",
            "task_selection",
            "search_seeds",
            "matched_contract",
            "hypotheses",
            "analysis",
            "spending",
            "phase_boundary",
        },
        "Phase 4 experiment",
    )
    if root["experiment_schema_version"] != PHASE4_EXPERIMENT_SCHEMA_VERSION:
        raise ConfigurationError("unsupported Phase 4 experiment schema")
    status = _string(root["status"], "status")
    if status not in {"fake-lifecycle", "development-pilot"}:
        raise ConfigurationError("Phase 4 executable registry status is invalid")
    expected_conditions = [condition.value for condition in CONDITIONS]
    if root["conditions"] != expected_conditions:
        raise ConfigurationError("Phase 4 conditions must be frozen in A/B/C order")
    prerequisite_canary: JsonObject | None = None
    if status == "fake-lifecycle":
        if root["prerequisite_canary"] is not None:
            raise ConfigurationError("fake Phase 4 registries cannot require a live canary")
    else:
        prerequisite = _mapping(
            root["prerequisite_canary"],
            {
                "run_directory",
                "run_id",
                "configuration_hash",
                "deterministic_summary_hash",
                "minimum_valid_items",
            },
            "prerequisite_canary",
        )
        _path(prerequisite["run_directory"], "prerequisite_canary.run_directory")
        _string(prerequisite["run_id"], "prerequisite_canary.run_id")
        for field in ("configuration_hash", "deterministic_summary_hash"):
            digest = _string(prerequisite[field], f"prerequisite_canary.{field}")
            if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
                raise ConfigurationError(f"prerequisite_canary.{field} must be a SHA-256")
        _integer(
            prerequisite["minimum_valid_items"],
            "prerequisite_canary.minimum_valid_items",
            minimum=1,
        )
        prerequisite_canary = cast(JsonObject, json.loads(canonical_json(prerequisite)))
    execution = _mapping(
        root["condition_execution"], {"policy", "base_order"}, "condition_execution"
    )
    if execution != {
        "policy": "deterministic-rotating-block-by-task-seed-v1",
        "base_order": expected_conditions,
    }:
        raise ConfigurationError("Phase 4 condition interleaving policy is not frozen")
    task_selection = _mapping(
        root["task_selection"], {"split", "policy", "task_ids"}, "task_selection"
    )
    try:
        split = SplitLabel(_string(task_selection["split"], "task_selection.split"))
    except ValueError as exc:
        raise ConfigurationError("Phase 4 task split is invalid") from exc
    expected_split = SplitLabel.TRAINING if status == "fake-lifecycle" else SplitLabel.DEVELOPMENT
    if split is not expected_split or task_selection["policy"] != "opaque-public-id-fixed-v1":
        raise ConfigurationError("Phase 4 task authority/policy does not match the profile")
    task_ids_raw = task_selection["task_ids"]
    expected_tasks = 1 if status == "fake-lifecycle" else 10
    if (
        not isinstance(task_ids_raw, list)
        or len(task_ids_raw) != expected_tasks
        or len(set(cast(list[object], task_ids_raw))) != expected_tasks
        or not all(
            isinstance(item, str) and len(item) == 24 and not set(item) - set("0123456789abcdef")
            for item in task_ids_raw
        )
    ):
        raise ConfigurationError(f"Phase 4 {status} requires {expected_tasks} opaque task IDs")
    seeds_raw = root["search_seeds"]
    expected_seeds = 1 if status == "fake-lifecycle" else 2
    if (
        not isinstance(seeds_raw, list)
        or len(seeds_raw) != expected_seeds
        or len(set(cast(list[object], seeds_raw))) != expected_seeds
        or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds_raw)
    ):
        raise ConfigurationError(f"Phase 4 {status} requires {expected_seeds} unique seeds")
    base_config_path = _path(root["base_config"], "base_config")
    base = load_config(base_config_path)
    if base.schema_version != 4 or base.phase4_budget is None or base.model is None:
        raise ConfigurationError("Phase 4 registry base configuration is incomplete")
    contract = _mapping(
        root["matched_contract"],
        {
            "model",
            "endpoint",
            "service_tier",
            "reasoning_effort",
            "batch_size",
            "model_request_cap",
            "input_token_cap",
            "output_token_cap",
            "total_token_cap",
            "proposal_item_cap",
            "oracle_call_cap",
            "role",
            "response_mode",
            "initialization",
            "continue_after_first_exact",
        },
        "matched_contract",
    )
    expected_contract: dict[str, object] = {
        "model": base.model.resolved_model,
        "endpoint": base.model.endpoint,
        "service_tier": base.model.service_tier,
        "reasoning_effort": base.model.reasoning_effort,
        "batch_size": base.proposer.batch_size,
        "model_request_cap": base.phase4_budget.model_request_cap,
        "input_token_cap": base.phase4_budget.input_token_cap,
        "output_token_cap": base.phase4_budget.output_token_cap,
        "total_token_cap": base.phase4_budget.total_token_cap,
        "proposal_item_cap": base.phase4_budget.proposal_item_cap,
        "oracle_call_cap": base.phase4_budget.oracle_call_cap,
        "role": "exploit",
        "response_mode": "score-only",
        "initialization": "seven-shared-charged-public-candidates-v1",
        "continue_after_first_exact": True,
    }
    if contract != expected_contract:
        raise ConfigurationError("Phase 4 A/B/C matched contract differs from its base config")
    hypotheses = _mapping(root["hypotheses"], {"H1", "H2", "secondary"}, "hypotheses")
    if hypotheses != {
        "H1": "single-incumbent-v1-minus-direct-llm-v1",
        "H2": "uniform-diverse-archive-v1-minus-single-incumbent-v1",
        "secondary": "uniform-diverse-archive-v1-minus-direct-llm-v1",
    }:
        raise ConfigurationError("Phase 4 hypothesis contrasts are not predeclared")
    analysis = _mapping(
        root["analysis"],
        {
            "primary_endpoint",
            "pairing",
            "primary_bootstrap",
            "sensitivity_bootstrap",
            "bootstrap_seed",
            "bootstrap_replicates",
            "confidence_level",
            "multiplicity",
        },
        "analysis",
    )
    if (
        analysis["primary_endpoint"] != "normalized-exact-solve-auc-v1"
        or analysis["pairing"] != "exact-task-id-and-search-seed-v1"
        or analysis["primary_bootstrap"] != "task-clustered-v1"
        or analysis["sensitivity_bootstrap"] != "task-seed-pair-v1"
        or analysis["confidence_level"] != 95
        or analysis["multiplicity"] != "holm-two-sided-two-hypotheses-v1"
    ):
        raise ConfigurationError("Phase 4 analysis declaration differs from the frozen contract")
    replicates = _integer(analysis["bootstrap_replicates"], "bootstrap_replicates", minimum=1000)
    bootstrap_seed = _integer(analysis["bootstrap_seed"], "bootstrap_seed")
    spending = _mapping(
        root["spending"],
        {"stage", "price_policy", "child_nano_usd_cap", "aggregate_nano_usd_cap"},
        "spending",
    )
    if base.phase4_policy is None:
        raise ConfigurationError("Phase 4 base config has no spending policy")
    policy = load_price_policy(base.phase4_policy.price_policy)
    expected_stage = "fake" if status == "fake-lifecycle" else "pilot"
    if (
        spending["stage"] != expected_stage
        or _path(spending["price_policy"], "spending.price_policy")
        != base.phase4_policy.price_policy
        or spending["child_nano_usd_cap"] != base.phase4_budget.child_nano_usd_cap
        or spending["aggregate_nano_usd_cap"] != policy.stage_cap(expected_stage)
    ):
        raise ConfigurationError("Phase 4 registry spending contract is inconsistent")
    if root["phase_boundary"] != "phase4-no-memory-no-interestingness-no-active-query-v1":
        raise ConfigurationError("Phase 4 registry crosses the declared phase boundary")
    raw = cast(JsonObject, json.loads(canonical_json(root)))
    return Phase4ExperimentRegistry(
        experiment_id=_string(root["experiment_id"], "experiment_id"),
        status=status,
        base_config=base_config_path,
        output_root=_path(root["output_root"], "output_root"),
        child_runs_root=_path(root["child_runs_root"], "child_runs_root"),
        report_root=_path(root["report_root"], "report_root"),
        prerequisite_canary=prerequisite_canary,
        split=split,
        task_ids=tuple(cast(list[str], task_ids_raw)),
        search_seeds=tuple(cast(list[int], seeds_raw)),
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=replicates,
        raw=raw,
    )


def _ledger_balance(path: Path, policy: PricePolicy) -> JsonObject:
    if not path.is_file():
        return {
            "state": "not-created",
            "opening_nano_usd": policy.opening_balance_nano_usd,
            "actual_nano_usd": 0,
            "uncertain_nano_usd": 0,
            "active_reserved_nano_usd": 0,
            "committed_nano_usd": policy.opening_balance_nano_usd,
        }
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
        row = connection.execute(
            """SELECT COALESCE(SUM(actual_nano_usd),0) actual,
                      COALESCE(SUM(uncertain_nano_usd),0) uncertain,
                      COALESCE(SUM(CASE WHEN state='active' THEN reserved_nano_usd ELSE 0 END),0)
                          reserved
               FROM reservation"""
        ).fetchone()
        connection.close()
    except sqlite3.Error as exc:
        raise PersistenceError("cannot read the project cost ledger for dry-run") from exc
    if metadata.get("policy_hash") != policy.content_hash or row is None:
        raise PersistenceError("project ledger policy hash is missing or inconsistent")
    actual = int(row["actual"])
    uncertain = int(row["uncertain"])
    reserved = int(row["reserved"])
    return {
        "state": "existing-verified",
        "opening_nano_usd": policy.opening_balance_nano_usd,
        "actual_nano_usd": actual,
        "uncertain_nano_usd": uncertain,
        "active_reserved_nano_usd": reserved,
        "committed_nano_usd": policy.opening_balance_nano_usd + actual + uncertain + reserved,
    }


def _canary_prerequisite_status(
    *, repository_root: Path, prerequisite: JsonObject
) -> tuple[JsonObject, list[str]]:
    run_directory = repository_root / Path(cast(str, prerequisite["run_directory"]))
    try:
        manifest = load_manifest(run_directory)
        results = parse_json_object(read_text_artifact(run_directory / "results.json"))
    except (OSError, ConfigurationError, PersistenceError):
        return (
            {
                "required": True,
                "status": "missing-or-unreadable",
                "run_id": prerequisite["run_id"],
            },
            ["the exact successful training-canary record is missing or unreadable"],
        )
    budget = results.get("budget")
    counters = budget.get("counters") if isinstance(budget, dict) else None
    metrics = results.get("metrics")
    request_states = metrics.get("request_states") if isinstance(metrics, dict) else None
    valid_items_value = counters.get("valid_items") if isinstance(counters, dict) else None
    valid_items = (
        valid_items_value
        if isinstance(valid_items_value, int) and not isinstance(valid_items_value, bool)
        else -1
    )
    completed_requests_value = (
        request_states.get("completed") if isinstance(request_states, dict) else None
    )
    completed_requests = (
        completed_requests_value
        if isinstance(completed_requests_value, int)
        and not isinstance(completed_requests_value, bool)
        else 0
    )
    checks = {
        "run_id": manifest.get("run_id") == prerequisite["run_id"],
        "configuration_hash": manifest.get("configuration_hash")
        == prerequisite["configuration_hash"],
        "deterministic_summary_hash": results.get("deterministic_summary_hash")
        == prerequisite["deterministic_summary_hash"],
        "completed_live_run": results.get("status") == "completed"
        and results.get("evidence_class") == "live",
        "completed_request": completed_requests >= 1,
        "valid_items": valid_items >= cast(int, prerequisite["minimum_valid_items"]),
    }
    passed = all(checks.values())
    status: JsonObject = {
        "required": True,
        "status": "passed" if passed else "failed-verification",
        "run_id": prerequisite["run_id"],
        "checks": cast(JsonObject, checks),
        "observed_valid_items": valid_items,
        "observed_completed_requests": completed_requests,
    }
    blockers = [] if passed else ["the exact training-canary prerequisite failed verification"]
    return status, blockers


def phase4_dry_run(*, repository_root: Path, registry: Phase4ExperimentRegistry) -> JsonObject:
    """Forecast a registry with no hidden task access, output write, key read, or provider call."""

    base = load_config(registry.base_config)
    if base.model is None or base.phase4_budget is None or base.phase4_policy is None:
        raise ConfigurationError("Phase 4 dry-run base configuration is incomplete")
    policy = load_price_policy(repository_root / base.phase4_policy.price_policy)
    children = len(CONDITIONS) * len(registry.task_ids) * len(registry.search_seeds)
    child_exposure = base.phase4_budget.child_nano_usd_cap
    planned_exposure = children * child_exposure
    logical_calls_per_child = (
        base.phase4_budget.proposal_item_cap + base.proposer.batch_size - 1
    ) // base.proposer.batch_size
    stage_cap = policy.stage_cap(base.phase4_policy.stage)
    blockers: list[str] = []
    if planned_exposure > stage_cap:
        blockers.append("sum of per-child ceilings exceeds the experiment-stage ceiling")
    prerequisite_status: JsonObject = {"required": False, "status": "not-required"}
    if registry.status == "development-pilot":
        if registry.prerequisite_canary is None:
            raise AssertionError("validated development pilot has no canary prerequisite")
        prerequisite_status, prerequisite_blockers = _canary_prerequisite_status(
            repository_root=repository_root,
            prerequisite=registry.prerequisite_canary,
        )
        blockers.extend(prerequisite_blockers)
    return {
        "dry_run_schema_version": 2,
        "network_calls": 0,
        "provider_calls": 0,
        "hidden_oracle_accesses": 0,
        "registry_id": registry.experiment_id,
        "registry_hash": registry.content_hash,
        "status": "blocked" if blockers else "ready-no-cost",
        "blockers": cast(list[JsonValue], blockers),
        "planned_split": registry.split.value,
        "opaque_task_ids": list(registry.task_ids),
        "search_seeds": list(registry.search_seeds),
        "condition_execution": registry.raw["condition_execution"],
        "prerequisite_canary": prerequisite_status,
        "children": children,
        "planned_work": {
            "logical_model_calls_per_child": logical_calls_per_child,
            "logical_model_calls_all_children": children * logical_calls_per_child,
            "maximum_physical_attempts_all_children": children
            * base.phase4_budget.model_request_cap,
            "proposal_items_all_children": children * base.phase4_budget.proposal_item_cap,
            "oracle_calls_all_children": children * base.phase4_budget.oracle_call_cap,
        },
        "resolved_model_contract": {
            "backend": base.model.backend_id,
            "provider": base.model.provider_id,
            "model": base.model.resolved_model,
            "endpoint": base.model.endpoint,
            "service_tier": base.model.service_tier,
            "settings": base.model.request_settings(),
        },
        "joint_child_caps": {
            "model_requests": base.phase4_budget.model_request_cap,
            "input_tokens": base.phase4_budget.input_token_cap,
            "output_tokens": base.phase4_budget.output_token_cap,
            "total_tokens": base.phase4_budget.total_token_cap,
            "proposal_items": base.phase4_budget.proposal_item_cap,
            "oracle_calls": base.phase4_budget.oracle_call_cap,
            "child_nano_usd": base.phase4_budget.child_nano_usd_cap,
        },
        "worst_case_nano_usd": {
            "one_request": policy.request_cap_nano_usd,
            "one_child": child_exposure,
            "all_children": planned_exposure,
            "stage": stage_cap,
            "phase4": policy.phase4_cap_nano_usd,
            "project": policy.project_lifetime_cap_nano_usd,
            "prior_phase_0_3_spend": policy.prior_phase_0_3_spend_nano_usd,
        },
        "price_policy_hash": policy.content_hash,
        "ledger": _ledger_balance(repository_root / base.phase4_policy.ledger, policy),
        "evidence_class": "forecast-only-no-scientific-result",
    }


def _child_config(
    base: AppConfig,
    registry: Phase4ExperimentRegistry,
    *,
    task_id: str,
    seed: int,
    condition: Phase4Condition,
) -> AppConfig:
    if base.cache is None:
        raise AssertionError("validated Phase 4 base has no cache")
    child_namespace = (
        f"{base.cache.namespace}-{registry.experiment_id}-{task_id}-{seed}-{condition.value}"
    )
    return replace(
        base,
        run=replace(
            base.run,
            root=registry.child_runs_root,
            task_id=task_id,
            split=registry.split,
            seed=seed,
            condition_id=condition.value,
        ),
        cache=replace(base.cache, namespace=child_namespace),
    )


def _condition_order(task_id: str, seed: int) -> tuple[Phase4Condition, ...]:
    offset = int(sha256_text(f"phase4-condition-block-v1\0{task_id}\0{seed}")[:8], 16) % 3
    return CONDITIONS[offset:] + CONDITIONS[:offset]


def _child_id(task_id: str, seed: int, condition: Phase4Condition) -> str:
    label = {
        Phase4Condition.DIRECT: "A",
        Phase4Condition.INCUMBENT: "B",
        Phase4Condition.DIVERSE: "C",
    }[condition]
    return f"p4-{task_id}-s{seed}-{label}"


def run_phase4_experiment(
    *,
    repository_root: Path,
    registry_path: Path,
    allow_live_model: bool,
) -> JsonObject:
    registry = load_phase4_experiment_registry(registry_path)
    base = load_config(registry.base_config)
    output_root = repository_root / registry.output_root
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        existing_manifest = parse_json_object(
            read_text_artifact(output_root / "experiment-manifest.json")
        )
        if existing_manifest.get("registry_hash") != registry.content_hash:
            raise PersistenceError("completed Phase 4 experiment registry hash differs")
        return parse_json_object(read_text_artifact(summary_path))
    rows: list[JsonObject] = []
    execution_order: list[JsonObject] = []
    for task_id in registry.task_ids:
        for seed in registry.search_seeds:
            for condition in _condition_order(task_id, seed):
                child = _child_config(
                    base, registry, task_id=task_id, seed=seed, condition=condition
                )
                run_id = _child_id(task_id, seed, condition)
                execution_order.append(
                    {"task_id": task_id, "search_seed": seed, "condition": condition.value}
                )
                run_directory = repository_root / registry.child_runs_root / run_id
                if run_directory.exists():
                    child_manifest = load_manifest(run_directory)
                    if child_manifest.get("configuration_hash") != child.content_hash:
                        raise PersistenceError("foreign or differently configured Phase 4 child")
                    results_path = run_directory / "results.json"
                    if not results_path.is_file():
                        outcome = resume_phase4_run(
                            repository_root=repository_root,
                            run_directory=run_directory,
                            config=child,
                            manifest=child_manifest,
                            interrupt_after=None,
                            allow_live_model=allow_live_model,
                        )
                        if outcome.status not in {
                            "completed",
                            "cost-cap-exhausted",
                            "usage-uncertain",
                            "failed",
                        }:
                            raise PersistenceError(
                                "resumed Phase 4 experiment child is not terminal"
                            )
                    result = parse_json_object(read_text_artifact(results_path))
                else:
                    outcome = start_phase4_run(
                        repository_root=repository_root,
                        config=child,
                        config_source=str(registry.base_config),
                        run_id=run_id,
                        interrupt_after=None,
                        allow_live_model=allow_live_model,
                        authority=Phase4Authority.ordinary(),
                    )
                    if outcome.status not in {
                        "completed",
                        "cost-cap-exhausted",
                        "usage-uncertain",
                        "failed",
                    }:
                        raise PersistenceError("Phase 4 experiment child is not terminal")
                    result = parse_json_object(read_text_artifact(run_directory / "results.json"))
                metrics = result.get("metrics")
                budget = result.get("budget")
                if not isinstance(metrics, dict) or not isinstance(budget, dict):
                    raise PersistenceError("Phase 4 child results are malformed")
                rows.append(
                    {
                        "task_id": task_id,
                        "search_seed": seed,
                        "condition_id": condition.value,
                        "run_id": run_id,
                        "configuration_hash": child.content_hash,
                        "evidence_class": result.get("evidence_class"),
                        "metrics": metrics,
                        "budget": budget,
                        "deterministic_summary_hash": result.get("deterministic_summary_hash"),
                    }
                )
    analysis = analyze_phase4_rows(
        rows,
        bootstrap_seed=registry.bootstrap_seed,
        bootstrap_replicates=registry.bootstrap_replicates,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    manifest: JsonObject = {
        "experiment_manifest_version": 2,
        "registry_hash": registry.content_hash,
        "base_configuration_hash": base.content_hash,
        "execution_order": cast(list[JsonValue], execution_order),
        "mechanism_differences": {
            "A_vs_B": "direct-no-parent versus incumbent-parent-score-and-retention",
            "B_vs_C": "single-incumbent versus uniform-branch diverse archive",
            "shared": "model/settings/role/batch/caps/initialization/oracle/stopping",
        },
    }
    write_content_artifact(output_root / "experiment-manifest.json", canonical_json(manifest))
    analysis_root = output_root / "analysis"
    analysis_root.mkdir()
    raw_text = canonical_json({"rows": rows})
    files: dict[str, str] = {}
    files["raw-rows.json"] = write_content_artifact(analysis_root / "raw-rows.json", raw_text)
    files["paired-analysis.json"] = write_content_artifact(
        analysis_root / "paired-analysis.json", canonical_json(analysis)
    )
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(["task_id", "search_seed", "condition_id", "normalized_exact_auc", "run_id"])
    for row in rows:
        csv_metrics = cast(dict[str, object], row["metrics"])
        writer.writerow(
            [
                row["task_id"],
                row["search_seed"],
                row["condition_id"],
                csv_metrics["normalized_exact_auc"],
                row["run_id"],
            ]
        )
    files["raw-rows.csv"] = write_content_artifact(
        analysis_root / "raw-rows.csv", csv_buffer.getvalue().rstrip("\n")
    )
    analysis_manifest: JsonObject = {
        "analysis_manifest_version": 1,
        "source": "completed-child-records-only",
        "files": cast(JsonObject, files),
    }
    analysis_manifest_hash = write_content_artifact(
        analysis_root / "manifest.json", canonical_json(analysis_manifest)
    )
    evidence = {row.get("evidence_class") for row in rows}
    summary: JsonObject = {
        "experiment_summary_version": 2,
        "experiment_id": registry.experiment_id,
        "registry_hash": registry.content_hash,
        "evidence_class": "fake" if evidence == {"fake"} else "live",
        "scientific_gate": (
            "blocked-fake-evidence-only" if evidence == {"fake"} else "pilot-not-confirmatory"
        ),
        "child_count": len(rows),
        "analysis": analysis,
        "analysis_manifest_hash": analysis_manifest_hash,
        "locked_test_disposition": "not-frozen-pending-pilot-power-and-cost-review",
    }
    summary["deterministic_summary_hash"] = sha256_json(summary)
    write_content_artifact(summary_path, canonical_json(summary))
    _write_presentable_artifact(
        repository_root=repository_root,
        registry=registry,
        base=base,
        rows=rows,
        analysis=analysis,
        summary=summary,
        execution_order=execution_order,
    )
    return summary


def _write_presentable_artifact(
    *,
    repository_root: Path,
    registry: Phase4ExperimentRegistry,
    base: AppConfig,
    rows: list[JsonObject],
    analysis: JsonObject,
    summary: JsonObject,
    execution_order: list[JsonObject],
) -> None:
    """Assemble the Phase 4 research artifact entirely from frozen child records."""

    if base.phase4_budget is None or base.phase4_policy is None or base.model is None:
        raise AssertionError("Phase 4 presentable artifact requires complete base settings")
    report_root = repository_root / registry.report_root
    report_root.mkdir(parents=True, exist_ok=False)
    children: list[JsonObject] = []
    actual = 0
    uncertain = 0
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for row in rows:
        run_id = str(row["run_id"])
        run_directory = repository_root / registry.child_runs_root / run_id
        manifest_text = read_text_artifact(run_directory / "manifest.json")
        results_text = read_text_artifact(run_directory / "results.json")
        analysis_manifest_text = read_text_artifact(run_directory / "analysis/manifest.json")
        child_analysis = parse_json_object(analysis_manifest_text)
        child_files = child_analysis.get("files")
        if not isinstance(child_files, dict):
            raise PersistenceError("Phase 4 child analysis file map is malformed")
        frozen_files: JsonObject = {}
        for name, digest in child_files.items():
            if not isinstance(name, str) or not isinstance(digest, str):
                raise PersistenceError("Phase 4 child analysis hash is malformed")
            content = read_text_artifact(run_directory / "analysis" / name)
            if sha256_text(content) != digest:
                raise PersistenceError("Phase 4 child analysis artifact hash mismatch")
            frozen_files[name] = digest
        request_manifest = parse_json_object(
            read_text_artifact(run_directory / "analysis/request-manifest.json")
        )
        result = parse_json_object(results_text)
        budget = result.get("budget")
        counters = budget.get("counters") if isinstance(budget, dict) else None
        if not isinstance(counters, dict):
            raise PersistenceError("Phase 4 child budget counters are malformed")

        actual += _record_counter(counters, "actual_nano_usd")
        uncertain += _record_counter(counters, "uncertain_nano_usd")
        input_tokens += _record_counter(counters, "input_tokens")
        cached_tokens += _record_counter(counters, "cached_input_tokens")
        output_tokens += _record_counter(counters, "output_tokens")
        total_tokens += _record_counter(counters, "total_tokens")
        manifest = parse_json_object(manifest_text)
        runtime = manifest.get("runtime")
        dependency_lock = runtime.get("dependency_lock") if isinstance(runtime, dict) else None
        dependency_lock_hash = (
            dependency_lock.get("sha256") if isinstance(dependency_lock, dict) else None
        )
        children.append(
            {
                "run_id": run_id,
                "task_id": row["task_id"],
                "search_seed": row["search_seed"],
                "condition_id": row["condition_id"],
                "manifest_hash": sha256_text(manifest_text),
                "results_hash": sha256_text(results_text),
                "analysis_manifest_hash": sha256_text(analysis_manifest_text),
                "configuration_hash": row["configuration_hash"],
                "dependency_lock_hash": dependency_lock_hash,
                "version_contract": manifest.get("versions"),
                "model_contract": manifest.get("proposer"),
                "price_contract": manifest.get("budget"),
                "request_manifest": request_manifest,
                "analysis_file_hashes": frozen_files,
                "metrics": result.get("metrics"),
                "budget": budget,
            }
        )
    policy = load_price_policy(repository_root / base.phase4_policy.price_policy)
    artifact: JsonObject = {
        "presentable_artifact_version": "phase4-section-11.5-recorded-artifact-v1",
        "source": "immutable-child-and-aggregate-records-only",
        "experiment_id": registry.experiment_id,
        "registry_hash": registry.content_hash,
        "evidence_class": summary["evidence_class"],
        "scientific_gate": summary["scientific_gate"],
        "locked_test_disposition": summary["locked_test_disposition"],
        "F0_only_deviation": (
            "Only elementary binary radius-one F0 exists; F1/F2 are not implemented and no broad "
            "world-model or general LLM superiority claim is made."
        ),
        "condition_contract": {
            "conditions": cast(list[JsonValue], [condition.value for condition in CONDITIONS]),
            "execution_order": cast(list[JsonValue], execution_order),
            "mechanism_differences": {
                "H1": "B-A isolates iterative parent score feedback and incumbent retention",
                "H2": "C-B isolates uniform branch diversity and archive retention",
                "secondary": "C-A is descriptive only",
            },
            "matched": registry.raw["matched_contract"],
        },
        "dry_run_versus_actual": {
            "forecast_child_ceiling_nano_usd": base.phase4_budget.child_nano_usd_cap,
            "forecast_all_child_ceiling_nano_usd": len(rows)
            * base.phase4_budget.child_nano_usd_cap,
            "actual_estimated_nano_usd": actual,
            "uncertain_nano_usd": uncertain,
            "price_policy_hash": policy.content_hash,
            "published_rate_estimate_not_provider_invoice": True,
        },
        "token_reconciliation": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens_including_reasoning": output_tokens,
            "total_tokens": total_tokens,
            "total_equals_input_plus_output": total_tokens == input_tokens + output_tokens,
        },
        "paired_H1_H2_analysis": analysis,
        "raw_rows": cast(list[JsonValue], rows),
        "children": cast(list[JsonValue], children),
        "artifact_views": {
            "exact_curves": "each child analysis/exact-curves.csv",
            "best_program_and_MDL": "child results metrics plus analysis/lineage.json",
            "archive_coverage_separate_from_correctness": True,
            "validity_duplicate_retry_cache": "child analysis/proposal-diagnostics.json",
            "selected_lineages": "child analysis/lineage.json",
            "runtime_and_provider_latency": "child analysis/runtime-diagnostics.json",
            "split_access": "child analysis/access-ledger.json",
            "failure_analysis": "child analysis/failure-analysis.json",
        },
    }
    artifact_hash = write_content_artifact(
        report_root / "phase4-artifact.json", canonical_json(artifact)
    )
    markdown = (
        "# Phase 4 recorded A/B/C artifact\n\n"
        f"Evidence class: `{summary['evidence_class']}`. Scientific gate: "
        f"`{summary['scientific_gate']}`.\n\n"
        "H1 is B-A; H2 is C-B. The JSON artifact retains every paired row, interval, "
        "request/prompt/response hash, budget, cost, token count, lineage and diagnostic-file "
        "reference. Fake evidence validates the machinery only and cannot establish H1/H2.\n\n"
        "This repository has F0 only; F1/F2 and Phase 5+ mechanisms are absent.\n"
    )
    markdown_hash = write_content_artifact(report_root / "README.md", markdown.rstrip("\n"))
    report_manifest: JsonObject = {
        "report_manifest_version": 1,
        "source": "recorded-artifacts-only",
        "files": {
            "phase4-artifact.json": artifact_hash,
            "README.md": markdown_hash,
        },
    }
    write_content_artifact(report_root / "manifest.json", canonical_json(report_manifest))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def _bootstrap_samples(
    differences: dict[str, list[float]], *, seed: int, replicates: int, clustered: bool
) -> list[float]:
    rng = random.Random(seed)
    tasks = sorted(differences)
    pairs = [(task, value) for task in tasks for value in differences[task]]
    samples: list[float] = []
    for _ in range(replicates):
        if clustered:
            selected = [tasks[rng.randrange(len(tasks))] for _ in tasks]
            values = [value for task in selected for value in differences[task]]
        else:
            values = [pairs[rng.randrange(len(pairs))][1] for _ in pairs]
        samples.append(mean(values))
    return samples


def _bootstrap_interval(
    differences: dict[str, list[float]], *, seed: int, replicates: int, clustered: bool
) -> tuple[float, float]:
    samples = _bootstrap_samples(differences, seed=seed, replicates=replicates, clustered=clustered)
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def _clustered_two_sided_p_value(
    differences: dict[str, list[float]], *, seed: int, replicates: int
) -> float:
    observed = mean(value for values in differences.values() for value in values)
    centered = {
        task: [value - observed for value in values] for task, values in differences.items()
    }
    null_samples = _bootstrap_samples(centered, seed=seed, replicates=replicates, clustered=True)
    extreme = sum(abs(sample) >= abs(observed) for sample in null_samples)
    return (extreme + 1) / (replicates + 1)


def _holm_two_hypotheses(p_values: dict[str, float]) -> dict[str, tuple[float, bool]]:
    if set(p_values) != {"H1_B_minus_A", "H2_C_minus_B"}:
        raise ValueError("Holm correction requires exactly the two primary hypotheses")
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = running
    rejected: dict[str, bool] = {}
    continue_rejecting = True
    for rank, name in enumerate(ordered):
        threshold = 0.05 / (len(ordered) - rank)
        reject = continue_rejecting and p_values[name] <= threshold
        rejected[name] = reject
        continue_rejecting = reject
    return {name: (adjusted[name], rejected[name]) for name in p_values}


def analyze_phase4_rows(
    rows: list[JsonObject], *, bootstrap_seed: int, bootstrap_replicates: int
) -> JsonObject:
    """Predeclared paired H1/H2 analysis; negative and null results are retained."""

    indexed: dict[tuple[str, int, str], float] = {}
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise PersistenceError("Phase 4 experiment row has no metrics")
        value = metrics.get("normalized_exact_auc")
        task_id = row.get("task_id")
        seed = row.get("search_seed")
        condition = row.get("condition_id")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not isinstance(task_id, str)
            or not isinstance(seed, int)
            or not isinstance(condition, str)
        ):
            raise PersistenceError("Phase 4 experiment primary metric row is malformed")
        indexed[(task_id, seed, condition)] = float(value)
    tasks_and_seeds = sorted({(task, seed) for task, seed, _ in indexed})
    expected = len(tasks_and_seeds) * 3
    if len(indexed) != expected:
        raise PersistenceError("Phase 4 analysis requires complete paired A/B/C rows")
    contrasts = {
        "H1_B_minus_A": (Phase4Condition.INCUMBENT.value, Phase4Condition.DIRECT.value),
        "H2_C_minus_B": (Phase4Condition.DIVERSE.value, Phase4Condition.INCUMBENT.value),
        "secondary_C_minus_A": (Phase4Condition.DIVERSE.value, Phase4Condition.DIRECT.value),
    }
    output: JsonObject = {
        "analysis_version": "phase4-paired-cluster-bootstrap-v1",
        "primary_endpoint": "normalized-exact-solve-auc-v1",
        "multiplicity": "holm-two-sided-two-hypotheses-v1",
        "paired_rows": [],
        "contrasts": {},
    }
    paired_rows: list[JsonObject] = []
    contrast_output: JsonObject = {}
    primary_p_values: dict[str, float] = {}
    for contrast_index, (name, (left, right)) in enumerate(contrasts.items()):
        by_task: dict[str, list[float]] = {}
        for task, seed in tasks_and_seeds:
            difference = indexed[(task, seed, left)] - indexed[(task, seed, right)]
            by_task.setdefault(task, []).append(difference)
            paired_rows.append(
                {
                    "contrast": name,
                    "task_id": task,
                    "search_seed": seed,
                    "left_condition": left,
                    "right_condition": right,
                    "difference": difference,
                }
            )
        values = [value for task_values in by_task.values() for value in task_values]
        clustered = _bootstrap_interval(
            by_task,
            seed=bootstrap_seed + contrast_index * 2,
            replicates=bootstrap_replicates,
            clustered=True,
        )
        sensitivity = _bootstrap_interval(
            by_task,
            seed=bootstrap_seed + contrast_index * 2 + 1,
            replicates=bootstrap_replicates,
            clustered=False,
        )
        point_estimate = mean(values)
        p_value = _clustered_two_sided_p_value(
            by_task,
            seed=bootstrap_seed + 10_000 + contrast_index,
            replicates=bootstrap_replicates,
        )
        if name in {"H1_B_minus_A", "H2_C_minus_B"}:
            primary_p_values[name] = p_value
        contrast_output[name] = {
            "point_estimate": point_estimate,
            "task_clustered_95_interval": list(clustered),
            "task_seed_pair_sensitivity_95_interval": list(sensitivity),
            "task_clustered_two_sided_p_value": p_value,
            "task_count": len(by_task),
            "pair_count": len(values),
            "holm_adjusted_p_value": None,
            "holm_reject_0_05": False,
            "superiority_established": False,
        }
    holm = _holm_two_hypotheses(primary_p_values)
    for name, (adjusted_p_value, reject) in holm.items():
        result = contrast_output[name]
        if not isinstance(result, dict):
            raise AssertionError("Phase 4 contrast output is malformed")
        result["holm_adjusted_p_value"] = adjusted_p_value
        result["holm_reject_0_05"] = reject
        point = result.get("point_estimate")
        if not isinstance(point, int | float) or isinstance(point, bool):
            raise AssertionError("Phase 4 point estimate is malformed")
        result["superiority_established"] = reject and point > 0
    output["paired_rows"] = cast(list[JsonValue], paired_rows)
    output["contrasts"] = contrast_output
    return output
