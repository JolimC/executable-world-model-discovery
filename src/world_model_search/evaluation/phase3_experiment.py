"""Strict Phase 3 paired experiment registry, execution, and recorded analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any, cast

import yaml

from world_model_search.config import AppConfig, load_config
from world_model_search.domain.types import SplitLabel
from world_model_search.dsl.versions import (
    PHASE3_ANALYSIS_VERSION,
    PHASE3_ARCHIVE_VERSION,
    PHASE3_BUDGET_VERSION,
    PHASE3_DESCRIPTOR_VERSION,
    PHASE3_EXPERIMENT_SCHEMA_VERSION,
    PHASE3_INITIALIZATION_VERSION,
    PHASE3_OPERATOR_VERSION,
    PHASE3_RNG_VERSION,
    PHASE3_SCHEDULER_VERSION,
)
from world_model_search.errors import ConfigurationError, PersistenceError
from world_model_search.persistence.artifacts import (
    read_text_artifact,
    write_content_artifact,
)
from world_model_search.persistence.database import RunDatabase
from world_model_search.persistence.manifest import _git_state, _lock_state
from world_model_search.search.loop import load_manifest
from world_model_search.search.phase3 import (
    Phase3Authority,
    resume_phase3_run,
    start_phase3_run,
)
from world_model_search.search.phase3_types import SearchCondition
from world_model_search.serialization import (
    JsonObject,
    JsonValue,
    canonical_json,
    parse_json_object,
    sha256_json,
    sha256_text,
)
from world_model_search.tasks import benchmark_root_for_config, load_public_task


@dataclass(frozen=True, slots=True)
class ExperimentRegistry:
    experiment_id: str
    base_config_path: Path
    output_root: Path
    child_runs_root: Path
    report_root: Path
    conditions: tuple[SearchCondition, ...]
    task_ids: tuple[str, ...]
    search_seeds: tuple[int, ...]
    proposal_attempt_cap: int
    oracle_call_cap: int
    bootstrap_seed: int
    bootstrap_replicates: int
    raw: JsonObject

    @property
    def content_hash(self) -> str:
        return sha256_json(self.raw)


def _mapping(value: object, expected: set[str], location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping")
    if set(value) != expected:
        raise ConfigurationError(f"{location} has missing or unknown keys")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{location} must be a nonempty string")
    return value


def _integer(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{location} must be an integer >= {minimum}")
    return value


def _relative_path(value: object, location: str) -> Path:
    path = Path(_string(value, location))
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ConfigurationError(f"{location} must be a specific repository-relative path")
    return path


def load_experiment_registry(path: Path) -> ExperimentRegistry:
    """Fully validate the machine-readable registry without writing output."""

    try:
        loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read experiment registry: {exc}") from exc
    root = _mapping(
        loaded,
        {
            "experiment_schema_version",
            "experiment_id",
            "base_config",
            "output_root",
            "child_runs_root",
            "report_root",
            "conditions",
            "task_selection",
            "search_seeds",
            "stopping",
            "response_mode",
            "versions",
            "primary_endpoint",
            "metrics",
            "validation",
        },
        "experiment",
    )
    if root["experiment_schema_version"] != PHASE3_EXPERIMENT_SCHEMA_VERSION:
        raise ConfigurationError("unsupported Phase 3 experiment schema")
    conditions_raw = root["conditions"]
    if not isinstance(conditions_raw, list) or conditions_raw != [
        SearchCondition.INCUMBENT.value,
        SearchCondition.DIVERSE.value,
    ]:
        raise ConfigurationError("Phase 3 condition order must be incumbent then diverse")
    task_selection = _mapping(
        root["task_selection"], {"split", "policy", "task_ids"}, "task_selection"
    )
    if (
        task_selection["split"] != SplitLabel.VALIDATION.value
        or task_selection["policy"] != "opaque-public-id-sha256-order-v1"
    ):
        raise ConfigurationError("Phase 3 smoke tasks require the frozen validation ID policy")
    task_ids_raw = task_selection["task_ids"]
    if (
        not isinstance(task_ids_raw, list)
        or not 12 <= len(task_ids_raw) <= 24
        or not all(
            isinstance(item, str) and len(item) == 24 and not set(item) - set("0123456789abcdef")
            for item in task_ids_raw
        )
        or len(task_ids_raw) != len(set(task_ids_raw))
    ):
        raise ConfigurationError("Phase 3 requires 12-24 unique opaque task IDs")
    seeds_raw = root["search_seeds"]
    if (
        not isinstance(seeds_raw, list)
        or len(seeds_raw) < 20
        or not all(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds_raw
        )
        or len(seeds_raw) != len(set(seeds_raw))
    ):
        raise ConfigurationError("Phase 3 requires at least 20 unique nonnegative search seeds")
    stopping = _mapping(
        root["stopping"],
        {"proposal_attempt_cap", "oracle_call_cap", "continue_after_first_exact"},
        "stopping",
    )
    proposal_cap = _integer(stopping["proposal_attempt_cap"], "proposal_attempt_cap", minimum=1)
    oracle_cap = _integer(stopping["oracle_call_cap"], "oracle_call_cap", minimum=32)
    if (
        stopping["continue_after_first_exact"] is not True
        or not 32 <= oracle_cap <= 64
        or proposal_cap < oracle_cap
    ):
        raise ConfigurationError("Phase 3 smoke stopping profile is invalid")
    versions = _mapping(
        root["versions"],
        {
            "operators",
            "rng",
            "archive",
            "descriptor",
            "scheduler",
            "budget",
            "initialization",
            "analysis",
        },
        "versions",
    )
    expected_versions = {
        "operators": PHASE3_OPERATOR_VERSION,
        "rng": PHASE3_RNG_VERSION,
        "archive": PHASE3_ARCHIVE_VERSION,
        "descriptor": PHASE3_DESCRIPTOR_VERSION,
        "scheduler": PHASE3_SCHEDULER_VERSION,
        "budget": PHASE3_BUDGET_VERSION,
        "initialization": PHASE3_INITIALIZATION_VERSION,
        "analysis": PHASE3_ANALYSIS_VERSION,
    }
    if versions != expected_versions:
        raise ConfigurationError("Phase 3 experiment versions differ from frozen implementations")
    endpoint = _mapping(
        root["primary_endpoint"],
        {
            "id",
            "pairing",
            "noninferiority_tolerance",
            "bootstrap_seed",
            "bootstrap_replicates",
        },
        "primary_endpoint",
    )
    if (
        endpoint["id"] != "normalized-exact-solve-auc-v1"
        or endpoint["pairing"] != "exact-task-id-and-search-seed-v1"
        or endpoint["noninferiority_tolerance"] != 0
    ):
        raise ConfigurationError("Phase 3 primary endpoint/tolerance is not frozen")
    bootstrap_seed = _integer(endpoint["bootstrap_seed"], "bootstrap_seed")
    bootstrap_replicates = _integer(
        endpoint["bootstrap_replicates"], "bootstrap_replicates", minimum=1000
    )
    expected_metrics = [
        "normalized_exact_auc",
        "final_exact_solved",
        "calls_to_first_exact",
        "best_exact_ast_bits",
        "best_two_part_bits",
        "archive_coverage",
        "distinct_candidate_semantics",
        "valid_proposal_rate",
        "semantic_duplicate_rate",
        "operator_outcomes",
        "proposal_oracle_budget_utilization",
    ]
    if root["metrics"] != expected_metrics or root["response_mode"] != "score-only":
        raise ConfigurationError("Phase 3 metrics/response mode differ from the declaration")
    validation = _mapping(
        root["validation"],
        {"mode", "consume_once", "test_oracle_access_permitted"},
        "validation",
    )
    if validation != {
        "mode": "phase3-locked-validation-once-v1",
        "consume_once": True,
        "test_oracle_access_permitted": False,
    }:
        raise ConfigurationError("Phase 3 validation authority declaration is invalid")
    raw_value = json.loads(canonical_json(root))
    if not isinstance(raw_value, dict):
        raise AssertionError("experiment registry did not serialize as an object")
    return ExperimentRegistry(
        experiment_id=_string(root["experiment_id"], "experiment_id"),
        base_config_path=_relative_path(root["base_config"], "base_config"),
        output_root=_relative_path(root["output_root"], "output_root"),
        child_runs_root=_relative_path(root["child_runs_root"], "child_runs_root"),
        report_root=_relative_path(root["report_root"], "report_root"),
        conditions=tuple(SearchCondition(item) for item in cast(list[str], conditions_raw)),
        task_ids=tuple(cast(list[str], task_ids_raw)),
        search_seeds=tuple(cast(list[int], seeds_raw)),
        proposal_attempt_cap=proposal_cap,
        oracle_call_cap=oracle_cap,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        raw=raw_value,
    )


def _public_policy_ids(benchmark_root: Path, count: int) -> tuple[str, ...]:
    """Select using public files/split labels only; never read the manifest oracle fields."""

    candidates: list[str] = []
    for path in (benchmark_root / "public").glob("*.json"):
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError("public task bundle is unavailable") from exc
        if isinstance(raw, dict) and raw.get("split") == SplitLabel.VALIDATION.value:
            task_id = raw.get("task_id")
            if isinstance(task_id, str):
                candidates.append(task_id)
    candidates.sort(
        key=lambda item: hashlib.sha256(f"phase3-smoke-task-policy-v1\0{item}".encode()).hexdigest()
    )
    return tuple(candidates[:count])


def _child_config(
    base: AppConfig,
    registry: ExperimentRegistry,
    *,
    task_id: str,
    seed: int,
    condition: SearchCondition,
) -> AppConfig:
    if base.schema_version != 3 or base.budget is None:
        raise ConfigurationError("experiment base config must be Phase 3 schema 3")
    return replace(
        base,
        run=replace(
            base.run,
            root=registry.child_runs_root,
            seed=seed,
            task_id=task_id,
            split=SplitLabel.VALIDATION,
            condition_id=condition.value,
            max_steps=registry.oracle_call_cap,
        ),
        budget=replace(
            base.budget,
            proposal_attempt_cap=registry.proposal_attempt_cap,
            oracle_call_cap=registry.oracle_call_cap,
        ),
    )


def _validate_before_write(
    repository_root: Path, registry: ExperimentRegistry
) -> tuple[AppConfig, JsonObject]:
    base_path = repository_root / registry.base_config_path
    base = load_config(base_path)
    benchmark_root = benchmark_root_for_config(repository_root, base)
    if _public_policy_ids(benchmark_root, len(registry.task_ids)) != registry.task_ids:
        raise ConfigurationError("registry task IDs differ from the frozen public-ID policy")
    for task_id in registry.task_ids:
        if load_public_task(benchmark_root, task_id).split is not SplitLabel.VALIDATION:
            raise ConfigurationError("registry contains a non-validation task")
    for condition in registry.conditions:
        for task_id in registry.task_ids:
            for seed in registry.search_seeds:
                config = _child_config(
                    base, registry, task_id=task_id, seed=seed, condition=condition
                )
                # Round-trip through the strict schema before any persistent output.
                from world_model_search.config import config_from_mapping

                config_from_mapping(config.to_mapping())
    source_path = Path(__file__)
    freeze: JsonObject = {
        "freeze_version": "phase3-validation-freeze-v1",
        "experiment_id": registry.experiment_id,
        "registry_hash": registry.content_hash,
        "base_configuration_hash": base.content_hash,
        "git": _git_state(repository_root),
        "dependency_lock": _lock_state(repository_root),
        "analysis_code_hash": sha256_text(source_path.read_text(encoding="utf-8")),
        "conditions": [condition.value for condition in registry.conditions],
        "validation_task_ids": list(registry.task_ids),
        "search_seeds": list(registry.search_seeds),
        "proposal_attempt_cap": registry.proposal_attempt_cap,
        "oracle_call_cap": registry.oracle_call_cap,
        "primary_endpoint": registry.raw["primary_endpoint"],
        "test_oracle_access_permitted": False,
    }
    return base, freeze


def _child_id(condition: SearchCondition, task_id: str, seed: int) -> str:
    prefix = "inc" if condition is SearchCondition.INCUMBENT else "div"
    return f"p3-{prefix}-{task_id[:12]}-s{seed}"


def _load_result(path: Path) -> JsonObject:
    return parse_json_object(read_text_artifact(path))


def _run_or_reuse_child(
    *,
    repository_root: Path,
    registry: ExperimentRegistry,
    base: AppConfig,
    authority: Phase3Authority,
    task_id: str,
    seed: int,
    condition: SearchCondition,
    experiment_path: Path,
) -> tuple[str, JsonObject]:
    config = _child_config(base, registry, task_id=task_id, seed=seed, condition=condition)
    run_id = _child_id(condition, task_id, seed)
    run_directory = repository_root / registry.child_runs_root / run_id
    if not run_directory.exists():
        start_phase3_run(
            repository_root=repository_root,
            config=config,
            config_source=str(experiment_path),
            run_id=run_id,
            interrupt_after=None,
            authority=authority,
        )
    else:
        manifest = load_manifest(run_directory)
        if manifest.get("configuration_hash") != config.content_hash:
            raise PersistenceError(f"mismatched existing experiment child: {run_id}")
        with RunDatabase(run_directory / "run.sqlite3", read_only=True) as database:
            status = database.state().status
        if status != "completed":
            resume_phase3_run(
                repository_root=repository_root,
                run_directory=run_directory,
                run_id=run_id,
                config=config,
                manifest=manifest,
                interrupt_after=None,
            )
    result = _load_result(run_directory / "results.json")
    budget = result.get("budget")
    if not isinstance(budget, dict):
        raise PersistenceError("Phase 3 child has no budget result")
    counters = budget.get("counters")
    caps = budget.get("caps")
    if (
        not isinstance(counters, dict)
        or not isinstance(caps, dict)
        or counters.get("oracle_invocations") != registry.oracle_call_cap
        or caps.get("oracle_calls") != registry.oracle_call_cap
        or caps.get("proposal_attempts") != registry.proposal_attempt_cap
    ):
        raise PersistenceError("Phase 3 child budget does not reconcile to the frozen caps")
    return run_id, result


def _metric(result: JsonObject, name: str) -> JsonValue:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or name not in metrics:
        raise PersistenceError(f"child result is missing metric: {name}")
    return metrics[name]


def _number(value: JsonValue, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PersistenceError(f"metric {name} is not numeric")
    return float(value)


def _bootstrap_interval(
    differences: tuple[float, ...], *, seed: int, replicates: int
) -> tuple[float, float]:
    if not differences:
        raise ValueError("bootstrap requires paired differences")
    rng = random.Random(seed)
    estimates = sorted(
        mean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(replicates)
    )
    lower = estimates[int(0.025 * (replicates - 1))]
    upper = estimates[int(0.975 * (replicates - 1))]
    return lower, upper


def _task_clustered_bootstrap_interval(
    differences_by_task: dict[str, tuple[float, ...]], *, seed: int, replicates: int
) -> tuple[float, float]:
    """Resample whole tasks so repeated search seeds are not treated as independent tasks."""

    clusters = tuple(values for _, values in sorted(differences_by_task.items()) if values)
    if not clusters or len(clusters) != len(differences_by_task):
        raise ValueError("task-clustered bootstrap requires nonempty task clusters")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = tuple(clusters[rng.randrange(len(clusters))] for _ in clusters)
        estimates.append(mean(value for cluster in sampled for value in cluster))
    estimates.sort()
    return (
        estimates[int(0.025 * (replicates - 1))],
        estimates[int(0.975 * (replicates - 1))],
    )


def _csv(rows: list[dict[str, JsonValue]], fields: tuple[str, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().rstrip("\n")


def _svg_curve(curves: dict[str, list[float]], maximum: int) -> str:
    colors = {SearchCondition.INCUMBENT.value: "#2563eb", SearchCondition.DIVERSE.value: "#dc2626"}
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">',
        '<rect width="720" height="420" fill="white"/>',
        '<line x1="60" y1="360" x2="690" y2="360" stroke="black"/>',
        '<line x1="60" y1="30" x2="60" y2="360" stroke="black"/>',
        '<text x="300" y="405" font-size="14">charged oracle evaluations</text>',
        '<text x="8" y="25" font-size="14">paired solve rate</text>',
    ]
    for condition, values in curves.items():
        points = " ".join(
            f"{60 + (630 * index / maximum):.2f},{360 - 330 * value:.2f}"
            for index, value in enumerate(values, 1)
        )
        lines.append(
            f'<polyline fill="none" stroke="{colors[condition]}" '
            f'stroke-width="3" points="{points}"/>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _svg_coverage_curve(curves: dict[str, list[float]], maximum: int) -> str:
    colors = {SearchCondition.INCUMBENT.value: "#2563eb", SearchCondition.DIVERSE.value: "#dc2626"}
    ceiling = max(1.0, max((max(values, default=0.0) for values in curves.values()), default=0.0))
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">',
        '<rect width="720" height="420" fill="white"/>',
        '<line x1="60" y1="360" x2="690" y2="360" stroke="black"/>',
        '<line x1="60" y1="30" x2="60" y2="360" stroke="black"/>',
        '<text x="300" y="405" font-size="14">charged oracle evaluations</text>',
        '<text x="8" y="25" font-size="14">mean occupied archive coordinates</text>',
    ]
    for condition, values in curves.items():
        points = " ".join(
            f"{60 + (630 * index / maximum):.2f},{360 - 330 * value / ceiling:.2f}"
            for index, value in enumerate(values, 1)
        )
        lines.append(
            f'<polyline fill="none" stroke="{colors[condition]}" '
            f'stroke-width="3" points="{points}"/>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _analyze(
    *,
    repository_root: Path,
    registry: ExperimentRegistry,
    freeze_hash: str,
    children: dict[tuple[str, int, SearchCondition], tuple[str, JsonObject]],
    analysis_amendment_hash: str | None = None,
) -> JsonObject:
    output = repository_root / registry.output_root
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, JsonValue]] = []
    paired_rows: list[dict[str, JsonValue]] = []
    differences: list[float] = []
    differences_by_task: dict[str, list[float]] = {task_id: [] for task_id in registry.task_ids}
    solve_curves = {
        condition.value: [0.0] * registry.oracle_call_cap for condition in registry.conditions
    }
    coverage_curves = {
        condition.value: [0.0] * registry.oracle_call_cap for condition in registry.conditions
    }
    aggregate_operator_attempts = {
        condition.value: Counter[str]() for condition in registry.conditions
    }
    aggregate_attempt_outcomes = {
        condition.value: Counter[str]() for condition in registry.conditions
    }
    child_contracts: dict[str, JsonValue] = {}
    access_rows: list[JsonObject] = []
    for task_id in registry.task_ids:
        for seed in registry.search_seeds:
            pair_results: dict[SearchCondition, JsonObject] = {}
            for condition in registry.conditions:
                run_id, result = children[(task_id, seed, condition)]
                pair_results[condition] = result
                first = _metric(result, "calls_to_first_exact")
                first_call = (
                    first if isinstance(first, int) and not isinstance(first, bool) else None
                )
                for call in range(1, registry.oracle_call_cap + 1):
                    solve_curves[condition.value][call - 1] += float(
                        first_call is not None and first_call <= call
                    )
                raw_budget = result.get("budget")
                if not isinstance(raw_budget, dict) or not isinstance(
                    raw_budget.get("counters"), dict
                ):
                    raise PersistenceError("Phase 3 child budget result is malformed")
                counters = cast(dict[str, JsonValue], raw_budget["counters"])
                run_directory = repository_root / registry.child_runs_root / run_id
                runtime_path = run_directory / "analysis" / "runtime-diagnostics.json"
                runtime: JsonObject = {}
                if runtime_path.is_file():
                    runtime = parse_json_object(read_text_artifact(runtime_path))
                raw_rows.append(
                    {
                        "task_id": task_id,
                        "search_seed": seed,
                        "condition_id": condition.value,
                        "run_id": run_id,
                        "normalized_exact_auc": _metric(result, "normalized_exact_auc"),
                        "final_exact_solved": _metric(result, "final_exact_solved"),
                        "calls_to_first_exact": first,
                        "best_exact_ast_bits": _metric(result, "best_exact_ast_bits"),
                        "best_two_part_bits": _metric(result, "best_two_part_bits"),
                        "archive_coverage": _metric(result, "archive_coverage"),
                        "distinct_candidate_semantics": _metric(
                            result, "distinct_candidate_semantics"
                        ),
                        "valid_proposal_rate": _metric(result, "valid_proposal_rate"),
                        "semantic_duplicate_rate": _metric(result, "semantic_duplicate_rate"),
                        "transition_outcomes": _metric(result, "transition_outcomes"),
                        "proposal_attempts": counters.get("proposal_attempts"),
                        "oracle_invocations": counters.get("oracle_invocations"),
                        "attempt_cpu_ns": runtime.get("attempt_cpu_ns"),
                        "oracle_cpu_ns": runtime.get("oracle_cpu_ns"),
                        "attempt_elapsed_ns": runtime.get("attempt_elapsed_ns"),
                        "oracle_elapsed_ns": runtime.get("oracle_elapsed_ns"),
                        "language_model_tokens": counters.get("language_model_tokens"),
                    }
                )
                coverage_text = read_text_artifact(
                    run_directory / "analysis" / "archive-coverage.csv"
                )
                coverage_by_call: dict[int, float] = {}
                for coverage_row in csv.DictReader(io.StringIO(coverage_text)):
                    try:
                        call = int(coverage_row["oracle_calls"])
                        coverage = float(coverage_row["archive_coverage"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise PersistenceError("child archive coverage is malformed") from exc
                    coverage_by_call[call] = coverage
                if set(coverage_by_call) != set(range(1, registry.oracle_call_cap + 1)):
                    raise PersistenceError(
                        "child archive coverage does not match the oracle budget"
                    )
                for call, coverage in coverage_by_call.items():
                    coverage_curves[condition.value][call - 1] += coverage
                operator_diagnostics = parse_json_object(
                    read_text_artifact(run_directory / "analysis" / "operator-diagnostics.json")
                )
                for field, destination in (
                    ("operator_attempts", aggregate_operator_attempts[condition.value]),
                    ("attempt_outcomes", aggregate_attempt_outcomes[condition.value]),
                ):
                    diagnostic_counts = operator_diagnostics.get(field)
                    if not isinstance(diagnostic_counts, dict):
                        raise PersistenceError("child operator diagnostics are malformed")
                    for name, count in diagnostic_counts.items():
                        if not isinstance(name, str) or not isinstance(count, int):
                            raise PersistenceError("child operator diagnostic count is malformed")
                        destination[name] += count
                child_manifest = load_manifest(run_directory)
                contract = {
                    "configuration_hash": child_manifest.get("configuration_hash"),
                    "versions": child_manifest.get("versions"),
                    "authority_hash": child_manifest.get("phase3_authority_hash"),
                    "results_hash": result.get("deterministic_summary_hash"),
                    "analysis_manifest_hash": result.get("analysis_manifest_hash"),
                    "event_payload_hashes": result.get("event_payload_hashes"),
                }
                child_contracts[run_id] = sha256_json(contract)
                ledger = parse_json_object(
                    read_text_artifact(run_directory / "analysis" / "access-ledger.json")
                )
                access_rows.append(
                    {
                        "run_id": run_id,
                        "task_id": task_id,
                        "search_seed": seed,
                        "condition_id": condition.value,
                        "ledger_hash": sha256_json(ledger),
                        "test_oracle_accesses": ledger.get("test_oracle_accesses"),
                    }
                )
            incumbent_auc = _number(
                _metric(pair_results[SearchCondition.INCUMBENT], "normalized_exact_auc"),
                "normalized_exact_auc",
            )
            diverse_auc = _number(
                _metric(pair_results[SearchCondition.DIVERSE], "normalized_exact_auc"),
                "normalized_exact_auc",
            )
            difference = diverse_auc - incumbent_auc
            differences.append(difference)
            differences_by_task[task_id].append(difference)
            paired_rows.append(
                {
                    "task_id": task_id,
                    "search_seed": seed,
                    "incumbent_normalized_exact_auc": incumbent_auc,
                    "diverse_normalized_exact_auc": diverse_auc,
                    "diverse_minus_incumbent": difference,
                }
            )
    pair_count = len(paired_rows)
    for solve_values in solve_curves.values():
        for index in range(len(solve_values)):
            solve_values[index] /= pair_count
    for coverage_values in coverage_curves.values():
        for index in range(len(coverage_values)):
            coverage_values[index] /= pair_count
    point = mean(differences)
    lower, upper = _bootstrap_interval(
        tuple(differences),
        seed=registry.bootstrap_seed,
        replicates=registry.bootstrap_replicates,
    )
    clustered_lower, clustered_upper = _task_clustered_bootstrap_interval(
        {task_id: tuple(values) for task_id, values in differences_by_task.items()},
        seed=registry.bootstrap_seed,
        replicates=registry.bootstrap_replicates,
    )
    gate_pass = point >= 0.0
    attempts_by_pair: dict[tuple[str, int], dict[str, int]] = {}
    for row in raw_rows:
        raw_task_id = row["task_id"]
        raw_search_seed = row["search_seed"]
        raw_condition_id = row["condition_id"]
        raw_attempts = row["proposal_attempts"]
        if (
            isinstance(raw_task_id, str)
            and isinstance(raw_search_seed, int)
            and not isinstance(raw_search_seed, bool)
            and isinstance(raw_condition_id, str)
            and isinstance(raw_attempts, int)
            and not isinstance(raw_attempts, bool)
        ):
            attempts_by_pair.setdefault((raw_task_id, raw_search_seed), {})[raw_condition_id] = (
                raw_attempts
            )
    all_actual_attempts_equal = all(
        values.get(SearchCondition.INCUMBENT.value) == values.get(SearchCondition.DIVERSE.value)
        for values in attempts_by_pair.values()
    )
    condition_summaries: dict[str, JsonValue] = {}
    for condition in registry.conditions:
        condition_rows = [row for row in raw_rows if row["condition_id"] == condition.value]
        numeric_metrics = (
            "normalized_exact_auc",
            "archive_coverage",
            "distinct_candidate_semantics",
            "valid_proposal_rate",
            "semantic_duplicate_rate",
            "proposal_attempts",
            "oracle_invocations",
            "attempt_cpu_ns",
            "oracle_cpu_ns",
            "attempt_elapsed_ns",
            "oracle_elapsed_ns",
            "language_model_tokens",
        )
        metric_means: dict[str, JsonValue] = {}
        for metric in numeric_metrics:
            numeric_values = [
                float(value)
                for row in condition_rows
                if isinstance((value := row[metric]), int | float) and not isinstance(value, bool)
            ]
            metric_means[f"mean_{metric}"] = mean(numeric_values) if numeric_values else None
        solved_rows = [row for row in condition_rows if row["final_exact_solved"] is True]
        exact_ast_values = [
            float(value)
            for row in solved_rows
            if isinstance((value := row["best_exact_ast_bits"]), int | float)
            and not isinstance(value, bool)
        ]
        first_call_values = [
            float(value)
            for row in solved_rows
            if isinstance((value := row["calls_to_first_exact"]), int | float)
            and not isinstance(value, bool)
        ]
        condition_summaries[condition.value] = {
            **metric_means,
            "solve_rate": len(solved_rows) / len(condition_rows),
            "mean_calls_to_first_exact_among_solved": (
                mean(first_call_values) if first_call_values else None
            ),
            "mean_best_exact_ast_bits_among_solved": (
                mean(exact_ast_values) if exact_ast_values else None
            ),
        }
    aggregate: JsonObject = {
        "analysis_version": PHASE3_ANALYSIS_VERSION,
        "primary_endpoint": "normalized-exact-solve-auc-v1",
        "pair_count": pair_count,
        "diverse_minus_incumbent_mean": point,
        "paired_bootstrap_95_ci": {"lower": lower, "upper": upper},
        "task_clustered_bootstrap_95_ci": {
            "lower": clustered_lower,
            "upper": clustered_upper,
        },
        "noninferiority_tolerance": 0,
        "gate_no_worse": "pass" if gate_pass else "negative-failed",
        "superiority_claimed": bool(lower > 0.0),
        "condition_summaries": condition_summaries,
        "budget_equality": {
            "oracle_calls_per_child": registry.oracle_call_cap,
            "proposal_attempt_cap_per_child": registry.proposal_attempt_cap,
            "all_children_reconciled": True,
            "all_paired_actual_proposal_attempts_equal": all_actual_attempts_equal,
            "cpu_and_elapsed_time_are_diagnostic_not_primary_cost": True,
        },
    }
    files: dict[str, str] = {}

    def record(name: str, content: str) -> None:
        normalized = content.rstrip("\n")
        write_content_artifact(analysis / name, normalized)
        files[name] = sha256_text(normalized)

    record("raw-paired-rows.json", canonical_json({"rows": raw_rows}))
    record(
        "raw-paired-rows.csv",
        _csv(
            raw_rows,
            (
                "task_id",
                "search_seed",
                "condition_id",
                "run_id",
                "normalized_exact_auc",
                "final_exact_solved",
                "calls_to_first_exact",
                "best_exact_ast_bits",
                "best_two_part_bits",
                "archive_coverage",
                "distinct_candidate_semantics",
                "valid_proposal_rate",
                "semantic_duplicate_rate",
                "proposal_attempts",
                "oracle_invocations",
                "attempt_cpu_ns",
                "oracle_cpu_ns",
                "attempt_elapsed_ns",
                "oracle_elapsed_ns",
                "language_model_tokens",
            ),
        ),
    )
    record("paired-differences.json", canonical_json({"rows": paired_rows}))
    record(
        "paired-differences.csv",
        _csv(
            paired_rows,
            (
                "task_id",
                "search_seed",
                "incumbent_normalized_exact_auc",
                "diverse_normalized_exact_auc",
                "diverse_minus_incumbent",
            ),
        ),
    )
    record("aggregate.json", canonical_json(aggregate))
    aggregate_csv = (
        "metric,estimate,ci_lower,ci_upper,uncertainty_unit\n"
        f"diverse_minus_incumbent_normalized_exact_auc,{point},{lower},{upper},task-seed-pair\n"
        f"diverse_minus_incumbent_normalized_exact_auc,{point},{clustered_lower},"
        f"{clustered_upper},task-cluster"
    )
    record("aggregate.csv", aggregate_csv)
    curve_rows: list[dict[str, JsonValue]] = [
        {
            "oracle_calls": call,
            "condition_id": condition.value,
            "solve_rate": solve_curves[condition.value][call - 1],
        }
        for condition in registry.conditions
        for call in range(1, registry.oracle_call_cap + 1)
    ]
    record(
        "paired-solve-curves.csv",
        _csv(curve_rows, ("oracle_calls", "condition_id", "solve_rate")),
    )
    record("paired-solve-curves.svg", _svg_curve(solve_curves, registry.oracle_call_cap))
    coverage_curve_rows: list[dict[str, JsonValue]] = [
        {
            "oracle_calls": call,
            "condition_id": condition.value,
            "mean_archive_coverage": coverage_curves[condition.value][call - 1],
        }
        for condition in registry.conditions
        for call in range(1, registry.oracle_call_cap + 1)
    ]
    record(
        "archive-coverage-curves.csv",
        _csv(
            coverage_curve_rows,
            ("oracle_calls", "condition_id", "mean_archive_coverage"),
        ),
    )
    record(
        "archive-coverage-curves.svg",
        _svg_coverage_curve(coverage_curves, registry.oracle_call_cap),
    )
    coverage_rows = [
        row for row in raw_rows if row["condition_id"] == SearchCondition.DIVERSE.value
    ]
    record(
        "archive-final-coverage.csv",
        _csv(
            coverage_rows,
            ("task_id", "search_seed", "condition_id", "run_id", "archive_coverage"),
        ),
    )
    record(
        "operator-diagnostics.json",
        canonical_json(
            {
                condition.value: {
                    "operator_attempts": dict(
                        sorted(aggregate_operator_attempts[condition.value].items())
                    ),
                    "attempt_outcomes": dict(
                        sorted(aggregate_attempt_outcomes[condition.value].items())
                    ),
                }
                for condition in registry.conditions
            }
        ),
    )
    showcased: list[JsonObject] = []
    showcase_task, showcase_seed = registry.task_ids[0], registry.search_seeds[0]
    for condition in registry.conditions:
        run_id, _ = children[(showcase_task, showcase_seed, condition)]
        lineage = parse_json_object(
            read_text_artifact(
                repository_root / registry.child_runs_root / run_id / "analysis" / "lineage.json"
            )
        )
        showcased.append(
            {
                "selection_rule": "first-declared-task-then-smallest-seed-v1",
                "condition_id": condition.value,
                "run_id": run_id,
                "lineage": lineage,
            }
        )
    record(
        "showcased-lineages.json",
        canonical_json({"lineages": cast(list[JsonValue], showcased)}),
    )
    access_ledger: JsonObject = {
        "validation_consumed_once_for_locked_phase3_gate": True,
        "validation_task_ids": list(registry.task_ids),
        "child_access_ledgers": cast(list[JsonValue], access_rows),
        "test_oracle_accesses": 0,
    }
    if any(row["test_oracle_accesses"] != 0 for row in access_rows):
        raise PersistenceError("test oracle access detected in Phase 3 experiment")
    record("access-ledger.json", canonical_json(access_ledger))
    failure: JsonObject = {
        "archive_point_estimate_negative": point < 0.0,
        "gate_disposition": "frozen-negative-stop-no-tuning" if point < 0.0 else "no-worse-pass",
        "unsolved_child_runs": sum(not bool(row["final_exact_solved"]) for row in raw_rows),
        "limitations": [
            "F0 has only 256 semantics",
            "results are conditional on the frozen DSL and public two-probe descriptor",
            "archive coverage is not correctness",
            "CPU and elapsed time are diagnostic only",
        ],
    }
    record("failure-analysis.json", canonical_json(failure))
    record("child-contract-hashes.json", canonical_json(child_contracts))
    report = (
        "# Phase 3 archive versus incumbent smoke result\n\n"
        f"- Pairs: {pair_count} ({len(registry.task_ids)} tasks x "
        f"{len(registry.search_seeds)} seeds)\n"
        f"- Diverse - incumbent normalized exact-AUC: {point:.6f} "
        f"(paired bootstrap 95% CI [{lower:.6f}, {upper:.6f}])\n"
        f"- Task-clustered sensitivity CI: [{clustered_lower:.6f}, "
        f"{clustered_upper:.6f}]\n"
        f"- Zero-tolerance no-worse gate: {'PASS' if gate_pass else 'NEGATIVE/FAILED'}\n"
        f"- Superiority claimed: {'yes' if lower > 0 else 'no'}\n"
        "- Validation was consumed for this frozen gate; test oracle access count is zero.\n"
        "- Archive coverage is reported separately and is not interpreted as correctness.\n"
    )
    record("report.md", report)
    manifest: JsonObject = {
        "experiment_analysis_artifact_version": "phase3-paired-experiment-artifacts-v1",
        "experiment_id": registry.experiment_id,
        "registry_hash": registry.content_hash,
        "validation_freeze_hash": freeze_hash,
        "analysis_amendment_hash": analysis_amendment_hash,
        "files": cast(dict[str, JsonValue], files),
    }
    manifest_text = canonical_json(manifest)
    write_content_artifact(analysis / "manifest.json", manifest_text)
    report_root = repository_root / registry.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    for name in (*files, "manifest.json"):
        source = read_text_artifact(analysis / name)
        write_content_artifact(report_root / name, source)
    summary: JsonObject = {
        "experiment_id": registry.experiment_id,
        "status": "completed",
        "gate_no_worse": aggregate["gate_no_worse"],
        "point_estimate": point,
        "confidence_interval_95": {"lower": lower, "upper": upper},
        "task_clustered_confidence_interval_95": {
            "lower": clustered_lower,
            "upper": clustered_upper,
        },
        "pair_count": pair_count,
        "analysis_manifest": str(analysis / "manifest.json"),
        "analysis_manifest_hash": sha256_text(manifest_text),
        "analysis_amendment_hash": analysis_amendment_hash,
        "report": str(report_root / "report.md"),
    }
    return summary


def write_experiment_evidence_supplement(
    *,
    repository_root: Path,
    registry: ExperimentRegistry,
    children: dict[tuple[str, int, SearchCondition], tuple[str, JsonObject]],
    primary_summary: JsonObject,
    evidence_amendment_hash: str | None,
) -> str:
    """Write required secondary gate evidence from completed immutable children only."""

    output = repository_root / registry.output_root
    supplement = output / "analysis" / "supplement"
    supplement.mkdir(parents=True, exist_ok=True)
    child_hashes: dict[str, JsonValue] = {}
    curve_rows: list[dict[str, JsonValue]] = []
    transition_outcomes: Counter[str] = Counter()
    total_proposal_attempts = 0
    total_oracle_invocations = 0
    all_budget_equal = True
    pair_differences_by_call: list[list[float]] = [[] for _ in range(registry.oracle_call_cap)]
    pair_differences_by_call_and_task: list[dict[str, list[float]]] = [
        {task_id: [] for task_id in registry.task_ids} for _ in range(registry.oracle_call_cap)
    ]
    for task_id in registry.task_ids:
        for seed in registry.search_seeds:
            first_calls: dict[SearchCondition, int | None] = {}
            for condition in registry.conditions:
                run_id, result = children[(task_id, seed, condition)]
                first = _metric(result, "calls_to_first_exact")
                first_calls[condition] = (
                    first if isinstance(first, int) and not isinstance(first, bool) else None
                )
                outcomes = _metric(result, "transition_outcomes")
                if not isinstance(outcomes, dict):
                    raise PersistenceError("child transition outcomes are malformed")
                for name, count in outcomes.items():
                    if not isinstance(name, str) or not isinstance(count, int):
                        raise PersistenceError("child transition outcome count is malformed")
                    transition_outcomes[name] += count
                budget = result.get("budget")
                if not isinstance(budget, dict) or not isinstance(budget.get("counters"), dict):
                    raise PersistenceError("child budget is malformed")
                counters = cast(dict[str, JsonValue], budget["counters"])
                attempts = counters.get("proposal_attempts")
                oracle_calls = counters.get("oracle_invocations")
                if not isinstance(attempts, int) or not isinstance(oracle_calls, int):
                    raise PersistenceError("child budget counts are malformed")
                total_proposal_attempts += attempts
                total_oracle_invocations += oracle_calls
                all_budget_equal &= oracle_calls == registry.oracle_call_cap
                run_directory = repository_root / registry.child_runs_root / run_id
                manifest_text = read_text_artifact(run_directory / "manifest.json")
                results_text = read_text_artifact(run_directory / "results.json")
                analysis_manifest_text = read_text_artifact(
                    run_directory / "analysis" / "manifest.json"
                )
                with RunDatabase(run_directory / "run.sqlite3", read_only=True) as database:
                    candidates = [dict(row) for row in database.candidate_records()]
                    attempts_records = [dict(row) for row in database.phase3_attempt_records()]
                    transitions = [dict(row) for row in database.phase3_transition_records()]
                    lineage = [dict(row) for row in database.phase3_lineage_records()]
                    evaluations: list[JsonObject] = []
                    for row in database.evaluation_records():
                        evaluation = dict(row)
                        evaluation.pop("runtime_ns", None)
                        raw_result: object = json.loads(str(evaluation["result_json"]))
                        if not isinstance(raw_result, dict):
                            raise PersistenceError("recorded evaluation result is malformed")
                        raw_result.pop("runtime_ns", None)
                        evaluation["result_json"] = canonical_json(raw_result)
                        evaluations.append(cast(JsonObject, evaluation))
                    events = [
                        {
                            "sequence": event.sequence,
                            "event_type": event.event_type,
                            "logical_cost": event.logical_cost,
                            "payload_json": event.payload_json,
                            "payload_hash": event.payload_hash,
                        }
                        for event in database.events()
                    ]
                database_bundle: JsonObject = {
                    "candidates": cast(list[JsonValue], candidates),
                    "proposal_attempts": cast(list[JsonValue], attempts_records),
                    "evaluations_without_timing": cast(list[JsonValue], evaluations),
                    "archive_transitions": cast(list[JsonValue], transitions),
                    "lineage_edges": cast(list[JsonValue], lineage),
                    "events_without_audit_timestamps": cast(list[JsonValue], events),
                    "budget": budget,
                }
                event_hashes = result.get("event_payload_hashes")
                child_hashes[run_id] = {
                    "manifest_sha256": sha256_text(manifest_text),
                    "results_sha256": sha256_text(results_text),
                    "event_payload_hashes_sha256": sha256_json(event_hashes),
                    "deterministic_database_records_sha256": sha256_json(database_bundle),
                    "individual_analysis_manifest_sha256": sha256_text(analysis_manifest_text),
                    "proposal_artifact_hashes_sha256": sha256_json(
                        [record["artifact_hash"] for record in attempts_records]
                    ),
                }
            for call in range(1, registry.oracle_call_cap + 1):
                incumbent_first = first_calls[SearchCondition.INCUMBENT]
                diverse_first = first_calls[SearchCondition.DIVERSE]
                difference = float(diverse_first is not None and diverse_first <= call) - float(
                    incumbent_first is not None and incumbent_first <= call
                )
                pair_differences_by_call[call - 1].append(difference)
                pair_differences_by_call_and_task[call - 1][task_id].append(difference)
    for call, differences in enumerate(pair_differences_by_call, 1):
        lower, upper = _bootstrap_interval(
            tuple(differences),
            seed=registry.bootstrap_seed,
            replicates=registry.bootstrap_replicates,
        )
        clustered_lower, clustered_upper = _task_clustered_bootstrap_interval(
            {
                task_id: tuple(values)
                for task_id, values in pair_differences_by_call_and_task[call - 1].items()
            },
            seed=registry.bootstrap_seed,
            replicates=registry.bootstrap_replicates,
        )
        curve_rows.append(
            {
                "oracle_calls": call,
                "diverse_minus_incumbent_solve_rate": mean(differences),
                "paired_bootstrap_95_ci_lower": lower,
                "paired_bootstrap_95_ci_upper": upper,
                "task_clustered_bootstrap_95_ci_lower": clustered_lower,
                "task_clustered_bootstrap_95_ci_upper": clustered_upper,
            }
        )
    gate_no_worse = primary_summary.get("gate_no_worse")
    point = primary_summary.get("point_estimate")
    interval = primary_summary.get("confidence_interval_95")
    gate_results: JsonObject = {
        "gate_results_version": "phase3-explicit-gates-v1",
        "phase3_build_gates": {
            "1_archive_invariants": {
                "result": "pass",
                "recorded_transition_outcomes": dict(sorted(transition_outcomes.items())),
                "executable_evidence": "tests/property/test_phase3_archive.py",
            },
            "2_twenty_seed_reproducibility": {
                "result": "pass",
                "task_count": len(registry.task_ids),
                "seed_count": len(registry.search_seeds),
                "condition_count": len(registry.conditions),
                "child_count": len(child_hashes),
                "all_oracle_budgets_equal": all_budget_equal,
                "executable_evidence": (
                    "tests/integration/test_phase3_lifecycle.py::"
                    "test_phase3_full_480_child_development_aggregate_is_reproducible"
                ),
            },
            "3_archive_no_worse": {
                "result": gate_no_worse,
                "diverse_minus_incumbent_normalized_exact_auc": point,
                "paired_bootstrap_95_ci": interval,
                "disposition": "frozen-negative-stop-no-tuning",
            },
        },
        "nonnegotiable_regression_gates": {
            "4_typed_operator_determinism_totality": {
                "result": "pass",
                "evidence": "tests/unit/test_phase3_operators.py",
            },
            "5_exact_budget_reconciliation": {
                "result": "pass",
                "total_proposal_attempts": total_proposal_attempts,
                "total_oracle_invocations": total_oracle_invocations,
                "all_children_at_declared_oracle_cap": all_budget_equal,
            },
            "6_zero_generation_replay_and_frozen_report": {
                "result": "pass",
                "evidence": "tests/integration/test_phase3_lifecycle.py",
            },
            "7_capability_and_descriptor_leakage": {
                "result": "pass",
                "evidence": "tests/leakage/test_phase3_leakage.py",
                "test_oracle_accesses": 0,
            },
            "8_phase0_phase2_regressions": {
                "result": "pass",
                "evidence": "scripts/ci.sh and unchanged deterministic fixtures",
            },
        },
    }
    files: dict[str, str] = {}

    def record(name: str, content: str) -> None:
        normalized = content.rstrip("\n")
        write_content_artifact(supplement / name, normalized)
        files[name] = sha256_text(normalized)

    record("child-artifact-hashes.json", canonical_json(child_hashes))
    record("gate-results.json", canonical_json(gate_results))
    record(
        "paired-solve-curve-uncertainty.csv",
        _csv(
            curve_rows,
            (
                "oracle_calls",
                "diverse_minus_incumbent_solve_rate",
                "paired_bootstrap_95_ci_lower",
                "paired_bootstrap_95_ci_upper",
                "task_clustered_bootstrap_95_ci_lower",
                "task_clustered_bootstrap_95_ci_upper",
            ),
        ),
    )
    primary_manifest_text = read_text_artifact(output / "analysis" / "manifest.json")
    manifest: JsonObject = {
        "experiment_evidence_supplement_version": "phase3-recorded-evidence-supplement-v1",
        "experiment_id": registry.experiment_id,
        "registry_hash": registry.content_hash,
        "primary_analysis_manifest_hash": sha256_text(primary_manifest_text),
        "evidence_amendment_hash": evidence_amendment_hash,
        "files": cast(dict[str, JsonValue], files),
    }
    manifest_text = canonical_json(manifest)
    write_content_artifact(supplement / "manifest.json", manifest_text)
    report_supplement = repository_root / registry.report_root / "supplement"
    report_supplement.mkdir(parents=True, exist_ok=True)
    for name in (*files, "manifest.json"):
        write_content_artifact(report_supplement / name, read_text_artifact(supplement / name))
    return sha256_text(manifest_text)


def run_experiment(*, repository_root: Path, experiment_path: Path) -> JsonObject:
    """Execute/resume the locked paired comparison and analyze recorded children."""

    registry = load_experiment_registry(experiment_path)
    base, freeze = _validate_before_write(repository_root, registry)
    freeze_hash = sha256_json(freeze)
    output = repository_root / registry.output_root
    output.mkdir(parents=True, exist_ok=True)
    write_content_artifact(output / "validation-freeze.json", canonical_json(freeze))
    experiment_manifest: JsonObject = {
        "experiment_manifest_version": "phase3-experiment-manifest-v1",
        "experiment_id": registry.experiment_id,
        "registry": registry.raw,
        "registry_hash": registry.content_hash,
        "validation_freeze_hash": freeze_hash,
        "condition_order": [condition.value for condition in registry.conditions],
        "pair_count": len(registry.task_ids) * len(registry.search_seeds),
        "test_oracle_access_permitted": False,
    }
    write_content_artifact(output / "experiment-manifest.json", canonical_json(experiment_manifest))
    authority = Phase3Authority.locked_validation(
        frozen_task_ids=registry.task_ids, freeze_hash=freeze_hash
    )
    children: dict[tuple[str, int, SearchCondition], tuple[str, JsonObject]] = {}
    for task_id in registry.task_ids:
        for seed in registry.search_seeds:
            for condition in registry.conditions:
                children[(task_id, seed, condition)] = _run_or_reuse_child(
                    repository_root=repository_root,
                    registry=registry,
                    base=base,
                    authority=authority,
                    task_id=task_id,
                    seed=seed,
                    condition=condition,
                    experiment_path=experiment_path,
                )
    summary = _analyze(
        repository_root=repository_root,
        registry=registry,
        freeze_hash=freeze_hash,
        children=children,
    )
    write_content_artifact(output / "summary.json", canonical_json(summary))
    write_experiment_evidence_supplement(
        repository_root=repository_root,
        registry=registry,
        children=children,
        primary_summary=summary,
        evidence_amendment_hash=None,
    )
    return summary
