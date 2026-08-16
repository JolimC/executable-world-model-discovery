"""Frozen offline source import and matched experiment runner for contextual memory v3."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import yaml

from world_model_search.config import AppConfig, config_from_mapping, load_config
from world_model_search.domain.types import Candidate, OracleResult, SplitLabel
from world_model_search.dsl.ast import AstLimits, BitExpr
from world_model_search.dsl.json_schema import DslCandidateDocument
from world_model_search.errors import ConfigurationError, PersistenceError
from world_model_search.memory.contextual import (
    ContextFeatureExtractor,
    ContextMode,
    DownstreamOutcomeAnnotator,
    ExperienceRecord,
    RetrievalMode,
    create_experience_record,
    extract_ast_delta,
)
from world_model_search.memory.contextual_retrieval import (
    ContextualExperienceRuntime,
    ContextualMemoryConfig,
    ExperienceSnapshot,
    RandomizedExposureConfig,
    SimilarityWeights,
)
from world_model_search.model.contextual_prompts import assert_contextual_prompt_isolation
from world_model_search.persistence.artifacts import read_text_artifact, write_content_artifact
from world_model_search.persistence.phase4_database import Phase4Database
from world_model_search.search.archive import (
    ArchiveCoordinate,
    ArchiveLayer,
    InsertionOutcome,
    RepresentationFamily,
    descriptor,
)
from world_model_search.search.loop import load_manifest
from world_model_search.search.phase4 import (
    Phase4Outcome,
    resume_phase4_run,
    start_phase4_run,
)
from world_model_search.serialization import (
    JsonObject,
    canonical_json,
    parse_json_object,
    sha256_bytes,
    sha256_json,
)
from world_model_search.tasks import benchmark_root_for_config, load_public_task

CONTEXTUAL_EXPERIMENT_SCHEMA = 1
CONTEXTUAL_ANALYSIS_SCHEMA = "phase5-contextual-experiment-analysis-v1"
FROZEN_ARMS = (
    "A-no-memory",
    "B-positive-rich",
    "C-contrastive-rich",
    "D-family-only",
    "D-rich-context",
)
ARM_RUN_CODES = {
    "A-no-memory": "A",
    "B-positive-rich": "B",
    "C-contrastive-rich": "C",
    "D-family-only": "DF",
    "D-rich-context": "DR",
}


def _mapping(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{label} must be a mapping")
    if set(value) != keys:
        raise ConfigurationError(f"{label} has missing or unknown fields")
    return value


def _path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{label} must be a repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ConfigurationError(f"{label} must be a specific repository-relative path")
    return path


def _strings(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ConfigurationError(f"{label} must contain unique nonempty strings")
    return tuple(cast(list[str], value))


def _integers(value: object, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ConfigurationError(f"{label} must contain unique integers")
    return tuple(cast(list[int], value))


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ContextualExperimentRegistry:
    experiment_id: str
    status: str
    base_config: Path
    source_run_root: Path
    source_run_suffix: str
    expected_source_task_ids: tuple[str, ...]
    snapshot_path: Path
    target_split: SplitLabel
    target_task_ids: tuple[str, ...]
    search_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    output_root: Path
    runs_root: Path
    memory_config: ContextualMemoryConfig
    raw: JsonObject

    @property
    def registry_hash(self) -> str:
        return sha256_json(self.raw)

    def source_run_directories(self, repository_root: Path) -> tuple[Path, ...]:
        root = repository_root / self.source_run_root
        runs = tuple(
            sorted(
                path
                for path in root.iterdir()
                if path.is_dir() and path.name.endswith(self.source_run_suffix)
            )
        )
        if not runs:
            raise ConfigurationError("contextual source run selection is empty")
        return runs


def load_contextual_experiment_registry(path: Path) -> ContextualExperimentRegistry:
    try:
        loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read contextual experiment registry: {exc}") from exc
    root = _mapping(
        loaded,
        {
            "experiment_schema_version",
            "experiment_id",
            "status",
            "base_config",
            "source",
            "snapshot_path",
            "target",
            "search_seeds",
            "arms",
            "output_root",
            "runs_root",
            "memory",
            "matched_contract",
        },
        "contextual experiment",
    )
    if root["experiment_schema_version"] != CONTEXTUAL_EXPERIMENT_SCHEMA:
        raise ConfigurationError("unsupported contextual experiment schema")
    experiment_id = str(root["experiment_id"])
    status = str(root["status"])
    if not experiment_id or status not in {"provider-free-smoke", "controlled-development"}:
        raise ConfigurationError("contextual experiment identity/status is invalid")
    base_path = _path(root["base_config"], "base_config")
    base = load_config(base_path)
    if base.schema_version != 4 or base.phase4_budget is None:
        raise ConfigurationError("contextual experiment requires a Phase 4 base config")
    source = _mapping(
        root["source"],
        {"run_root", "run_suffix", "expected_condition", "expected_task_ids"},
        "source",
    )
    if source["expected_condition"] != "uniform-diverse-archive-v1":
        raise ConfigurationError("contextual source must be Phase 4 condition C")
    suffix = str(source["run_suffix"])
    if not suffix or "/" in suffix:
        raise ConfigurationError("source run suffix is invalid")
    target = _mapping(root["target"], {"split", "task_ids"}, "target")
    try:
        target_split = SplitLabel(str(target["split"]))
    except ValueError as exc:
        raise ConfigurationError("contextual target split is invalid") from exc
    task_ids = tuple(sorted(_strings(target["task_ids"], "target.task_ids")))
    source_task_ids = tuple(
        sorted(_strings(source["expected_task_ids"], "source.expected_task_ids"))
    )
    if set(task_ids) & set(source_task_ids):
        raise ConfigurationError("contextual source and target task IDs overlap")
    arms = _strings(root["arms"], "arms")
    if arms != FROZEN_ARMS:
        raise ConfigurationError("contextual experiment arms/order differ from A/B/C/D")
    memory = _mapping(
        root["memory"],
        {
            "minimum_similarity_ppm",
            "max_memory_records",
            "max_memory_bytes",
            "max_memory_tokens_conservative",
            "include_diverse_third",
            "short_horizon_steps",
            "weights",
            "randomized_exposure",
        },
        "memory",
    )
    weights_raw = _mapping(memory["weights"], set(SimilarityWeights().to_value()), "weights")
    weights = SimilarityWeights(
        **{key: _integer(value, f"weights.{key}") for key, value in weights_raw.items()}
    )
    exposure_raw = _mapping(
        memory["randomized_exposure"],
        {"enabled", "probability_numerator", "probability_denominator", "randomization_seed"},
        "randomized_exposure",
    )
    if not isinstance(exposure_raw["enabled"], bool):
        raise ConfigurationError("randomized_exposure.enabled must be Boolean")
    exposure = RandomizedExposureConfig(
        exposure_raw["enabled"],
        _integer(exposure_raw["probability_numerator"], "probability_numerator"),
        _integer(exposure_raw["probability_denominator"], "probability_denominator", minimum=1),
        _integer(exposure_raw["randomization_seed"], "randomization_seed"),
    )
    if not isinstance(memory["include_diverse_third"], bool):
        raise ConfigurationError("include_diverse_third must be Boolean")
    config = ContextualMemoryConfig(
        minimum_similarity_ppm=_integer(memory["minimum_similarity_ppm"], "minimum_similarity_ppm"),
        max_memory_records=_integer(memory["max_memory_records"], "max_memory_records", minimum=1),
        max_memory_bytes=_integer(memory["max_memory_bytes"], "max_memory_bytes", minimum=1),
        max_memory_tokens_conservative=_integer(
            memory["max_memory_tokens_conservative"],
            "max_memory_tokens_conservative",
            minimum=1,
        ),
        include_diverse_third=memory["include_diverse_third"],
        short_horizon_steps=_integer(
            memory["short_horizon_steps"], "short_horizon_steps", minimum=1
        ),
        weights=weights,
        exposure=exposure,
    )
    contract = _mapping(
        root["matched_contract"],
        {"condition", "scheduler", "sole_arm_difference", "sealed_results_reusable"},
        "matched_contract",
    )
    expected_contract = {
        "condition": "uniform-diverse-archive-v1",
        "scheduler": "uniform-sorted-branches-v1",
        "sole_arm_difference": "canonical-cross-task-memory-block-v1",
        "sealed_results_reusable": False,
    }
    if contract != expected_contract:
        raise ConfigurationError("contextual matched contract differs from the required isolation")
    raw = cast(JsonObject, json.loads(canonical_json(root)))
    return ContextualExperimentRegistry(
        experiment_id,
        status,
        base_path,
        _path(source["run_root"], "source.run_root"),
        suffix,
        source_task_ids,
        _path(root["snapshot_path"], "snapshot_path"),
        target_split,
        task_ids,
        _integers(root["search_seeds"], "search_seeds"),
        arms,
        _path(root["output_root"], "output_root"),
        _path(root["runs_root"], "runs_root"),
        config,
        raw,
    )


def _result(data: str) -> OracleResult:
    raw = parse_json_object(data)
    response = raw.get("response")
    if not isinstance(response, dict):
        raise PersistenceError("source result feedback is malformed")
    from world_model_search.domain.types import OracleFeedback, OracleResponseMode

    summary = response.get("summary")
    if not isinstance(summary, list):
        raise PersistenceError("source result summary is malformed")
    return OracleResult(
        bool(raw["type_valid"]),
        bool(raw["total"]),
        _integer(raw["local_errors"], "local_errors"),
        _integer(raw["local_cases"], "local_cases"),
        bool(raw["rollout_pass"]),
        bool(raw["exact"]),
        _integer(raw["ast_bits"], "ast_bits"),
        _integer(raw["residual_bits"], "residual_bits"),
        _integer(raw["runtime_ns"], "runtime_ns"),
        OracleFeedback(
            OracleResponseMode(str(response["mode"])),
            tuple(str(item) for item in summary),
            str(response["counterexample"]) if response.get("counterexample") is not None else None,
        ),
    )


def _candidate(row: object, limits: AstLimits) -> Candidate:
    mapping = dict(cast(Any, row))
    document = DslCandidateDocument.from_json(
        canonical_json(
            {
                "candidate_schema_version": 1,
                "dsl_version": "binary-ca-radius1-dsl-v1",
                "ast": json.loads(str(mapping["canonical_ast_json"])),
            }
        ),
        limits=limits,
    )
    parents = json.loads(str(mapping["parent_ids_json"]))
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise PersistenceError("source candidate parents are malformed")
    return Candidate(
        str(mapping["candidate_id"]),
        str(mapping["task_id"]),
        document.ast,
        tuple(parents),
        str(mapping["proposer_id"]),
        str(mapping["operator_id"]),
        str(mapping["context_hash"]),
        str(mapping["payload_hash"]),
        str(mapping["semantic_hash"]),
    )


def _coordinate(value: object) -> ArchiveCoordinate:
    if not isinstance(value, dict):
        raise PersistenceError("source archive coordinate is malformed")
    return ArchiveCoordinate(
        str(value["size_bin"]),
        RepresentationFamily(str(value["representation_family"])),
        str(value["error_signature_cluster"]),
        ArchiveLayer(str(value["layer"])),
    )


def _lineage_context(
    *,
    parent: Candidate,
    candidates: dict[str, Candidate],
    results: dict[str, OracleResult],
    task: object,
    search_step: int,
) -> tuple[int, int, tuple[str, ...]]:
    lineage_depth = 0
    plateau = 0
    plateau_open = True
    recent: list[str] = []
    current = parent
    current_errors = results[parent.candidate_id].local_errors
    seen: set[str] = set()
    while current.parent_ids and current.candidate_id not in seen:
        seen.add(current.candidate_id)
        ancestor = candidates[current.parent_ids[0]]
        if not isinstance(ancestor.ast, BitExpr) or not isinstance(current.ast, BitExpr):
            raise PersistenceError("source lineage AST is untyped")
        lineage_depth += 1
        if len(recent) < 8:
            recent.extend(
                item.value for item in extract_ast_delta(ancestor.ast, current.ast).edit_classes
            )
        if plateau_open and results[ancestor.candidate_id].local_errors == current_errors:
            plateau += 1
        else:
            plateau_open = False
        current = ancestor
    return lineage_depth, plateau, tuple(recent[:8])


def extract_contextual_records_from_run(
    *, repository_root: Path, run_directory: Path
) -> tuple[ExperienceRecord, ...]:
    """Retrospectively import every valid evaluated proposal without provider/oracle access."""

    manifest = load_manifest(run_directory)
    config_raw = manifest.get("resolved_configuration")
    config = config_from_mapping(config_raw)
    if config.dsl is None or config.run.condition_id != "uniform-diverse-archive-v1":
        raise ConfigurationError("contextual source run must use Phase 4 condition C")
    limits = AstLimits(config.dsl.max_depth, config.dsl.max_nodes, config.dsl.max_cases)
    task = load_public_task(
        benchmark_root_for_config(repository_root, config),
        config.run.task_id,
    )
    public_task = task.public_view()
    with Phase4Database(run_directory / "run.sqlite3", read_only=True) as database:
        candidate_rows = database.candidates()
        candidates = {str(row["candidate_id"]): _candidate(row, limits) for row in candidate_rows}
        evaluation_rows = database.evaluations()
        results = {
            str(row["candidate_id"]): _result(str(row["result_json"])) for row in evaluation_rows
        }
        evaluation_index: dict[str, int] = {}
        for row in evaluation_rows:
            evaluation_index.setdefault(
                str(row["candidate_id"]),
                int(row["evaluation_index"]),
            )
        item_rows = {
            (int(row["request_index"]), int(row["ordinal"])): row for row in database.items()
        }
        transitions = {
            int(row["evaluation_index"]): parse_json_object(str(row["decision_json"]))
            for row in database.transitions()
        }
        request_first_evaluation: dict[int, int] = {}
        for row in evaluation_rows:
            if row["request_index"] is not None:
                request_index = int(row["request_index"])
                request_first_evaluation[request_index] = min(
                    request_first_evaluation.get(request_index, int(row["evaluation_index"])),
                    int(row["evaluation_index"]),
                )
        task_row = database.connection.execute("SELECT * FROM task").fetchone()
        if task_row is None:
            raise PersistenceError("source run has no task record")
        records: list[ExperienceRecord] = []
        for row in evaluation_rows:
            if row["request_index"] is None or row["item_ordinal"] is None:
                continue
            child = candidates[str(row["candidate_id"])]
            if not child.parent_ids:
                continue
            parent = candidates[child.parent_ids[0]]
            if not isinstance(parent.ast, BitExpr):
                raise PersistenceError("source parent is untyped")
            request_index = int(row["request_index"])
            ordinal = int(row["item_ordinal"])
            depth, plateau, recent = _lineage_context(
                parent=parent,
                candidates=candidates,
                results=results,
                task=public_task,
                search_step=request_first_evaluation[request_index],
            )
            parent_coordinate = descriptor(parent.ast, results[parent.candidate_id], public_task)
            context = ContextFeatureExtractor().extract(
                parent=parent.ast,
                result=results[parent.candidate_id],
                task=public_task,
                coordinate=parent_coordinate,
                search_step=request_first_evaluation[request_index],
                lineage_depth=depth,
                plateau_length=plateau,
                recent_edit_classes=recent,
            )
            transition = transitions[int(row["evaluation_index"])]
            coordinate = _coordinate(transition["coordinate"])
            item = item_rows[(request_index, ordinal)]
            record = create_experience_record(
                task_generator_family=str(task_row["internal_family_id"]),
                task_split=SplitLabel(str(task_row["split"])),
                run_id=str(manifest["run_id"]),
                search_seed=config.run.seed,
                request_index=request_index,
                item_ordinal=ordinal,
                evaluation_index=int(row["evaluation_index"]),
                parent=parent,
                parent_result=results[parent.candidate_id],
                parent_context=context,
                child=child,
                child_result=results[child.candidate_id],
                child_coordinate=coordinate,
                archive_outcome=InsertionOutcome(str(transition["outcome"])),
                canonical_duplicate=bool(item["canonical_duplicate"]),
                semantic_duplicate=bool(item["semantic_duplicate"]),
                sealed_test=SplitLabel(str(task_row["split"])) is SplitLabel.TEST,
                evidence_timing="retrospective",
            )
            records.append(record)
        parents_by_candidate = {
            candidate_id: candidate.parent_ids for candidate_id, candidate in candidates.items()
        }
        scores = {candidate_id: 8 - result.local_errors for candidate_id, result in results.items()}
        exact = {candidate_id: result.exact for candidate_id, result in results.items()}
        return DownstreamOutcomeAnnotator().annotate(
            tuple(records),
            parent_ids=parents_by_candidate,
            score_by_candidate=scores,
            exact_by_candidate=exact,
            evaluation_by_candidate=evaluation_index,
        )


def freeze_contextual_snapshot(
    *, repository_root: Path, registry: ContextualExperimentRegistry
) -> JsonObject:
    records: list[ExperienceRecord] = []
    source_hashes: list[str] = []
    observed_tasks: set[str] = set()
    runs = registry.source_run_directories(repository_root)
    for run in runs:
        extracted = extract_contextual_records_from_run(
            repository_root=repository_root,
            run_directory=run,
        )
        records.extend(extracted)
        observed_tasks.update(record.provenance.task_id for record in extracted)
        source_hashes.append(sha256_bytes((run / "run.sqlite3").read_bytes()))
    if observed_tasks != set(registry.expected_source_task_ids):
        raise ConfigurationError("observed contextual source tasks differ from the frozen registry")
    snapshot = ExperienceSnapshot(
        tuple(sorted(observed_tasks)),
        tuple(sorted(records, key=lambda item: item.record_id)),
        tuple(sorted(source_hashes)),
    )
    snapshot_hash = snapshot.write(repository_root / registry.snapshot_path)
    summary: JsonObject = {
        "schema_version": "contextual-snapshot-freeze-summary-v1",
        "registry_hash": registry.registry_hash,
        "snapshot_hash": snapshot.snapshot_hash,
        "artifact_hash": snapshot_hash,
        "source_run_count": len(runs),
        "source_task_count": len(observed_tasks),
        "record_count": len(snapshot.records),
        "positive_count": sum(
            record.immediate_outcome.score_delta > 0 for record in snapshot.records
        ),
        "neutral_count": sum(
            record.immediate_outcome.score_delta == 0 for record in snapshot.records
        ),
        "regression_count": sum(
            record.immediate_outcome.score_delta < 0 for record in snapshot.records
        ),
    }
    write_content_artifact(
        repository_root / registry.output_root / "snapshot-freeze-summary.json",
        canonical_json(summary),
    )
    return summary


def _arm_runtime(
    *,
    arm: str,
    snapshot: ExperienceSnapshot,
    base: ContextualMemoryConfig,
    target_task_ids: tuple[str, ...],
) -> ContextualExperienceRuntime:
    if arm == "A-no-memory":
        config = replace(base, retrieval_mode=RetrievalMode.DISABLED, context_mode=ContextMode.RICH)
        source: ExperienceSnapshot | None = None
    elif arm in {"B-positive-rich", "D-rich-context"}:
        config = replace(
            base, retrieval_mode=RetrievalMode.POSITIVE_ONLY, context_mode=ContextMode.RICH
        )
        source = snapshot
    elif arm == "C-contrastive-rich":
        config = replace(
            base, retrieval_mode=RetrievalMode.CONTRASTIVE, context_mode=ContextMode.RICH
        )
        source = snapshot
    elif arm == "D-family-only":
        config = replace(
            base,
            retrieval_mode=RetrievalMode.POSITIVE_ONLY,
            context_mode=ContextMode.FAMILY_ONLY,
        )
        source = snapshot
    else:
        raise ConfigurationError(f"unknown contextual experiment arm: {arm}")
    return ContextualExperienceRuntime(arm, source, config, target_task_ids)


def _child_config(
    base: AppConfig,
    *,
    registry: ContextualExperimentRegistry,
    task_id: str,
    seed: int,
) -> AppConfig:
    if base.cache is None:
        raise ConfigurationError("contextual base config has no exact cache")
    return replace(
        base,
        run=replace(
            base.run,
            root=registry.runs_root,
            seed=seed,
            task_id=task_id,
            split=registry.target_split,
            condition_id="uniform-diverse-archive-v1",
        ),
        cache=replace(base.cache, namespace=f"{base.cache.namespace}-contextual-v3"),
    )


def _child_run_id(*, task_id: str, seed: int, arm: str) -> str:
    try:
        code = ARM_RUN_CODES[arm]
    except KeyError as exc:
        raise ConfigurationError(f"unknown contextual experiment arm: {arm}") from exc
    return f"p5cv3-{task_id}-s{seed}-{code}"


def contextual_experiment_dry_run(
    *, repository_root: Path, registry: ContextualExperimentRegistry
) -> JsonObject:
    snapshot = ExperienceSnapshot.read(repository_root / registry.snapshot_path)
    if set(snapshot.source_task_ids) != set(registry.expected_source_task_ids):
        raise ConfigurationError("frozen snapshot source tasks differ from the registry")
    base = load_config(repository_root / registry.base_config)
    children = len(registry.target_task_ids) * len(registry.search_seeds) * len(registry.arms)
    return {
        "schema_version": "contextual-experiment-dry-run-v1",
        "experiment_id": registry.experiment_id,
        "registry_hash": registry.registry_hash,
        "snapshot_hash": snapshot.snapshot_hash,
        "child_run_count": children,
        "task_seed_pair_count": len(registry.target_task_ids) * len(registry.search_seeds),
        "arms": list(registry.arms),
        "provider_id": base.model.provider_id if base.model else None,
        "new_provider_calls": 0,
        "scheduler": "uniform-sorted-branches-v1-unchanged",
    }


def _first_prompt(run_directory: Path) -> str | None:
    prompts = sorted((run_directory / "prompts").glob("request-*.json"))
    if not prompts:
        return None
    return read_text_artifact(prompts[0])


def _check_pair_prompt_isolation(control: Path, treatment: Path) -> bool:
    control_prompt = _first_prompt(control)
    treatment_prompt = _first_prompt(treatment)
    if control_prompt is None or treatment_prompt is None:
        return False
    assert_contextual_prompt_isolation(control_prompt, treatment_prompt)
    return True


def _analyze_experiment(
    *,
    repository_root: Path,
    registry: ContextualExperimentRegistry,
    outcomes: list[tuple[str, str, int, Phase4Outcome]],
) -> JsonObject:
    rows: list[JsonObject] = []
    by_pair: dict[tuple[str, int], dict[str, Path]] = {}
    for arm, task_id, seed, outcome in outcomes:
        result = parse_json_object(read_text_artifact(outcome.run_directory / "results.json"))
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise PersistenceError("contextual child results have no metrics")
        retrievals = sorted((outcome.run_directory / "experience_v3" / "retrieval").glob("*.json"))
        shown = 0
        eligible = 0
        for path in retrievals:
            audit = parse_json_object(read_text_artifact(path))
            retrieval = audit.get("retrieval")
            rendered = audit.get("rendered_memory")
            eligible_values = (
                retrieval.get("all_eligible_scores") if isinstance(retrieval, dict) else None
            )
            shown_values = rendered.get("shown_record_ids") if isinstance(rendered, dict) else None
            if isinstance(eligible_values, list):
                eligible += len(eligible_values)
            if isinstance(shown_values, list):
                shown += len(shown_values)
        rows.append(
            {
                "schema_version": CONTEXTUAL_ANALYSIS_SCHEMA,
                "arm": arm,
                "task_id": task_id,
                "search_seed": seed,
                "run_id": outcome.run_id,
                "status": outcome.status,
                "normalized_exact_auc": metrics["normalized_exact_auc"],
                "final_exact_solved": metrics["final_exact_solved"],
                "calls_to_first_exact": metrics["calls_to_first_exact"],
                "archive_coverage": metrics["archive_coverage"],
                "eligible_memory_count_across_retrievals": eligible,
                "shown_memory_count_across_retrievals": shown,
            }
        )
        by_pair.setdefault((task_id, seed), {})[arm] = outcome.run_directory
    prompt_isolation_comparisons = 0
    prompt_isolation_skipped_no_prompt = 0
    for pair, directories in by_pair.items():
        control = directories.get("A-no-memory")
        if control is None:
            raise PersistenceError(f"contextual pair has no control: {pair}")
        for arm in registry.arms[1:]:
            treatment = directories.get(arm)
            if treatment is None:
                raise PersistenceError(f"contextual pair has no treatment: {pair} {arm}")
            if _check_pair_prompt_isolation(control, treatment):
                prompt_isolation_comparisons += 1
            else:
                prompt_isolation_skipped_no_prompt += 1
    root = repository_root / registry.output_root / "analysis"
    jsonl = "\n".join(canonical_json(row) for row in rows)
    jsonl_hash = write_content_artifact(root / "raw-rows.jsonl", jsonl)
    output = io.StringIO(newline="")
    fieldnames = tuple(rows[0]) if rows else ("schema_version",)
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_hash = write_content_artifact(root / "raw-rows.csv", output.getvalue().rstrip("\n"))
    summary: JsonObject = {
        "schema_version": CONTEXTUAL_ANALYSIS_SCHEMA,
        "experiment_id": registry.experiment_id,
        "registry_hash": registry.registry_hash,
        "child_run_count": len(rows),
        "matched_task_seed_pair_count": len(by_pair),
        "arms": list(registry.arms),
        "prompt_isolation_checked": prompt_isolation_comparisons > 0,
        "prompt_isolation_comparison_count": prompt_isolation_comparisons,
        "prompt_isolation_skipped_no_prompt_count": prompt_isolation_skipped_no_prompt,
        "raw_jsonl_hash": jsonl_hash,
        "raw_csv_hash": csv_hash,
    }
    write_content_artifact(root / "summary.json", canonical_json(summary))
    human = "\n".join(
        (
            "# Contextual memory experiment",
            "",
            f"- Matched task-seed pairs: {len(by_pair)}",
            f"- Child runs: {len(rows)}",
            "- Arms: " + ", ".join(registry.arms),
            (
                "- First-request prompt isolation: passed for "
                f"{prompt_isolation_comparisons} comparisons; "
                f"{prompt_isolation_skipped_no_prompt} skipped because a budget-capped child "
                "had no prompt"
            ),
            "",
            "Downstream and retrieval associations are descriptive unless exposure was randomized.",
        )
    )
    write_content_artifact(root / "summary.md", human)
    return summary


def run_contextual_experiment(
    *,
    repository_root: Path,
    registry: ContextualExperimentRegistry,
    allow_live_model: bool = False,
) -> JsonObject:
    base = load_config(repository_root / registry.base_config)
    if base.model is None:
        raise ConfigurationError("contextual experiment base config has no model")
    if base.model.provider_id == "openai" and not allow_live_model:
        raise ConfigurationError(
            "live contextual experiment requires explicit provider authorization"
        )
    snapshot = ExperienceSnapshot.read(repository_root / registry.snapshot_path)
    target_ids = tuple(sorted(registry.target_task_ids))
    outcomes: list[tuple[str, str, int, Phase4Outcome]] = []
    for task_index, task_id in enumerate(registry.target_task_ids):
        for seed_index, seed in enumerate(registry.search_seeds):
            offset = (task_index + seed_index) % len(registry.arms)
            arm_order = (*registry.arms[offset:], *registry.arms[:offset])
            for arm in arm_order:
                runtime = _arm_runtime(
                    arm=arm,
                    snapshot=snapshot,
                    base=registry.memory_config,
                    target_task_ids=target_ids,
                )
                child = _child_config(
                    base,
                    registry=registry,
                    task_id=task_id,
                    seed=seed,
                )
                run_id = _child_run_id(task_id=task_id, seed=seed, arm=arm)
                run_directory = repository_root / registry.runs_root / run_id
                legacy_run_id = f"{registry.experiment_id}-{task_id}-s{seed}-{arm}"
                legacy_directory = repository_root / registry.runs_root / legacy_run_id
                if not run_directory.exists() and legacy_directory.exists():
                    run_directory = legacy_directory
                if run_directory.exists():
                    outcome = resume_phase4_run(
                        repository_root=repository_root,
                        run_directory=run_directory,
                        config=child,
                        manifest=load_manifest(run_directory),
                        interrupt_after=None,
                        allow_live_model=allow_live_model,
                        contextual_experience_runtime=runtime,
                    )
                else:
                    outcome = start_phase4_run(
                        repository_root=repository_root,
                        config=child,
                        config_source=str(registry.base_config),
                        run_id=run_id,
                        interrupt_after=None,
                        allow_live_model=allow_live_model,
                        contextual_experience_runtime=runtime,
                    )
                outcomes.append((arm, task_id, seed, outcome))
    return _analyze_experiment(
        repository_root=repository_root,
        registry=registry,
        outcomes=outcomes,
    )
