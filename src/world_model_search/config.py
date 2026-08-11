"""Strict YAML configuration loading with persistence-free validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from world_model_search.domain.types import OracleResponseMode, SplitLabel
from world_model_search.dsl.versions import (
    DSL_VERSION,
    ENUMERATOR_VERSION,
    PHASE3_ANALYSIS_VERSION,
    PHASE3_ARCHIVE_VERSION,
    PHASE3_BUDGET_VERSION,
    PHASE3_DESCRIPTOR_VERSION,
    PHASE3_INCUMBENT_VERSION,
    PHASE3_INITIALIZATION_VERSION,
    PHASE3_OPERATOR_VERSION,
    PHASE3_SCHEDULER_VERSION,
)
from world_model_search.errors import ConfigurationError
from world_model_search.serialization import JsonObject, sha256_json, to_json_value


def _expect_mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _require_keys(mapping: dict[str, object], expected: set[str], location: str) -> None:
    missing = expected - mapping.keys()
    unknown = mapping.keys() - expected
    if missing:
        raise ConfigurationError(f"{location} is missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigurationError(f"{location} has unknown keys: {', '.join(sorted(unknown))}")


def _expect_str(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty string")
    return value


def _expect_int(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{location} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class RunSettings:
    root: Path
    seed: int
    max_steps: int
    task_id: str
    split: SplitLabel
    condition_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProposerSettings:
    proposer_id: str
    batch_size: int


@dataclass(frozen=True, slots=True)
class OracleSettings:
    oracle_id: str
    response_mode: OracleResponseMode = OracleResponseMode.SCORE_ONLY


@dataclass(frozen=True, slots=True)
class DslSettings:
    dsl_version: str
    max_depth: int
    max_nodes: int
    max_cases: int
    allowed_macros: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnumeratorSettings:
    enumerator_version: str
    max_bits: int
    max_depth: int
    max_nodes: int
    max_candidates: int


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str


@dataclass(frozen=True, slots=True)
class OperatorSettings:
    operator_version: str
    weights: tuple[tuple[str, int], ...]
    retry_limit: int
    fallback_policy: str


@dataclass(frozen=True, slots=True)
class ArchiveSettings:
    archive_version: str
    descriptor_version: str
    reserve_size: int


@dataclass(frozen=True, slots=True)
class SchedulerSettings:
    scheduler_version: str


@dataclass(frozen=True, slots=True)
class BudgetSettings:
    budget_version: str
    proposal_attempt_cap: int
    oracle_call_cap: int


@dataclass(frozen=True, slots=True)
class InitializationSettings:
    initialization_version: str


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    analysis_version: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    schema_version: int
    run: RunSettings
    proposer: ProposerSettings
    oracle: OracleSettings
    logging: LoggingSettings
    dsl: DslSettings | None = None
    enumerator: EnumeratorSettings | None = None
    operators: OperatorSettings | None = None
    archive: ArchiveSettings | None = None
    scheduler: SchedulerSettings | None = None
    budget: BudgetSettings | None = None
    initialization: InitializationSettings | None = None
    analysis: AnalysisSettings | None = None

    @property
    def content_hash(self) -> str:
        return sha256_json(self.to_mapping())

    def to_mapping(self) -> JsonObject:
        raw: dict[str, object] = {
            "schema_version": self.schema_version,
            "run": {
                "root": self.run.root,
                "seed": self.run.seed,
                "max_steps": self.run.max_steps,
                "task_id": self.run.task_id,
                "split": self.run.split,
            },
            "proposer": {
                "id": self.proposer.proposer_id,
                "batch_size": self.proposer.batch_size,
            },
            "oracle": {"id": self.oracle.oracle_id},
            "logging": {"level": self.logging.level},
        }
        if self.schema_version in {2, 3}:
            if self.schema_version == 3:
                raw["run"] = {
                    "root": self.run.root,
                    "seed": self.run.seed,
                    "task_id": self.run.task_id,
                    "split": self.run.split,
                    "condition_id": self.run.condition_id,
                }
            if self.schema_version == 2 and (self.dsl is None or self.enumerator is None):
                raise AssertionError("Phase 2 configuration is missing DSL settings")
            raw["oracle"] = {
                "id": self.oracle.oracle_id,
                "response_mode": self.oracle.response_mode,
            }
            if self.dsl is None:
                raise AssertionError("typed configuration is missing DSL settings")
            raw["dsl"] = {
                "version": self.dsl.dsl_version,
                "max_depth": self.dsl.max_depth,
                "max_nodes": self.dsl.max_nodes,
                "max_cases": self.dsl.max_cases,
                "allowed_macros": self.dsl.allowed_macros,
            }
            if self.schema_version == 2:
                if self.enumerator is None:
                    raise AssertionError("Phase 2 configuration is missing enumerator settings")
                raw["enumerator"] = {
                    "version": self.enumerator.enumerator_version,
                    "max_bits": self.enumerator.max_bits,
                    "max_depth": self.enumerator.max_depth,
                    "max_nodes": self.enumerator.max_nodes,
                    "max_candidates": self.enumerator.max_candidates,
                }
            else:
                if any(
                    value is None
                    for value in (
                        self.operators,
                        self.archive,
                        self.scheduler,
                        self.budget,
                        self.initialization,
                        self.analysis,
                    )
                ):
                    raise AssertionError("Phase 3 configuration is incomplete")
                assert self.operators is not None
                assert self.archive is not None
                assert self.scheduler is not None
                assert self.budget is not None
                assert self.initialization is not None
                assert self.analysis is not None
                raw["operators"] = {
                    "version": self.operators.operator_version,
                    "weights": dict(self.operators.weights),
                    "retry_limit": self.operators.retry_limit,
                    "fallback_policy": self.operators.fallback_policy,
                }
                raw["archive"] = {
                    "version": self.archive.archive_version,
                    "descriptor_version": self.archive.descriptor_version,
                    "reserve_size": self.archive.reserve_size,
                }
                raw["scheduler"] = {"version": self.scheduler.scheduler_version}
                raw["budget"] = {
                    "version": self.budget.budget_version,
                    "proposal_attempt_cap": self.budget.proposal_attempt_cap,
                    "oracle_call_cap": self.budget.oracle_call_cap,
                }
                raw["initialization"] = {"version": self.initialization.initialization_version}
                raw["analysis"] = {"version": self.analysis.analysis_version}
        value = to_json_value(raw)
        if not isinstance(value, dict):  # pragma: no cover - mapping invariant
            raise AssertionError("configuration did not serialize as an object")
        return value


def config_from_mapping(raw: object) -> AppConfig:
    """Validate an in-memory configuration without creating any artifacts."""

    root = _expect_mapping(raw, "configuration")
    if "schema_version" not in root:
        raise ConfigurationError("configuration is missing keys: schema_version")
    schema_version = _expect_int(root["schema_version"], "schema_version", minimum=1)
    if schema_version not in {1, 2, 3}:
        raise ConfigurationError("schema_version must be 1, 2, or 3")
    expected_root = {"schema_version", "run", "proposer", "oracle", "logging"}
    if schema_version == 2:
        expected_root.update({"dsl", "enumerator"})
    if schema_version == 3:
        expected_root.update(
            {"dsl", "operators", "archive", "scheduler", "budget", "initialization", "analysis"}
        )
    _require_keys(root, expected_root, "configuration")

    run_raw = _expect_mapping(root["run"], "run")
    run_keys = (
        {"root", "seed", "max_steps", "task_id", "split"}
        if schema_version in {1, 2}
        else {"root", "seed", "task_id", "split", "condition_id"}
    )
    _require_keys(run_raw, run_keys, "run")
    root_text = _expect_str(run_raw["root"], "run.root")
    run_path = Path(root_text)
    if run_path.is_absolute() or run_path == Path(".") or ".." in run_path.parts:
        raise ConfigurationError(
            "run.root must be a specific repository-relative path without '..'"
        )
    seed = _expect_int(run_raw["seed"], "run.seed")
    max_steps = (
        _expect_int(run_raw["max_steps"], "run.max_steps", minimum=1)
        if schema_version in {1, 2}
        else 1
    )
    task_id = _expect_str(run_raw["task_id"], "run.task_id")
    try:
        split = SplitLabel(_expect_str(run_raw["split"], "run.split"))
    except ValueError as exc:
        labels = ", ".join(label.value for label in SplitLabel)
        raise ConfigurationError(f"run.split must be one of: {labels}") from exc
    condition_id: str | None = None
    if schema_version == 3:
        condition_id = _expect_str(run_raw["condition_id"], "run.condition_id")
        if condition_id not in {PHASE3_INCUMBENT_VERSION, "uniform-diverse-archive-v1"}:
            raise ConfigurationError(
                "run.condition_id must be single-incumbent-v1 or uniform-diverse-archive-v1"
            )

    proposer_raw = _expect_mapping(root["proposer"], "proposer")
    _require_keys(proposer_raw, {"id", "batch_size"}, "proposer")
    proposer_id = _expect_str(proposer_raw["id"], "proposer.id")
    expected_proposer = {1: "mock", 2: "enumerative", 3: "mutation"}[schema_version]
    if proposer_id != expected_proposer:
        raise ConfigurationError(
            f"schema {schema_version} requires proposer.id='{expected_proposer}'"
        )
    batch_size = _expect_int(proposer_raw["batch_size"], "proposer.batch_size", minimum=1)
    if batch_size != 1:
        raise ConfigurationError("the deterministic run lifecycle requires proposer.batch_size=1")

    oracle_raw = _expect_mapping(root["oracle"], "oracle")
    oracle_keys = {"id"} if schema_version == 1 else {"id", "response_mode"}
    _require_keys(oracle_raw, oracle_keys, "oracle")
    oracle_id = _expect_str(oracle_raw["id"], "oracle.id")
    expected_oracle = "mock-v1" if schema_version == 1 else "typed-elementary-exact-v1"
    if oracle_id != expected_oracle:
        raise ConfigurationError(f"schema {schema_version} requires oracle.id='{expected_oracle}'")
    response_mode = OracleResponseMode.SCORE_ONLY
    if schema_version in {2, 3}:
        try:
            response_mode = OracleResponseMode(
                _expect_str(oracle_raw["response_mode"], "oracle.response_mode")
            )
        except ValueError as exc:
            raise ConfigurationError("oracle.response_mode is invalid") from exc
        if response_mode is not OracleResponseMode.SCORE_ONLY:
            raise ConfigurationError("typed locked protocol requires score-only oracle feedback")

    logging_raw = _expect_mapping(root["logging"], "logging")
    _require_keys(logging_raw, {"level"}, "logging")
    level = _expect_str(logging_raw["level"], "logging.level").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigurationError("logging.level must be DEBUG, INFO, WARNING, or ERROR")

    dsl: DslSettings | None = None
    enumerator: EnumeratorSettings | None = None
    if schema_version in {2, 3}:
        dsl_raw = _expect_mapping(root["dsl"], "dsl")
        _require_keys(
            dsl_raw,
            {"version", "max_depth", "max_nodes", "max_cases", "allowed_macros"},
            "dsl",
        )
        dsl_version = _expect_str(dsl_raw["version"], "dsl.version")
        if dsl_version != DSL_VERSION:
            raise ConfigurationError(f"dsl.version must be '{DSL_VERSION}'")
        dsl_depth = _expect_int(dsl_raw["max_depth"], "dsl.max_depth", minimum=1)
        dsl_nodes = _expect_int(dsl_raw["max_nodes"], "dsl.max_nodes", minimum=1)
        dsl_cases = _expect_int(dsl_raw["max_cases"], "dsl.max_cases", minimum=1)
        if dsl_depth > 8 or dsl_nodes > 63 or dsl_cases != 8:
            raise ConfigurationError(
                "Phase 2 DSL bounds require depth <= 8, nodes <= 63, cases = 8"
            )
        macros_raw = dsl_raw["allowed_macros"]
        if (
            not isinstance(macros_raw, list)
            or not all(isinstance(item, str) for item in macros_raw)
            or len(macros_raw) != len(set(macros_raw))
            or set(macros_raw) != {"Parity", "Majority"}
        ):
            raise ConfigurationError("dsl.allowed_macros must contain Parity and Majority once")
        dsl = DslSettings(
            dsl_version=dsl_version,
            max_depth=dsl_depth,
            max_nodes=dsl_nodes,
            max_cases=dsl_cases,
            allowed_macros=tuple(macros_raw),
        )
    if schema_version == 2:
        enum_raw = _expect_mapping(root["enumerator"], "enumerator")
        _require_keys(
            enum_raw,
            {"version", "max_bits", "max_depth", "max_nodes", "max_candidates"},
            "enumerator",
        )
        enum_version = _expect_str(enum_raw["version"], "enumerator.version")
        if enum_version != ENUMERATOR_VERSION:
            raise ConfigurationError(f"enumerator.version must be '{ENUMERATOR_VERSION}'")
        enum_bits = _expect_int(enum_raw["max_bits"], "enumerator.max_bits", minimum=1)
        enum_depth = _expect_int(enum_raw["max_depth"], "enumerator.max_depth", minimum=1)
        enum_nodes = _expect_int(enum_raw["max_nodes"], "enumerator.max_nodes", minimum=1)
        enum_candidates = _expect_int(
            enum_raw["max_candidates"], "enumerator.max_candidates", minimum=1
        )
        if enum_bits > 64 or enum_depth > dsl_depth or enum_nodes > dsl_nodes:
            raise ConfigurationError("enumerator bounds exceed the frozen Phase 2 DSL limits")
        if enum_candidates > 100_000:
            raise ConfigurationError("enumerator.max_candidates must be <= 100000")
        enumerator = EnumeratorSettings(
            enumerator_version=enum_version,
            max_bits=enum_bits,
            max_depth=enum_depth,
            max_nodes=enum_nodes,
            max_candidates=enum_candidates,
        )

    operators: OperatorSettings | None = None
    archive: ArchiveSettings | None = None
    scheduler: SchedulerSettings | None = None
    budget: BudgetSettings | None = None
    initialization: InitializationSettings | None = None
    analysis: AnalysisSettings | None = None
    if schema_version == 3:
        operator_raw = _expect_mapping(root["operators"], "operators")
        _require_keys(
            operator_raw, {"version", "weights", "retry_limit", "fallback_policy"}, "operators"
        )
        if _expect_str(operator_raw["version"], "operators.version") != PHASE3_OPERATOR_VERSION:
            raise ConfigurationError(f"operators.version must be '{PHASE3_OPERATOR_VERSION}'")
        weights_raw = _expect_mapping(operator_raw["weights"], "operators.weights")
        expected_weights = {
            "local-mutation": 4,
            "subtree-replacement": 3,
            "simplification": 2,
            "typed-crossover": 3,
        }
        _require_keys(weights_raw, set(expected_weights), "operators.weights")
        weights = tuple(
            (name, _expect_int(weights_raw[name], f"operators.weights.{name}", minimum=1))
            for name in expected_weights
        )
        if dict(weights) != expected_weights:
            raise ConfigurationError("Phase 3 operator weights are frozen at 4/3/2/3")
        retry_limit = _expect_int(operator_raw["retry_limit"], "operators.retry_limit")
        fallback = _expect_str(operator_raw["fallback_policy"], "operators.fallback_policy")
        if retry_limit != 0 or fallback != "charged-no-op-v1":
            raise ConfigurationError("Phase 3 requires retry_limit=0 and charged-no-op-v1")
        operators = OperatorSettings(PHASE3_OPERATOR_VERSION, weights, retry_limit, fallback)

        archive_raw = _expect_mapping(root["archive"], "archive")
        _require_keys(archive_raw, {"version", "descriptor_version", "reserve_size"}, "archive")
        archive_version = _expect_str(archive_raw["version"], "archive.version")
        descriptor_version = _expect_str(
            archive_raw["descriptor_version"], "archive.descriptor_version"
        )
        reserve_size = _expect_int(archive_raw["reserve_size"], "archive.reserve_size")
        if archive_version != PHASE3_ARCHIVE_VERSION:
            raise ConfigurationError(f"archive.version must be '{PHASE3_ARCHIVE_VERSION}'")
        if descriptor_version != PHASE3_DESCRIPTOR_VERSION or reserve_size != 2:
            raise ConfigurationError("Phase 3 descriptor version/reserve size must be v1/2")
        archive = ArchiveSettings(archive_version, descriptor_version, reserve_size)

        scheduler_raw = _expect_mapping(root["scheduler"], "scheduler")
        _require_keys(scheduler_raw, {"version"}, "scheduler")
        scheduler_version = _expect_str(scheduler_raw["version"], "scheduler.version")
        if scheduler_version != PHASE3_SCHEDULER_VERSION:
            raise ConfigurationError(f"scheduler.version must be '{PHASE3_SCHEDULER_VERSION}'")
        scheduler = SchedulerSettings(scheduler_version)

        budget_raw = _expect_mapping(root["budget"], "budget")
        _require_keys(budget_raw, {"version", "proposal_attempt_cap", "oracle_call_cap"}, "budget")
        budget_version = _expect_str(budget_raw["version"], "budget.version")
        proposals = _expect_int(
            budget_raw["proposal_attempt_cap"], "budget.proposal_attempt_cap", minimum=1
        )
        calls = _expect_int(budget_raw["oracle_call_cap"], "budget.oracle_call_cap", minimum=1)
        if budget_version != PHASE3_BUDGET_VERSION:
            raise ConfigurationError(f"budget.version must be '{PHASE3_BUDGET_VERSION}'")
        if calls > proposals or calls > 2048 or proposals > 8192:
            raise ConfigurationError("Phase 3 budget caps are inconsistent or exceed safety limits")
        budget = BudgetSettings(budget_version, proposals, calls)
        max_steps = calls

        initialization_raw = _expect_mapping(root["initialization"], "initialization")
        _require_keys(initialization_raw, {"version"}, "initialization")
        initialization_version = _expect_str(
            initialization_raw["version"], "initialization.version"
        )
        if initialization_version != PHASE3_INITIALIZATION_VERSION:
            raise ConfigurationError(
                f"initialization.version must be '{PHASE3_INITIALIZATION_VERSION}'"
            )
        initialization = InitializationSettings(initialization_version)

        analysis_raw = _expect_mapping(root["analysis"], "analysis")
        _require_keys(analysis_raw, {"version"}, "analysis")
        analysis_version = _expect_str(analysis_raw["version"], "analysis.version")
        if analysis_version != PHASE3_ANALYSIS_VERSION:
            raise ConfigurationError(f"analysis.version must be '{PHASE3_ANALYSIS_VERSION}'")
        analysis = AnalysisSettings(analysis_version)

    return AppConfig(
        schema_version=schema_version,
        run=RunSettings(
            root=run_path,
            seed=seed,
            max_steps=max_steps,
            task_id=task_id,
            split=split,
            condition_id=condition_id,
        ),
        proposer=ProposerSettings(proposer_id=proposer_id, batch_size=batch_size),
        oracle=OracleSettings(oracle_id=oracle_id, response_mode=response_mode),
        logging=LoggingSettings(level=level),
        dsl=dsl,
        enumerator=enumerator,
        operators=operators,
        archive=archive,
        scheduler=scheduler,
        budget=budget,
        initialization=initialization,
        analysis=analysis,
    )


def load_config(path: Path) -> AppConfig:
    """Read and fully validate YAML. This function performs no writes."""

    if not path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {path}")
    try:
        loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
    return config_from_mapping(loaded)
