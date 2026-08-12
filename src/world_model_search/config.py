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
from world_model_search.model.cache import CACHE_VERSION
from world_model_search.model.policy import SUPPORTED_PRICE_POLICY_VERSIONS
from world_model_search.model.prompts import (
    DIRECT_PROMPT_VERSION,
    FEEDBACK_SCHEMA_VERSION,
    ITERATIVE_PROMPT_VERSION,
)
from world_model_search.model.schema import BATCH_SCHEMA_VERSION
from world_model_search.phase4_versions import (
    PHASE4_ANALYSIS_VERSION,
    PHASE4_BUDGET_VERSION,
    PHASE4_CONFIG_SCHEMA_VERSION,
    PHASE4_RETRY_VERSION,
    PHASE4_ROLE_SCHEDULE_VERSION,
)
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
class ModelSettings:
    backend_id: str
    provider_id: str
    resolved_model: str
    endpoint: str
    service_tier: str
    reasoning_effort: str
    max_output_tokens: int
    store: bool
    truncation: str

    def request_settings(self) -> JsonObject:
        return {
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "store": self.store,
            "truncation": self.truncation,
        }


@dataclass(frozen=True, slots=True)
class PromptSettings:
    direct_version: str
    iterative_version: str
    feedback_schema: str
    batch_schema_version: int
    role_schedule_version: str
    role: str


@dataclass(frozen=True, slots=True)
class CacheSettings:
    cache_version: str
    root: Path
    namespace: str


@dataclass(frozen=True, slots=True)
class RetrySettings:
    retry_version: str
    max_retries: int
    backoff_milliseconds: tuple[int, ...]
    retryable_categories: tuple[str, ...]
    repair_policy: str


@dataclass(frozen=True, slots=True)
class Phase4BudgetSettings:
    budget_version: str
    model_request_cap: int
    input_token_cap: int
    output_token_cap: int
    total_token_cap: int
    proposal_item_cap: int
    oracle_call_cap: int
    child_nano_usd_cap: int


@dataclass(frozen=True, slots=True)
class Phase4PolicySettings:
    stage: str
    price_policy: Path
    ledger: Path
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
    model: ModelSettings | None = None
    prompt: PromptSettings | None = None
    cache: CacheSettings | None = None
    retry: RetrySettings | None = None
    phase4_budget: Phase4BudgetSettings | None = None
    phase4_policy: Phase4PolicySettings | None = None

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
        if self.schema_version in {2, 3, 4}:
            if self.schema_version in {3, 4}:
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
            elif self.schema_version == 3:
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
            else:
                if any(
                    value is None
                    for value in (
                        self.archive,
                        self.scheduler,
                        self.initialization,
                        self.model,
                        self.prompt,
                        self.cache,
                        self.retry,
                        self.phase4_budget,
                        self.phase4_policy,
                    )
                ):
                    raise AssertionError("Phase 4 configuration is incomplete")
                assert self.archive is not None
                assert self.scheduler is not None
                assert self.initialization is not None
                assert self.model is not None
                assert self.prompt is not None
                assert self.cache is not None
                assert self.retry is not None
                assert self.phase4_budget is not None
                assert self.phase4_policy is not None
                raw["archive"] = {
                    "version": self.archive.archive_version,
                    "descriptor_version": self.archive.descriptor_version,
                    "reserve_size": self.archive.reserve_size,
                }
                raw["scheduler"] = {"version": self.scheduler.scheduler_version}
                raw["initialization"] = {"version": self.initialization.initialization_version}
                raw["model"] = {
                    "backend": self.model.backend_id,
                    "provider": self.model.provider_id,
                    "resolved_model": self.model.resolved_model,
                    "endpoint": self.model.endpoint,
                    "service_tier": self.model.service_tier,
                    "reasoning_effort": self.model.reasoning_effort,
                    "max_output_tokens": self.model.max_output_tokens,
                    "store": self.model.store,
                    "truncation": self.model.truncation,
                }
                raw["prompt"] = {
                    "direct_version": self.prompt.direct_version,
                    "iterative_version": self.prompt.iterative_version,
                    "feedback_schema": self.prompt.feedback_schema,
                    "batch_schema_version": self.prompt.batch_schema_version,
                    "role_schedule_version": self.prompt.role_schedule_version,
                    "role": self.prompt.role,
                }
                raw["cache"] = {
                    "version": self.cache.cache_version,
                    "root": self.cache.root,
                    "namespace": self.cache.namespace,
                }
                raw["retry"] = {
                    "version": self.retry.retry_version,
                    "max_retries": self.retry.max_retries,
                    "backoff_milliseconds": self.retry.backoff_milliseconds,
                    "retryable_categories": self.retry.retryable_categories,
                    "repair_policy": self.retry.repair_policy,
                }
                raw["budget"] = {
                    "version": self.phase4_budget.budget_version,
                    "model_request_cap": self.phase4_budget.model_request_cap,
                    "input_token_cap": self.phase4_budget.input_token_cap,
                    "output_token_cap": self.phase4_budget.output_token_cap,
                    "total_token_cap": self.phase4_budget.total_token_cap,
                    "proposal_item_cap": self.phase4_budget.proposal_item_cap,
                    "oracle_call_cap": self.phase4_budget.oracle_call_cap,
                    "child_nano_usd_cap": self.phase4_budget.child_nano_usd_cap,
                }
                raw["phase4"] = {
                    "stage": self.phase4_policy.stage,
                    "price_policy": self.phase4_policy.price_policy,
                    "ledger": self.phase4_policy.ledger,
                    "analysis_version": self.phase4_policy.analysis_version,
                }
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
    if schema_version not in {1, 2, 3, PHASE4_CONFIG_SCHEMA_VERSION}:
        raise ConfigurationError("schema_version must be 1, 2, 3, or 4")
    expected_root = {"schema_version", "run", "proposer", "oracle", "logging"}
    if schema_version == 2:
        expected_root.update({"dsl", "enumerator"})
    if schema_version == 3:
        expected_root.update(
            {"dsl", "operators", "archive", "scheduler", "budget", "initialization", "analysis"}
        )
    if schema_version == 4:
        expected_root.update(
            {
                "dsl",
                "archive",
                "scheduler",
                "initialization",
                "model",
                "prompt",
                "cache",
                "retry",
                "budget",
                "phase4",
            }
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
    if schema_version in {3, 4}:
        condition_id = _expect_str(run_raw["condition_id"], "run.condition_id")
        allowed_conditions = (
            {PHASE3_INCUMBENT_VERSION, "uniform-diverse-archive-v1"}
            if schema_version == 3
            else {"direct-llm-v1", PHASE3_INCUMBENT_VERSION, "uniform-diverse-archive-v1"}
        )
        if condition_id not in allowed_conditions:
            raise ConfigurationError(
                "run.condition_id is not a supported condition for this schema"
            )

    proposer_raw = _expect_mapping(root["proposer"], "proposer")
    _require_keys(proposer_raw, {"id", "batch_size"}, "proposer")
    proposer_id = _expect_str(proposer_raw["id"], "proposer.id")
    expected_proposer = {1: "mock", 2: "enumerative", 3: "mutation", 4: "llm"}[schema_version]
    if proposer_id != expected_proposer:
        raise ConfigurationError(
            f"schema {schema_version} requires proposer.id='{expected_proposer}'"
        )
    batch_size = _expect_int(proposer_raw["batch_size"], "proposer.batch_size", minimum=1)
    if schema_version != 4 and batch_size != 1:
        raise ConfigurationError("the deterministic run lifecycle requires proposer.batch_size=1")
    if schema_version == 4 and batch_size > 16:
        raise ConfigurationError("Phase 4 proposer.batch_size must be <= 16")

    oracle_raw = _expect_mapping(root["oracle"], "oracle")
    oracle_keys = {"id"} if schema_version == 1 else {"id", "response_mode"}
    _require_keys(oracle_raw, oracle_keys, "oracle")
    oracle_id = _expect_str(oracle_raw["id"], "oracle.id")
    expected_oracle = "mock-v1" if schema_version == 1 else "typed-elementary-exact-v1"
    if oracle_id != expected_oracle:
        raise ConfigurationError(f"schema {schema_version} requires oracle.id='{expected_oracle}'")
    response_mode = OracleResponseMode.SCORE_ONLY
    if schema_version in {2, 3, 4}:
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
    if schema_version in {2, 3, 4}:
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
    model: ModelSettings | None = None
    prompt: PromptSettings | None = None
    cache: CacheSettings | None = None
    retry: RetrySettings | None = None
    phase4_budget: Phase4BudgetSettings | None = None
    phase4_policy: Phase4PolicySettings | None = None
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

    if schema_version == 4:
        archive_raw = _expect_mapping(root["archive"], "archive")
        _require_keys(archive_raw, {"version", "descriptor_version", "reserve_size"}, "archive")
        archive_version = _expect_str(archive_raw["version"], "archive.version")
        descriptor_version = _expect_str(
            archive_raw["descriptor_version"], "archive.descriptor_version"
        )
        reserve_size = _expect_int(archive_raw["reserve_size"], "archive.reserve_size")
        if (
            archive_version != PHASE3_ARCHIVE_VERSION
            or descriptor_version != PHASE3_DESCRIPTOR_VERSION
            or reserve_size != 2
        ):
            raise ConfigurationError("Phase 4 preserves the frozen Phase 3 archive contract")
        archive = ArchiveSettings(archive_version, descriptor_version, reserve_size)

        scheduler_raw = _expect_mapping(root["scheduler"], "scheduler")
        _require_keys(scheduler_raw, {"version"}, "scheduler")
        scheduler_version = _expect_str(scheduler_raw["version"], "scheduler.version")
        if scheduler_version != PHASE3_SCHEDULER_VERSION:
            raise ConfigurationError("Phase 4 requires the uniform Phase 3 scheduler")
        scheduler = SchedulerSettings(scheduler_version)

        initialization_raw = _expect_mapping(root["initialization"], "initialization")
        _require_keys(initialization_raw, {"version"}, "initialization")
        initialization_version = _expect_str(
            initialization_raw["version"], "initialization.version"
        )
        if initialization_version != PHASE3_INITIALIZATION_VERSION:
            raise ConfigurationError("Phase 4 requires the seven shared Phase 3 initial candidates")
        initialization = InitializationSettings(initialization_version)

        model_raw = _expect_mapping(root["model"], "model")
        _require_keys(
            model_raw,
            {
                "backend",
                "provider",
                "resolved_model",
                "endpoint",
                "service_tier",
                "reasoning_effort",
                "max_output_tokens",
                "store",
                "truncation",
            },
            "model",
        )
        backend_id = _expect_str(model_raw["backend"], "model.backend")
        provider_id = _expect_str(model_raw["provider"], "model.provider")
        allowed_backends = {
            "scripted-deterministic-v1": "scripted",
            "openai-responses-sdk-v1": "openai",
        }
        if allowed_backends.get(backend_id) != provider_id:
            raise ConfigurationError("model backend/provider pair is not supported")
        resolved_model = _expect_str(model_raw["resolved_model"], "model.resolved_model")
        endpoint = _expect_str(model_raw["endpoint"], "model.endpoint")
        service_tier = _expect_str(model_raw["service_tier"], "model.service_tier")
        reasoning_effort = _expect_str(model_raw["reasoning_effort"], "model.reasoning_effort")
        max_output_tokens = _expect_int(
            model_raw["max_output_tokens"], "model.max_output_tokens", minimum=1
        )
        store = model_raw["store"]
        truncation = _expect_str(model_raw["truncation"], "model.truncation")
        if (
            resolved_model != "gpt-5-mini-2025-08-07"
            or endpoint != "v1/responses"
            or service_tier != "default"
            or reasoning_effort != "low"
            or max_output_tokens > 4096
            or store is not False
            or truncation != "disabled"
        ):
            raise ConfigurationError("Phase 4 model snapshot and supported settings are frozen")
        model = ModelSettings(
            backend_id,
            provider_id,
            resolved_model,
            endpoint,
            service_tier,
            reasoning_effort,
            max_output_tokens,
            False,
            truncation,
        )

        prompt_raw = _expect_mapping(root["prompt"], "prompt")
        _require_keys(
            prompt_raw,
            {
                "direct_version",
                "iterative_version",
                "feedback_schema",
                "batch_schema_version",
                "role_schedule_version",
                "role",
            },
            "prompt",
        )
        batch_schema_version = _expect_int(
            prompt_raw["batch_schema_version"], "prompt.batch_schema_version", minimum=1
        )
        prompt_values = (
            _expect_str(prompt_raw["direct_version"], "prompt.direct_version"),
            _expect_str(prompt_raw["iterative_version"], "prompt.iterative_version"),
            _expect_str(prompt_raw["feedback_schema"], "prompt.feedback_schema"),
            batch_schema_version,
            _expect_str(prompt_raw["role_schedule_version"], "prompt.role_schedule_version"),
            _expect_str(prompt_raw["role"], "prompt.role"),
        )
        if prompt_values != (
            DIRECT_PROMPT_VERSION,
            ITERATIVE_PROMPT_VERSION,
            FEEDBACK_SCHEMA_VERSION,
            BATCH_SCHEMA_VERSION,
            PHASE4_ROLE_SCHEDULE_VERSION,
            "exploit",
        ):
            raise ConfigurationError("Phase 4 prompt/schema/role schedule is frozen")
        prompt = PromptSettings(*prompt_values)

        cache_raw = _expect_mapping(root["cache"], "cache")
        _require_keys(cache_raw, {"version", "root", "namespace"}, "cache")
        cache_version = _expect_str(cache_raw["version"], "cache.version")
        cache_root = Path(_expect_str(cache_raw["root"], "cache.root"))
        cache_namespace = _expect_str(cache_raw["namespace"], "cache.namespace")
        if (
            cache_version != CACHE_VERSION
            or cache_root.is_absolute()
            or ".." in cache_root.parts
            or not cache_root.parts
            or "/" in cache_namespace
            or "\\" in cache_namespace
            or ".." in cache_namespace
        ):
            raise ConfigurationError("Phase 4 cache path/version/namespace is invalid")
        cache = CacheSettings(cache_version, cache_root, cache_namespace)

        retry_raw = _expect_mapping(root["retry"], "retry")
        _require_keys(
            retry_raw,
            {
                "version",
                "max_retries",
                "backoff_milliseconds",
                "retryable_categories",
                "repair_policy",
            },
            "retry",
        )
        retry_version = _expect_str(retry_raw["version"], "retry.version")
        max_retries = _expect_int(retry_raw["max_retries"], "retry.max_retries")
        backoff_raw = retry_raw["backoff_milliseconds"]
        categories_raw = retry_raw["retryable_categories"]
        if not isinstance(backoff_raw, list) or not isinstance(categories_raw, list):
            raise ConfigurationError("retry backoff/categories must be arrays")
        backoffs = tuple(
            _expect_int(value, "retry.backoff_milliseconds item") for value in backoff_raw
        )
        categories = tuple(
            _expect_str(value, "retry.retryable_categories item") for value in categories_raw
        )
        repair_policy = _expect_str(retry_raw["repair_policy"], "retry.repair_policy")
        expected_categories = ("rate-limit", "malformed-response")
        if (
            retry_version != PHASE4_RETRY_VERSION
            or max_retries != 1
            or backoffs != (0,)
            or categories != expected_categories
            or repair_policy != "identical-request-no-conversation-v1"
        ):
            raise ConfigurationError("Phase 4 retry policy is frozen at one identical retry")
        retry = RetrySettings(retry_version, max_retries, backoffs, categories, repair_policy)

        phase4_budget_raw = _expect_mapping(root["budget"], "budget")
        budget_fields = {
            "version",
            "model_request_cap",
            "input_token_cap",
            "output_token_cap",
            "total_token_cap",
            "proposal_item_cap",
            "oracle_call_cap",
            "child_nano_usd_cap",
        }
        _require_keys(phase4_budget_raw, budget_fields, "budget")
        phase4_budget_version = _expect_str(phase4_budget_raw["version"], "budget.version")
        request_cap = _expect_int(
            phase4_budget_raw["model_request_cap"], "budget.model_request_cap", minimum=1
        )
        input_cap = _expect_int(
            phase4_budget_raw["input_token_cap"], "budget.input_token_cap", minimum=1
        )
        output_cap = _expect_int(
            phase4_budget_raw["output_token_cap"], "budget.output_token_cap", minimum=1
        )
        total_cap = _expect_int(
            phase4_budget_raw["total_token_cap"], "budget.total_token_cap", minimum=1
        )
        item_cap = _expect_int(
            phase4_budget_raw["proposal_item_cap"], "budget.proposal_item_cap", minimum=1
        )
        oracle_cap = _expect_int(
            phase4_budget_raw["oracle_call_cap"], "budget.oracle_call_cap", minimum=7
        )
        child_cap = _expect_int(
            phase4_budget_raw["child_nano_usd_cap"], "budget.child_nano_usd_cap"
        )
        if (
            phase4_budget_version != PHASE4_BUDGET_VERSION
            or request_cap > 4096
            or input_cap > 10_000_000
            or output_cap > 1_000_000
            or total_cap != input_cap + output_cap
            or item_cap > request_cap * batch_size
            or oracle_cap > 7 + item_cap
            or oracle_cap > 4096
            or child_cap > 500_000_000
        ):
            raise ConfigurationError("Phase 4 joint budget vector is inconsistent or unsafe")
        phase4_budget = Phase4BudgetSettings(
            phase4_budget_version,
            request_cap,
            input_cap,
            output_cap,
            total_cap,
            item_cap,
            oracle_cap,
            child_cap,
        )
        max_steps = oracle_cap

        phase4_raw = _expect_mapping(root["phase4"], "phase4")
        _require_keys(
            phase4_raw,
            {"stage", "price_policy", "ledger", "analysis_version"},
            "phase4",
        )
        stage = _expect_str(phase4_raw["stage"], "phase4.stage")
        price_policy = Path(_expect_str(phase4_raw["price_policy"], "phase4.price_policy"))
        ledger = Path(_expect_str(phase4_raw["ledger"], "phase4.ledger"))
        phase4_analysis = _expect_str(phase4_raw["analysis_version"], "phase4.analysis_version")
        if (
            stage not in {"fake", "canary", "development", "pilot", "locked-test"}
            or price_policy.is_absolute()
            or ".." in price_policy.parts
            or ledger.is_absolute()
            or ".." in ledger.parts
            or not ledger.parts
            or ledger.parts[0] != "local_state"
            or phase4_analysis != PHASE4_ANALYSIS_VERSION
        ):
            raise ConfigurationError("Phase 4 stage/policy/ledger/analysis contract is invalid")
        if stage == "fake" and child_cap != 0:
            raise ConfigurationError("fake Phase 4 runs must have a zero dollar child cap")
        if stage in {"canary", "development", "pilot"} and child_cap > 150_000_000:
            raise ConfigurationError("non-locked Phase 4 child cap exceeds $0.15")
        phase4_policy = Phase4PolicySettings(stage, price_policy, ledger, phase4_analysis)

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
        model=model,
        prompt=prompt,
        cache=cache,
        retry=retry,
        phase4_budget=phase4_budget,
        phase4_policy=phase4_policy,
    )


def load_config(path: Path) -> AppConfig:
    """Read and fully validate YAML. This function performs no writes."""

    if not path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {path}")
    try:
        loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
    config = config_from_mapping(loaded)
    if config.schema_version == 4:
        if config.phase4_policy is None:
            raise AssertionError("validated Phase 4 config has no policy settings")
        from world_model_search.model.policy import load_price_policy

        policy = load_price_policy(Path.cwd() / config.phase4_policy.price_policy)
        if policy.policy_version not in SUPPORTED_PRICE_POLICY_VERSIONS:
            raise ConfigurationError("Phase 4 price policy version is invalid")
        if config.phase4_budget is None or config.model is None:
            raise AssertionError("validated Phase 4 config is incomplete")
        if config.phase4_budget.child_nano_usd_cap > policy.child_cap(config.phase4_policy.stage):
            raise ConfigurationError("configuration child dollar cap exceeds price policy")
        if config.model.provider_id == "openai" and (
            config.model.resolved_model != policy.price.model
            or config.model.endpoint != policy.price.endpoint
            or config.model.service_tier != policy.price.service_tier
        ):
            raise ConfigurationError("model configuration and price entry do not match")
    return config
