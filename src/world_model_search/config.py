"""Strict YAML configuration loading with persistence-free validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from world_model_search.domain.types import SplitLabel
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


@dataclass(frozen=True, slots=True)
class ProposerSettings:
    proposer_id: str
    batch_size: int


@dataclass(frozen=True, slots=True)
class OracleSettings:
    oracle_id: str


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    schema_version: int
    run: RunSettings
    proposer: ProposerSettings
    oracle: OracleSettings
    logging: LoggingSettings

    @property
    def content_hash(self) -> str:
        return sha256_json(self.to_mapping())

    def to_mapping(self) -> JsonObject:
        value = to_json_value(
            {
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
        )
        if not isinstance(value, dict):  # pragma: no cover - mapping invariant
            raise AssertionError("configuration did not serialize as an object")
        return value


def config_from_mapping(raw: object) -> AppConfig:
    """Validate an in-memory configuration without creating any artifacts."""

    root = _expect_mapping(raw, "configuration")
    _require_keys(root, {"schema_version", "run", "proposer", "oracle", "logging"}, "configuration")

    schema_version = _expect_int(root["schema_version"], "schema_version", minimum=1)
    if schema_version != 1:
        raise ConfigurationError("schema_version must be 1")

    run_raw = _expect_mapping(root["run"], "run")
    _require_keys(run_raw, {"root", "seed", "max_steps", "task_id", "split"}, "run")
    root_text = _expect_str(run_raw["root"], "run.root")
    run_path = Path(root_text)
    if run_path.is_absolute() or run_path == Path(".") or ".." in run_path.parts:
        raise ConfigurationError(
            "run.root must be a specific repository-relative path without '..'"
        )
    seed = _expect_int(run_raw["seed"], "run.seed")
    max_steps = _expect_int(run_raw["max_steps"], "run.max_steps", minimum=1)
    task_id = _expect_str(run_raw["task_id"], "run.task_id")
    try:
        split = SplitLabel(_expect_str(run_raw["split"], "run.split"))
    except ValueError as exc:
        labels = ", ".join(label.value for label in SplitLabel)
        raise ConfigurationError(f"run.split must be one of: {labels}") from exc

    proposer_raw = _expect_mapping(root["proposer"], "proposer")
    _require_keys(proposer_raw, {"id", "batch_size"}, "proposer")
    proposer_id = _expect_str(proposer_raw["id"], "proposer.id")
    if proposer_id != "mock":
        raise ConfigurationError("Phase 0 supports only proposer.id='mock'")
    batch_size = _expect_int(proposer_raw["batch_size"], "proposer.batch_size", minimum=1)
    if batch_size != 1:
        raise ConfigurationError("Phase 0 requires proposer.batch_size=1")

    oracle_raw = _expect_mapping(root["oracle"], "oracle")
    _require_keys(oracle_raw, {"id"}, "oracle")
    oracle_id = _expect_str(oracle_raw["id"], "oracle.id")
    if oracle_id != "mock-v1":
        raise ConfigurationError("Phase 0 supports only oracle.id='mock-v1'")

    logging_raw = _expect_mapping(root["logging"], "logging")
    _require_keys(logging_raw, {"level"}, "logging")
    level = _expect_str(logging_raw["level"], "logging.level").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigurationError("logging.level must be DEBUG, INFO, WARNING, or ERROR")

    return AppConfig(
        schema_version=schema_version,
        run=RunSettings(
            root=run_path,
            seed=seed,
            max_steps=max_steps,
            task_id=task_id,
            split=split,
        ),
        proposer=ProposerSettings(proposer_id=proposer_id, batch_size=batch_size),
        oracle=OracleSettings(oracle_id=oracle_id),
        logging=LoggingSettings(level=level),
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
