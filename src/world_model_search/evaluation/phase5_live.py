"""Frozen Phase 5 live canary/development preparation and fail-closed runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from world_model_search.domain.types import OracleResponseMode, ProposalRole, SplitLabel
from world_model_search.dsl.ast import AstLimits
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.primitive_schema import (
    PrimitiveCandidateJsonError,
    parse_primitive_candidate_batch,
    primitive_candidate_batch_json_schema,
)
from world_model_search.dsl.primitives import (
    PrimitiveRegistry,
    empty_primitive_registry,
    encode_program,
    library_definition_cost,
    load_primitive_registry,
)
from world_model_search.errors import (
    BudgetExhaustedError,
    ConfigurationError,
    PersistenceError,
    ReplayError,
)
from world_model_search.evaluation.phase5_transfer import (
    Phase5TaskStore,
    generate_transfer_benchmark,
    load_transfer_public_task,
    load_transfer_registry,
)
from world_model_search.memory.retrieval import retrieve_memory
from world_model_search.memory.types import MemorySnapshot, load_memory_snapshot
from world_model_search.model.backends import LiveOptIn, OpenAIResponsesBackend
from world_model_search.model.cache import ExactResponseCache
from world_model_search.model.ledger import ProjectLedger
from world_model_search.model.phase5_prompts import (
    Phase5ModelRequest,
    Phase5RequestBindings,
    assert_matched_prompt_isolation,
    render_phase5_prompt,
)
from world_model_search.model.policy import PricePolicy, load_price_policy
from world_model_search.model.types import (
    ModelBackend,
    ModelDispatchError,
    ModelResponse,
)
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.persistence.artifacts import (
    read_text_artifact,
    write_content_artifact,
)
from world_model_search.serialization import JsonObject, canonical_json, sha256_json, sha256_text

LIVE_EXPERIMENT_VERSION = "phase5-live-cd-experiment-v1"
LIVE_AUTHORITY_VERSION = "phase5-live-authority-v1"
EXPOSURE_POLICY_VERSION = "phase5-published-exposure-partition-v2"
CONDITION_C = "condition-c-empty-memory-v1"
CONDITION_D = "condition-d-typed-memory-v1"


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


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{location} must be boolean")
    return value


def _path(value: object, location: str) -> Path:
    path = Path(_string(value, location))
    if path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(f"{location} must be repository-relative")
    return path


def _hash(value: object, location: str) -> str:
    digest = _string(value, location)
    if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise ConfigurationError(f"{location} must be a lowercase SHA-256")
    return digest


def _yaml(path: Path, location: str) -> dict[str, object]:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{location} is unavailable or invalid") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class Phase5ExposurePolicy:
    parent_cash_policy: Path
    parent_cash_policy_hash: str
    canary_stage_nano_usd: int
    development_stage_nano_usd: int
    sealed_test_stage_nano_usd: int
    phase5_total_nano_usd: int
    request_nano_usd: int
    child_nano_usd: int
    personal_cash_ceiling_nano_usd: int
    source_hash: str


def load_phase5_exposure_policy(path: Path) -> Phase5ExposurePolicy:
    root = _mapping(
        _yaml(path, "Phase 5 exposure policy"),
        {
            "policy_version",
            "status",
            "currency_unit",
            "parent_cash_policy",
            "parent_cash_policy_hash",
            "published_exposure_ceilings_nano_usd",
            "boundary",
        },
        "Phase 5 exposure policy",
    )
    if (
        root["policy_version"] != EXPOSURE_POLICY_VERSION
        or root["status"] != "frozen-pending-user-review"
        or root["currency_unit"] != "nano-USD"
    ):
        raise ConfigurationError("Phase 5 exposure policy version/status/unit is invalid")
    caps = _mapping(
        root["published_exposure_ceilings_nano_usd"],
        {
            "one_request",
            "child",
            "training_canary_stage",
            "development_stage",
            "sealed_test_stage",
            "phase5_total",
        },
        "Phase 5 exposure ceilings",
    )
    boundary = _mapping(
        root["boundary"],
        {
            "personal_actual_cash_ceiling_nano_usd",
            "enforcement",
            "reconciliation_changes_design_inside_frozen_experiment",
        },
        "Phase 5 exposure boundary",
    )
    values = {name: _integer(value, f"Phase 5 exposure {name}") for name, value in caps.items()}
    if (
        values["training_canary_stage"] + values["development_stage"] + values["sealed_test_stage"]
        != values["phase5_total"]
    ):
        raise ConfigurationError("Phase 5 stage partitions must sum exactly to the total")
    if values["one_request"] > values["child"]:
        raise ConfigurationError("Phase 5 one-request exposure exceeds the child ceiling")
    if (
        boundary["enforcement"]
        != "reconciled-cash-plus-worst-case-unreconciled-published-and-reservations-v1"
        or boundary["reconciliation_changes_design_inside_frozen_experiment"] is not False
    ):
        raise ConfigurationError("Phase 5 cash boundary is not fail closed")
    return Phase5ExposurePolicy(
        parent_cash_policy=_path(root["parent_cash_policy"], "parent cash policy"),
        parent_cash_policy_hash=_hash(root["parent_cash_policy_hash"], "parent policy hash"),
        canary_stage_nano_usd=values["training_canary_stage"],
        development_stage_nano_usd=values["development_stage"],
        sealed_test_stage_nano_usd=values["sealed_test_stage"],
        phase5_total_nano_usd=values["phase5_total"],
        request_nano_usd=values["one_request"],
        child_nano_usd=values["child"],
        personal_cash_ceiling_nano_usd=_integer(
            boundary["personal_actual_cash_ceiling_nano_usd"], "personal cash ceiling"
        ),
        source_hash=sha256_json(root),
    )


@dataclass(frozen=True, slots=True)
class Phase5LiveExperiment:
    source_hash: str
    experiment_id: str
    stage: str
    transfer_registry: Path
    transfer_registry_hash: str
    exposure_policy: Path
    exposure_policy_hash: str
    memory_snapshot: Path
    memory_snapshot_hash: str
    memory_snapshot_artifact_hash: str
    primitive_registry: Path
    primitive_registry_hash: str
    primitive_registry_artifact_hash: str
    conditions: tuple[str, ...]
    split: SplitLabel
    task_ids: tuple[str, ...]
    search_seeds: tuple[int, ...]
    requests_per_child: int
    batch_size: int
    input_token_cap: int
    output_token_cap: int
    total_token_cap: int
    proposal_item_cap: int
    oracle_call_cap: int
    model: str
    endpoint: str
    service_tier: str
    reasoning_effort: str
    max_output_tokens: int
    backend_id: str
    provider_id: str
    retrieval_max_items: int
    retrieval_max_bytes: int
    retrieval_max_tokens: int
    output_root: Path
    cache_root: Path
    cache_namespace: str
    cash_ledger: Path
    prerequisite_canary_registry: Path | None
    prerequisite_canary_registry_hash: str | None
    prerequisite_canary_summary: Path | None

    @property
    def child_count(self) -> int:
        return len(self.task_ids) * len(self.search_seeds) * len(self.conditions)

    @property
    def total_request_cap(self) -> int:
        return self.child_count * self.requests_per_child


def load_phase5_live_experiment(path: Path) -> Phase5LiveExperiment:
    root = _mapping(
        _yaml(path, "Phase 5 live experiment"),
        {
            "experiment_schema_version",
            "experiment_version",
            "experiment_id",
            "status",
            "stage",
            "transfer_registry",
            "transfer_registry_hash",
            "exposure_policy",
            "exposure_policy_hash",
            "frozen_memory",
            "conditions",
            "task_selection",
            "search_seeds",
            "matched_contract",
            "retrieval",
            "storage",
            "prerequisite_canary",
            "analysis",
            "scientific_role",
        },
        "Phase 5 live experiment",
    )
    if (
        root["experiment_schema_version"] != 1
        or root["experiment_version"] != LIVE_EXPERIMENT_VERSION
        or root["status"] != "frozen-pending-user-authorization"
    ):
        raise ConfigurationError("Phase 5 live experiment version/status is invalid")
    stage = _string(root["stage"], "stage")
    if stage not in {"training-canary", "development-pilot"}:
        raise ConfigurationError("Phase 5 live stage is invalid")
    frozen = _mapping(
        root["frozen_memory"],
        {
            "snapshot",
            "snapshot_hash",
            "snapshot_artifact_hash",
            "primitive_registry",
            "primitive_registry_hash",
            "primitive_registry_artifact_hash",
        },
        "frozen memory",
    )
    task = _mapping(root["task_selection"], {"split", "task_ids"}, "task selection")
    try:
        split = SplitLabel(_string(task["split"], "task split"))
    except ValueError as exc:
        raise ConfigurationError("Phase 5 live task split is invalid") from exc
    ids_raw = task["task_ids"]
    seeds_raw = root["search_seeds"]
    conditions_raw = root["conditions"]
    if (
        not isinstance(ids_raw, list)
        or not ids_raw
        or any(not isinstance(item, str) or not item for item in ids_raw)
        or len(ids_raw) != len(set(ids_raw))
        or not isinstance(seeds_raw, list)
        or not seeds_raw
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in seeds_raw
        )
        or len(seeds_raw) != len(set(seeds_raw))
        or not isinstance(conditions_raw, list)
    ):
        raise ConfigurationError("Phase 5 tasks, seeds, or conditions are invalid")
    conditions = tuple(str(item) for item in conditions_raw)
    expected_conditions = (
        (CONDITION_D,) if stage == "training-canary" else (CONDITION_C, CONDITION_D)
    )
    expected_split = SplitLabel.TRAINING if stage == "training-canary" else SplitLabel.DEVELOPMENT
    if conditions != expected_conditions or split is not expected_split:
        raise ConfigurationError("Phase 5 stage conditions or data role are invalid")
    matched = _mapping(
        root["matched_contract"],
        {
            "algorithm",
            "scheduler",
            "backend",
            "provider",
            "model",
            "endpoint",
            "service_tier",
            "reasoning_effort",
            "max_output_tokens",
            "batch_size",
            "requests_per_child",
            "input_token_cap",
            "output_token_cap",
            "total_token_cap",
            "proposal_item_cap",
            "oracle_call_cap",
            "continue_after_first_exact",
            "prompt_template",
            "response_schema",
        },
        "matched contract",
    )
    required_model = {
        "backend": "openai-responses-sdk-v1",
        "provider": "openai",
        "model": "gpt-5-mini-2025-08-07",
        "endpoint": "v1/responses",
        "service_tier": "default",
        "reasoning_effort": "low",
        "max_output_tokens": 2048,
        "batch_size": 1,
        "continue_after_first_exact": True,
        "prompt_template": "phase5-public-task-with-explicit-memory-v1",
        "response_schema": "world_model_candidate_batch_v1-with-frozen-primitives",
        "algorithm": "independent-bounded-proposal-search-v1",
        "scheduler": "uniform-request-index-v1",
    }
    if any(matched.get(key) != value for key, value in required_model.items()):
        raise ConfigurationError("Phase 5 live model/search contract is not frozen exactly")
    retrieval = _mapping(root["retrieval"], {"max_items", "max_bytes", "max_tokens"}, "retrieval")
    storage = _mapping(
        root["storage"],
        {"output_root", "cache_root", "cache_namespace", "cash_ledger"},
        "storage",
    )
    prerequisite = root["prerequisite_canary"]
    analysis = _mapping(
        root["analysis"],
        {
            "primary_endpoint",
            "secondary_endpoints",
            "pairing",
            "uncertainty",
            "bootstrap_seed",
            "bootstrap_replicates",
            "confidence_level",
            "multiplicity",
        },
        "analysis",
    )
    expected_analysis = {
        "primary_endpoint": "net-held-out-two-part-code-length-gain",
        "secondary_endpoints": [
            "matched-search-quality",
            "retrieval-precision",
            "primitive-transfer-gain-by-target-family",
            "runtime",
            "model-token-usage",
            "published-rate-equivalent-cost",
            "reconciled-actual-cash",
        ],
        "pairing": "exact-task-id-and-search-seed-v1",
        "uncertainty": "family-stratified-task-cluster-bootstrap-v1",
        "bootstrap_seed": 56001,
        "bootstrap_replicates": 10000,
        "confidence_level": 95,
        "multiplicity": "holm-two-sided-across-predeclared-secondary-comparisons-v1",
    }
    if analysis != expected_analysis:
        raise ConfigurationError("Phase 5 live analysis plan differs from the frozen contract")
    prerequisite_registry: Path | None = None
    prerequisite_hash: str | None = None
    prerequisite_summary: Path | None = None
    if stage == "training-canary":
        if prerequisite is not None:
            raise ConfigurationError("training canary cannot have a canary prerequisite")
    else:
        raw_prerequisite = _mapping(
            prerequisite,
            {"experiment", "experiment_hash", "required_summary", "required_status"},
            "canary prerequisite",
        )
        if raw_prerequisite["required_status"] != "passed-live-training-canary":
            raise ConfigurationError("development prerequisite status is invalid")
        prerequisite_registry = _path(raw_prerequisite["experiment"], "canary experiment")
        prerequisite_hash = _hash(raw_prerequisite["experiment_hash"], "canary experiment hash")
        prerequisite_summary = _path(raw_prerequisite["required_summary"], "canary summary")
    if root["scientific_role"] != (
        "compatibility-only-not-scientific-evidence"
        if stage == "training-canary"
        else "development-pilot-not-confirmatory-h3"
    ):
        raise ConfigurationError("Phase 5 scientific role is mislabeled")
    experiment = Phase5LiveExperiment(
        source_hash=sha256_json(root),
        experiment_id=_string(root["experiment_id"], "experiment ID"),
        stage=stage,
        transfer_registry=_path(root["transfer_registry"], "transfer registry"),
        transfer_registry_hash=_hash(root["transfer_registry_hash"], "transfer registry hash"),
        exposure_policy=_path(root["exposure_policy"], "exposure policy"),
        exposure_policy_hash=_hash(root["exposure_policy_hash"], "exposure policy hash"),
        memory_snapshot=_path(frozen["snapshot"], "memory snapshot"),
        memory_snapshot_hash=_hash(frozen["snapshot_hash"], "memory snapshot hash"),
        memory_snapshot_artifact_hash=_hash(
            frozen["snapshot_artifact_hash"], "memory snapshot artifact hash"
        ),
        primitive_registry=_path(frozen["primitive_registry"], "primitive registry"),
        primitive_registry_hash=_hash(frozen["primitive_registry_hash"], "primitive registry hash"),
        primitive_registry_artifact_hash=_hash(
            frozen["primitive_registry_artifact_hash"], "primitive registry artifact hash"
        ),
        conditions=conditions,
        split=split,
        task_ids=tuple(cast(list[str], ids_raw)),
        search_seeds=tuple(cast(list[int], seeds_raw)),
        requests_per_child=_integer(matched["requests_per_child"], "requests per child", minimum=1),
        batch_size=1,
        input_token_cap=_integer(matched["input_token_cap"], "input token cap", minimum=1),
        output_token_cap=_integer(matched["output_token_cap"], "output token cap", minimum=1),
        total_token_cap=_integer(matched["total_token_cap"], "total token cap", minimum=1),
        proposal_item_cap=_integer(matched["proposal_item_cap"], "proposal item cap", minimum=1),
        oracle_call_cap=_integer(matched["oracle_call_cap"], "oracle call cap", minimum=1),
        model=str(matched["model"]),
        endpoint=str(matched["endpoint"]),
        service_tier=str(matched["service_tier"]),
        reasoning_effort=str(matched["reasoning_effort"]),
        max_output_tokens=2048,
        backend_id=str(matched["backend"]),
        provider_id=str(matched["provider"]),
        retrieval_max_items=_integer(retrieval["max_items"], "retrieval max items"),
        retrieval_max_bytes=_integer(retrieval["max_bytes"], "retrieval max bytes"),
        retrieval_max_tokens=_integer(retrieval["max_tokens"], "retrieval max tokens"),
        output_root=_path(storage["output_root"], "output root"),
        cache_root=_path(storage["cache_root"], "cache root"),
        cache_namespace=_string(storage["cache_namespace"], "cache namespace"),
        cash_ledger=_path(storage["cash_ledger"], "cash ledger"),
        prerequisite_canary_registry=prerequisite_registry,
        prerequisite_canary_registry_hash=prerequisite_hash,
        prerequisite_canary_summary=prerequisite_summary,
    )
    if (
        experiment.requests_per_child * experiment.batch_size > experiment.proposal_item_cap
        or experiment.requests_per_child * experiment.batch_size > experiment.oracle_call_cap
        or experiment.requests_per_child * experiment.max_output_tokens
        > experiment.output_token_cap
        or experiment.input_token_cap + experiment.output_token_cap > experiment.total_token_cap
    ):
        raise ConfigurationError("Phase 5 live child caps cannot contain the frozen request plan")
    return experiment


@dataclass(frozen=True, slots=True)
class Phase5LiveAuthority:
    status: str
    experiment_hash: str
    exposure_policy_hash: str
    model_calls: bool
    oracle_access: bool
    user_reviewed_exposure_policy: bool
    user_authorized_live_run: bool
    authorization_evidence: str | None
    source_hash: str

    @property
    def authorized(self) -> bool:
        return (
            self.status == "authorized"
            and all(
                (
                    self.model_calls,
                    self.oracle_access,
                    self.user_reviewed_exposure_policy,
                    self.user_authorized_live_run,
                )
            )
            and bool(self.authorization_evidence)
        )


def load_phase5_live_authority(path: Path) -> Phase5LiveAuthority:
    root = _mapping(
        _yaml(path, "Phase 5 live authority"),
        {
            "authority_version",
            "authority_id",
            "status",
            "experiment",
            "experiment_hash",
            "exposure_policy",
            "exposure_policy_hash",
            "authorization",
            "authorization_evidence",
            "fail_closed",
        },
        "Phase 5 live authority",
    )
    if root["authority_version"] != LIVE_AUTHORITY_VERSION or root["fail_closed"] is not True:
        raise ConfigurationError("Phase 5 live authority version/fail-closed flag is invalid")
    status = _string(root["status"], "authority status")
    if status not in {"pending-user-review-and-authorization", "authorized"}:
        raise ConfigurationError("Phase 5 live authority status is invalid")
    authorization = _mapping(
        root["authorization"],
        {
            "model_calls",
            "oracle_access",
            "user_reviewed_exposure_policy",
            "user_authorized_live_run",
        },
        "live authorization",
    )
    evidence = root["authorization_evidence"]
    if evidence is not None and (not isinstance(evidence, str) or not evidence):
        raise ConfigurationError("authorization evidence must be null or nonempty")
    authority = Phase5LiveAuthority(
        status=status,
        experiment_hash=_hash(root["experiment_hash"], "authority experiment hash"),
        exposure_policy_hash=_hash(root["exposure_policy_hash"], "authority exposure hash"),
        model_calls=_boolean(authorization["model_calls"], "model call authority"),
        oracle_access=_boolean(authorization["oracle_access"], "oracle authority"),
        user_reviewed_exposure_policy=_boolean(
            authorization["user_reviewed_exposure_policy"], "policy review authority"
        ),
        user_authorized_live_run=_boolean(
            authorization["user_authorized_live_run"], "live run authority"
        ),
        authorization_evidence=evidence,
        source_hash=sha256_json(root),
    )
    if status != "authorized" and any(
        (
            authority.model_calls,
            authority.oracle_access,
            authority.user_reviewed_exposure_policy,
            authority.user_authorized_live_run,
        )
    ):
        raise ConfigurationError("pending Phase 5 authority must deny every capability")
    if status == "authorized" and not authority.authorized:
        raise ConfigurationError("authorized Phase 5 authority is incomplete")
    return authority


def _validate_freeze(
    repository_root: Path, experiment: Phase5LiveExperiment
) -> tuple[MemorySnapshot, PrimitiveRegistry, Phase5ExposurePolicy, PricePolicy]:
    transfer = load_transfer_registry(repository_root / experiment.transfer_registry)
    if transfer.content_hash != experiment.transfer_registry_hash:
        raise ConfigurationError("live experiment transfer registry hash differs")
    exposure = load_phase5_exposure_policy(repository_root / experiment.exposure_policy)
    if exposure.source_hash != experiment.exposure_policy_hash:
        raise ConfigurationError("live experiment exposure policy hash differs")
    price = load_price_policy(repository_root / exposure.parent_cash_policy)
    if price.content_hash != exposure.parent_cash_policy_hash:
        raise ConfigurationError("Phase 5 parent cash policy hash differs")
    snapshot_path = repository_root / experiment.memory_snapshot
    primitive_path = repository_root / experiment.primitive_registry
    if sha256_text(read_text_artifact(snapshot_path)) != experiment.memory_snapshot_artifact_hash:
        raise ConfigurationError("frozen memory snapshot artifact hash differs")
    if (
        sha256_text(read_text_artifact(primitive_path))
        != experiment.primitive_registry_artifact_hash
    ):
        raise ConfigurationError("frozen primitive registry artifact hash differs")
    snapshot = load_memory_snapshot(snapshot_path)
    primitives = load_primitive_registry(primitive_path)
    if (
        snapshot.snapshot_hash != experiment.memory_snapshot_hash
        or primitives.registry_hash != experiment.primitive_registry_hash
        or snapshot.split_registry_hash != transfer.content_hash
        or primitives.split_registry_hash != transfer.content_hash
    ):
        raise ConfigurationError("frozen Phase 5 memory/primitive identities differ")
    return snapshot, primitives, exposure, price


def _ledger_committed_exposure(ledger: ProjectLedger, stage: str) -> tuple[int, int]:
    total = 0
    selected_stage = 0
    for row in ledger.records():
        amount = int(row["actual_nano_usd"]) + int(row["uncertain_nano_usd"])
        if row["state"] == "active":
            amount += int(row["reserved_nano_usd"])
        total += amount
        if row["stage"] == stage:
            selected_stage += amount
    return total, selected_stage


def phase5_live_dry_run(
    *, repository_root: Path, registry_path: Path, authority_path: Path
) -> JsonObject:
    experiment = load_phase5_live_experiment(registry_path)
    authority = load_phase5_live_authority(authority_path)
    snapshot, primitives, exposure, price = _validate_freeze(repository_root, experiment)
    if authority.experiment_hash != experiment.source_hash:
        raise ConfigurationError("Phase 5 authority binds another experiment")
    if authority.exposure_policy_hash != exposure.source_hash:
        raise ConfigurationError("Phase 5 authority binds another exposure policy")
    transfer = load_transfer_registry(repository_root / experiment.transfer_registry)
    benchmark = generate_transfer_benchmark(repository_root, transfer)
    indexed = {
        str(item["task_id"]): item
        for item in cast(list[dict[str, object]], benchmark.manifest["tasks"])
    }
    if set(experiment.task_ids) - set(indexed) or any(
        indexed[task_id].get("split") != experiment.split.value for task_id in experiment.task_ids
    ):
        raise ConfigurationError("Phase 5 live task selection crosses its frozen role")
    max_identity_input = 0
    for task_id in experiment.task_ids:
        task = load_transfer_public_task(benchmark.root, task_id).public_view()
        rendered: dict[str, str] = {}
        for condition in experiment.conditions:
            active_snapshot = snapshot if condition == CONDITION_D else _empty_snapshot(experiment)
            active_primitives = (
                primitives
                if condition == CONDITION_D
                else empty_primitive_registry(
                    experiment.transfer_registry_hash, primitives.analysis_plan_hash
                )
            )
            retrieval = retrieve_memory(
                task=task,
                snapshot=active_snapshot,
                public_search_state={
                    "search_stage": "independent-live-proposals",
                    "evaluations": 0,
                },
                max_items=experiment.retrieval_max_items,
                max_bytes=experiment.retrieval_max_bytes,
                max_tokens=experiment.retrieval_max_tokens,
            )
            schema = primitive_candidate_batch_json_schema(
                role=ProposalRole.TRANSFER,
                batch_size=experiment.batch_size,
                registry=active_primitives,
            )
            prompt = render_phase5_prompt(
                task=task,
                role=ProposalRole.TRANSFER,
                requested_batch_size=experiment.batch_size,
                retrieval=retrieval,
                primitives=active_primitives,
            )
            rendered[condition] = prompt
            bindings = Phase5RequestBindings(
                experiment.transfer_registry_hash,
                active_snapshot.database_export_hash,
                active_snapshot.snapshot_hash,
                sha256_json(retrieval.to_value()),
                active_primitives.registry_hash,
                sha256_json(schema),
                sha256_json(
                    {
                        "exposure_policy_hash": exposure.source_hash,
                        "cash_policy_hash": price.content_hash,
                    }
                ),
                experiment.source_hash,
            )
            request = Phase5ModelRequest(
                experiment.backend_id,
                experiment.provider_id,
                experiment.model,
                experiment.endpoint,
                experiment.service_tier,
                experiment.reasoning_effort,
                experiment.max_output_tokens,
                experiment.batch_size,
                ProposalRole.TRANSFER,
                prompt,
                schema,
                bindings,
                {"search_seed": experiment.search_seeds[0], "independent_sample_index": 0},
            )
            max_identity_input = max(max_identity_input, request.conservative_input_token_bound)
        if CONDITION_C in rendered and CONDITION_D in rendered:
            assert_matched_prompt_isolation(rendered[CONDITION_C], rendered[CONDITION_D])
    if max_identity_input > 12_000:
        raise ConfigurationError("Phase 5 request identity exceeds its frozen input bound")
    request_max = price.price.maximum_cost(
        input_token_bound=12_000,
        max_output_tokens=experiment.max_output_tokens,
    )
    child_max = request_max * experiment.requests_per_child
    aggregate_max = child_max * experiment.child_count
    stage_cap = (
        exposure.canary_stage_nano_usd
        if experiment.stage == "training-canary"
        else exposure.development_stage_nano_usd
    )
    if (
        request_max > exposure.request_nano_usd
        or child_max > exposure.child_nano_usd
        or aggregate_max > stage_cap
        or aggregate_max > exposure.phase5_total_nano_usd
    ):
        raise ConfigurationError("Phase 5 live forecast exceeds its exposure partition")
    ledger_stage = "canary" if experiment.stage == "training-canary" else "pilot"
    with ProjectLedger(repository_root / experiment.cash_ledger, price) as ledger:
        ledger_status = ledger.status()
        phase_committed, stage_committed = _ledger_committed_exposure(ledger, ledger_stage)
    cash = ledger_status.get("cash_budget")
    if not isinstance(cash, dict):
        raise ConfigurationError("Phase 5 requires the reconciled-cash ledger")
    cash_upper = _integer(cash.get("cash_upper_bound_nano_usd"), "cash upper bound")
    fits_cash = cash_upper + aggregate_max <= exposure.personal_cash_ceiling_nano_usd
    if stage_committed + aggregate_max > stage_cap:
        raise ConfigurationError("Phase 5 live forecast exceeds remaining stage exposure")
    if phase_committed + aggregate_max > exposure.phase5_total_nano_usd:
        raise ConfigurationError("Phase 5 live forecast exceeds remaining total exposure")
    return {
        "dry_run_version": "phase5-live-preflight-v1",
        "experiment_id": experiment.experiment_id,
        "experiment_hash": experiment.source_hash,
        "stage": experiment.stage,
        "task_count": len(experiment.task_ids),
        "child_count": experiment.child_count,
        "requests_per_child": experiment.requests_per_child,
        "total_request_cap": experiment.total_request_cap,
        "request_nano_usd_max": request_max,
        "maximum_observed_request_identity_input_bound": max_identity_input,
        "child_nano_usd_max": child_max,
        "aggregate_nano_usd_max": aggregate_max,
        "existing_phase5_committed_nano_usd": phase_committed,
        "existing_stage_committed_nano_usd": stage_committed,
        "memory_snapshot_hash": snapshot.snapshot_hash,
        "primitive_registry_hash": primitives.registry_hash,
        "ledger_status": ledger_status,
        "cash_upper_bound_after_forecast_nano_usd": cash_upper + aggregate_max,
        "fits_current_cash_headroom": fits_cash,
        "authority_status": authority.status,
        "live_authorized": authority.authorized,
        "provider_calls": 0,
        "oracle_accesses": 0,
        "sealed_test_accesses": 0,
    }


def _check_canary_prerequisite(repository_root: Path, experiment: Phase5LiveExperiment) -> None:
    if experiment.stage != "development-pilot":
        return
    if (
        experiment.prerequisite_canary_registry is None
        or experiment.prerequisite_canary_registry_hash is None
        or experiment.prerequisite_canary_summary is None
    ):
        raise ConfigurationError("development pilot has no frozen canary prerequisite")
    canary = load_phase5_live_experiment(repository_root / experiment.prerequisite_canary_registry)
    if canary.source_hash != experiment.prerequisite_canary_registry_hash:
        raise ConfigurationError("development pilot canary registry hash differs")
    try:
        summary: object = json.loads(
            (repository_root / experiment.prerequisite_canary_summary).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("successful Phase 5 canary summary is unavailable") from exc
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "passed-live-training-canary"
        or summary.get("experiment_hash") != canary.source_hash
        or int(summary.get("valid_candidates", 0)) < 1
    ):
        raise ConfigurationError("Phase 5 training canary did not pass the frozen gate")


def _empty_snapshot(experiment: Phase5LiveExperiment) -> MemorySnapshot:
    return MemorySnapshot(
        experiment.transfer_registry_hash,
        sha256_json(
            {"empty_memory_version": CONDITION_C, "experiment_hash": experiment.source_hash}
        ),
        (),
    )


def _live_code_hash(repository_root: Path) -> str:
    source_root = repository_root / "src/world_model_search"
    if source_root.is_dir():
        files = [
            {
                "path": str(path.relative_to(repository_root)),
                "content_hash": sha256_text(path.read_text(encoding="utf-8")),
            }
            for path in sorted(source_root.rglob("*.py"))
        ]
    else:
        package_root = Path(__file__).parents[1]
        files = [
            {"path": path.name, "content_hash": sha256_text(path.read_text(encoding="utf-8"))}
            for path in (
                Path(__file__),
                package_root / "model/phase5_prompts.py",
                package_root / "dsl/primitive_schema.py",
            )
        ]
    return sha256_json({"code_hash_version": "phase5-live-source-tree-v1", "files": files})


def _child_order(experiment: Phase5LiveExperiment) -> tuple[tuple[str, int, str], ...]:
    rows: list[tuple[str, int, str]] = []
    for task_index, task_id in enumerate(experiment.task_ids):
        for seed_index, seed in enumerate(experiment.search_seeds):
            conditions = list(experiment.conditions)
            if len(conditions) > 1 and (task_index + seed_index) % 2:
                conditions.reverse()
            rows.extend((task_id, seed, condition) for condition in conditions)
    return tuple(rows)


def _response_artifact(response: ModelResponse, request: Phase5ModelRequest) -> JsonObject:
    return {
        "artifact_version": "phase5-live-model-response-v1",
        "provider_id": request.provider_id,
        "request_hash": request.request_hash,
        "response": response.deterministic_value(),
        "diagnostics": {"provider_latency_ns": response.provider_latency_ns},
    }


def _run_child(
    *,
    repository_root: Path,
    experiment: Phase5LiveExperiment,
    snapshot: MemorySnapshot,
    primitives: PrimitiveRegistry,
    exposure: Phase5ExposurePolicy,
    price: PricePolicy,
    ledger: ProjectLedger,
    backend: ModelBackend,
    task_id: str,
    seed: int,
    condition: str,
) -> JsonObject:
    child_id = sha256_json(
        {
            "child_identity_version": "phase5-live-child-v1",
            "experiment_hash": experiment.source_hash,
            "task_id": task_id,
            "seed": seed,
            "condition": condition,
        }
    )[:24]
    child_root = repository_root / experiment.output_root / "children" / child_id
    summary_path = child_root / "summary.json"
    if summary_path.is_file():
        value: object = json.loads(read_text_artifact(summary_path))
        if not isinstance(value, dict) or value.get("experiment_hash") != experiment.source_hash:
            raise PersistenceError("completed Phase 5 child binds another experiment")
        return cast(JsonObject, value)
    transfer = load_transfer_registry(repository_root / experiment.transfer_registry)
    benchmark = generate_transfer_benchmark(repository_root, transfer)
    store = Phase5TaskStore(benchmark.root)
    hidden = store.load(
        task_id,
        allowed_splits=frozenset({experiment.split}),
        purpose=f"phase5-live-{experiment.stage}",
    )
    task = load_transfer_public_task(benchmark.root, task_id).public_view()
    active_snapshot = snapshot if condition == CONDITION_D else _empty_snapshot(experiment)
    active_primitives = (
        primitives
        if condition == CONDITION_D
        else empty_primitive_registry(
            experiment.transfer_registry_hash, primitives.analysis_plan_hash
        )
    )
    retrieval = retrieve_memory(
        task=task,
        snapshot=active_snapshot,
        public_search_state={"search_stage": "independent-live-proposals", "evaluations": 0},
        max_items=experiment.retrieval_max_items,
        max_bytes=experiment.retrieval_max_bytes,
        max_tokens=experiment.retrieval_max_tokens,
    )
    schema = primitive_candidate_batch_json_schema(
        role=ProposalRole.TRANSFER,
        batch_size=experiment.batch_size,
        registry=active_primitives,
    )
    prompt = render_phase5_prompt(
        task=task,
        role=ProposalRole.TRANSFER,
        requested_batch_size=experiment.batch_size,
        retrieval=retrieval,
        primitives=active_primitives,
    )
    code_hash = _live_code_hash(repository_root)
    manifest: JsonObject = {
        "manifest_version": "phase5-live-child-manifest-v1",
        "experiment_hash": experiment.source_hash,
        "authority_role": experiment.split.value,
        "task_id": task_id,
        "seed": seed,
        "condition": condition,
        "memory_snapshot_hash": active_snapshot.snapshot_hash,
        "retrieval": retrieval.to_value(),
        "primitive_registry_hash": active_primitives.registry_hash,
        "prompt_hash": sha256_text(prompt),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "schema_hash": sha256_json(schema),
        "code_hash": code_hash,
        "request_cap": experiment.requests_per_child,
    }
    write_content_artifact(child_root / "manifest.json", canonical_json(manifest))
    cache = ExactResponseCache(repository_root / experiment.cache_root, experiment.cache_namespace)
    results: list[JsonObject] = []
    physical_calls = cache_hits = valid = invalid = oracle_calls = 0
    input_tokens = output_tokens = total_tokens = actual_nano = 0
    for request_index in range(experiment.requests_per_child):
        result_path = child_root / "results" / f"request-{request_index:05d}.json"
        request_path = child_root / "requests" / f"request-{request_index:05d}.json"
        response_path = child_root / "responses" / f"request-{request_index:05d}.json"
        bindings = Phase5RequestBindings(
            experiment.transfer_registry_hash,
            active_snapshot.database_export_hash,
            active_snapshot.snapshot_hash,
            sha256_json(retrieval.to_value()),
            active_primitives.registry_hash,
            sha256_json(schema),
            sha256_json(
                {
                    "exposure_policy_hash": exposure.source_hash,
                    "cash_policy_hash": price.content_hash,
                }
            ),
            sha256_json(
                {
                    "code_hash": code_hash,
                    "experiment_hash": experiment.source_hash,
                    "child_manifest": manifest,
                }
            ),
        )
        request = Phase5ModelRequest(
            experiment.backend_id,
            experiment.provider_id,
            experiment.model,
            experiment.endpoint,
            experiment.service_tier,
            experiment.reasoning_effort,
            experiment.max_output_tokens,
            experiment.batch_size,
            ProposalRole.TRANSFER,
            prompt,
            schema,
            bindings,
            {"search_seed": seed, "independent_sample_index": request_index},
        )
        request_record: JsonObject = {
            "artifact_version": "phase5-live-model-request-v1",
            "request_hash": request.request_hash,
            "identity": request.identity_value(),
        }
        if result_path.is_file():
            prior: object = json.loads(read_text_artifact(result_path))
            if not isinstance(prior, dict) or prior.get("request_hash") != request.request_hash:
                raise PersistenceError("Phase 5 completed request identity differs")
            result = cast(JsonObject, prior)
            results.append(result)
            usage = cast(dict[str, object], result["usage"])
            physical_calls += _integer(result.get("physical_provider_call"), "provider calls")
            cache_hits += _integer(result.get("cache_hit"), "cache hits")
            valid += _integer(result.get("valid_candidates"), "valid candidates")
            invalid += _integer(result.get("invalid_candidates"), "invalid candidates")
            oracle_calls += _integer(result.get("oracle_calls"), "oracle calls")
            input_tokens += _integer(usage.get("input_tokens"), "input tokens")
            output_tokens += _integer(usage.get("output_tokens"), "output tokens")
            total_tokens += _integer(usage.get("total_tokens"), "total tokens")
            actual_nano += _integer(result.get("published_rate_nano_usd"), "published cost")
            continue
        if request_path.exists():
            raise PersistenceError(
                "ambiguous incomplete Phase 5 request may already be paid; do not duplicate"
            )
        if input_tokens + request.conservative_input_token_bound > experiment.input_token_cap:
            raise BudgetExhaustedError("Phase 5 child input-token preflight cap exhausted")
        if output_tokens + experiment.max_output_tokens > experiment.output_token_cap:
            raise BudgetExhaustedError("Phase 5 child output-token preflight cap exhausted")
        if (
            total_tokens + request.conservative_input_token_bound + experiment.max_output_tokens
            > experiment.total_token_cap
        ):
            raise BudgetExhaustedError("Phase 5 child total-token preflight cap exhausted")
        maximum_cost = price.price.maximum_cost(
            input_token_bound=request.conservative_input_token_bound,
            max_output_tokens=experiment.max_output_tokens,
        )
        if maximum_cost > exposure.request_nano_usd:
            raise BudgetExhaustedError("Phase 5 request exceeds its exposure ceiling")
        cached = cache.get(request)
        reservation_id = sha256_text(
            f"phase5-live-reservation-v1\0{child_id}\0{request_index}\0{request.request_hash}"
        )
        if cached is None:
            ledger.reserve(
                reservation_id=reservation_id,
                run_id=child_id,
                stage="canary" if experiment.stage == "training-canary" else "pilot",
                request_hash=request.request_hash,
                amount_nano_usd=maximum_cost,
                child_cap_nano_usd=exposure.child_nano_usd,
                phase_cap_override_nano_usd=exposure.phase5_total_nano_usd,
                stage_cap_override_nano_usd=(
                    exposure.canary_stage_nano_usd
                    if experiment.stage == "training-canary"
                    else exposure.development_stage_nano_usd
                ),
            )
        write_content_artifact(request_path, canonical_json(request_record))
        if cached is None:
            try:
                response = backend.dispatch(request)
            except ModelDispatchError as exc:
                failure: JsonObject = {
                    "artifact_version": "phase5-live-model-failure-v1",
                    "request_hash": request.request_hash,
                    "error": exc.error.to_value(),
                }
                failure_hash = write_content_artifact(
                    child_root / "responses" / f"request-{request_index:05d}-failure.json",
                    canonical_json(failure),
                )
                if exc.error.usage_uncertain:
                    ledger.mark_uncertain(
                        reservation_id=reservation_id,
                        failure_record={
                            "child_id": child_id,
                            "request_index": request_index,
                            "request_hash": request.request_hash,
                            "failure_hash": failure_hash,
                        },
                    )
                else:
                    ledger.reconcile(
                        reservation_id=reservation_id,
                        actual_nano_usd=0,
                        usage_record={
                            "child_id": child_id,
                            "request_index": request_index,
                            "request_hash": request.request_hash,
                            "failure_hash": failure_hash,
                            "actual_nano_usd": 0,
                        },
                    )
                raise PersistenceError(
                    f"Phase 5 provider dispatch failed closed: {exc.error.category.value}"
                ) from exc
            physical = 1
        else:
            response = cached
            physical = 0
        response_hash = write_content_artifact(
            response_path, canonical_json(_response_artifact(response, request))
        )
        charge = price.price.cost(response.usage) if cached is None else 0
        if cached is None:
            ledger.reconcile(
                reservation_id=reservation_id,
                actual_nano_usd=charge,
                usage_record={
                    "child_id": child_id,
                    "request_index": request_index,
                    "request_hash": request.request_hash,
                    "response_hash": response_hash,
                    "usage": response.usage.to_value(),
                    "actual_nano_usd": charge,
                    "price_policy_hash": price.content_hash,
                    "exposure_policy_hash": exposure.source_hash,
                },
            )
            cache.put(request, response)
        try:
            batch = parse_primitive_candidate_batch(
                response.raw_text,
                expected_role=ProposalRole.TRANSFER,
                requested_batch_size=experiment.batch_size,
                registry=active_primitives,
                limits=AstLimits(),
            )
            schema_error: str | None = None
        except PrimitiveCandidateJsonError as exc:
            batch = None
            schema_error = str(exc)
        evaluations: list[JsonObject] = []
        accepted = rejected = 0
        if batch is not None:
            for item in batch.items:
                if not item.accepted or item.expanded_ast is None or item.source_ast is None:
                    rejected += 1
                    evaluations.append(
                        {
                            "ordinal": item.ordinal,
                            "accepted": False,
                            "rejection_reason": item.rejection_reason,
                        }
                    )
                    continue
                evaluated = ExactDslOracle(
                    hidden.oracle_bundle,
                    limits=AstLimits(),
                    response_mode=OracleResponseMode.SCORE_ONLY,
                ).evaluate(item.expanded_ast)
                accepted += 1
                evaluations.append(
                    {
                        "ordinal": item.ordinal,
                        "accepted": True,
                        "exact": evaluated.result.exact,
                        "score": evaluated.result.local_cases - evaluated.result.local_errors,
                        "program_bits": len(encode_program(item.source_ast, active_primitives)),
                        "base_expanded_bits": encoded_length(item.expanded_ast),
                    }
                )
        result = cast(
            JsonObject,
            {
                "result_version": "phase5-live-request-result-v1",
                "request_hash": request.request_hash,
                "response_hash": response_hash,
                "cache_hit": int(cached is not None),
                "physical_provider_call": physical,
                "usage": response.usage.to_value(),
                "published_rate_nano_usd": charge,
                "valid_candidates": accepted,
                "invalid_candidates": rejected,
                "oracle_calls": accepted,
                "schema_error": schema_error,
                "evaluations": evaluations,
            },
        )
        write_content_artifact(result_path, canonical_json(result))
        results.append(result)
        physical_calls += physical
        cache_hits += int(cached is not None)
        valid += accepted
        invalid += rejected
        oracle_calls += accepted
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        total_tokens += response.usage.total_tokens
        actual_nano += charge
    exact = any(
        bool(evaluation.get("exact"))
        for result in results
        for evaluation in cast(list[dict[str, object]], result["evaluations"])
    )
    best_score = max(
        (
            _integer(evaluation.get("score", 0), "evaluation score")
            for result in results
            for evaluation in cast(list[dict[str, object]], result["evaluations"])
            if evaluation.get("accepted") is True
        ),
        default=0,
    )
    exact_program_bits = [
        _integer(evaluation.get("program_bits"), "exact program bits")
        for result in results
        for evaluation in cast(list[dict[str, object]], result["evaluations"])
        if evaluation.get("accepted") is True and evaluation.get("exact") is True
    ]
    summary: JsonObject = {
        "summary_version": "phase5-live-child-summary-v1",
        "experiment_hash": experiment.source_hash,
        "child_id": child_id,
        "task_id": task_id,
        "seed": seed,
        "condition": condition,
        "target_family": hidden.family_id,
        "status": "completed",
        "request_attempts": len(results),
        "physical_provider_calls": physical_calls,
        "cache_hits": cache_hits,
        "valid_candidates": valid,
        "invalid_candidates": invalid,
        "oracle_calls": oracle_calls,
        "exact_solved": exact,
        "best_score": best_score,
        "best_exact_program_bits": min(exact_program_bits) if exact_program_bits else None,
        "retrieval_hit": bool(retrieval.selected_record_ids),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "published_rate_nano_usd": actual_nano,
        "result_hashes": [sha256_json(result) for result in results],
        "sealed_test_accesses": 0,
    }
    write_content_artifact(summary_path, canonical_json(summary))
    return summary


def run_phase5_live_experiment(
    *,
    repository_root: Path,
    registry_path: Path,
    authority_path: Path,
    allow_live_model: bool,
    backend: ModelBackend | None = None,
) -> JsonObject:
    experiment = load_phase5_live_experiment(registry_path)
    authority = load_phase5_live_authority(authority_path)
    snapshot, primitives, exposure, price = _validate_freeze(repository_root, experiment)
    if (
        authority.experiment_hash != experiment.source_hash
        or authority.exposure_policy_hash != exposure.source_hash
        or not authority.authorized
    ):
        raise ConfigurationError("Phase 5 live experiment is pending explicit user authorization")
    if not allow_live_model:
        raise ConfigurationError("Phase 5 live execution requires the explicit CLI opt-in")
    _check_canary_prerequisite(repository_root, experiment)
    phase5_live_dry_run(
        repository_root=repository_root,
        registry_path=registry_path,
        authority_path=authority_path,
    )
    resolved_backend = backend
    if resolved_backend is None:
        try:
            resolved_backend = OpenAIResponsesBackend(opt_in=LiveOptIn.resolve(True))
        except ModelDispatchError as exc:
            raise ConfigurationError(
                "live model requires CLI approval, WMS_ALLOW_LIVE_MODEL=1, and OPENAI_API_KEY"
            ) from exc
    if (
        resolved_backend.backend_id != experiment.backend_id
        or resolved_backend.provider_id != experiment.provider_id
    ):
        raise ConfigurationError("Phase 5 backend identity differs from the frozen experiment")
    root = repository_root / experiment.output_root
    summary_path = root / "summary.json"
    if summary_path.is_file():
        value: object = json.loads(read_text_artifact(summary_path))
        if not isinstance(value, dict) or value.get("experiment_hash") != experiment.source_hash:
            raise PersistenceError("completed Phase 5 live experiment identity differs")
        return cast(JsonObject, value)
    children: list[JsonObject] = []
    with ProjectLedger(repository_root / experiment.cash_ledger, price) as ledger:
        for task_id, seed, condition in _child_order(experiment):
            children.append(
                _run_child(
                    repository_root=repository_root,
                    experiment=experiment,
                    snapshot=snapshot,
                    primitives=primitives,
                    exposure=exposure,
                    price=price,
                    ledger=ledger,
                    backend=resolved_backend,
                    task_id=task_id,
                    seed=seed,
                    condition=condition,
                )
            )
        ledger_status = ledger.status()
    valid = sum(_integer(child.get("valid_candidates"), "valid candidates") for child in children)
    status = (
        "passed-live-training-canary"
        if experiment.stage == "training-canary" and valid >= 1
        else "completed-live-development-pilot"
    )
    if experiment.stage == "training-canary" and valid < 1:
        status = "failed-live-training-canary"
    summary: JsonObject = {
        "summary_version": "phase5-live-experiment-summary-v1",
        "experiment_id": experiment.experiment_id,
        "experiment_hash": experiment.source_hash,
        "authority_hash": authority.source_hash,
        "stage": experiment.stage,
        "status": status,
        "child_count": len(children),
        "request_attempts": sum(
            _integer(child.get("request_attempts"), "request attempts") for child in children
        ),
        "physical_provider_calls": sum(
            _integer(child.get("physical_provider_calls"), "provider calls") for child in children
        ),
        "cache_hits": sum(_integer(child.get("cache_hits"), "cache hits") for child in children),
        "valid_candidates": valid,
        "invalid_candidates": sum(
            _integer(child.get("invalid_candidates"), "invalid candidates") for child in children
        ),
        "oracle_calls": sum(
            _integer(child.get("oracle_calls"), "oracle calls") for child in children
        ),
        "exact_solved_children": sum(1 for child in children if child.get("exact_solved") is True),
        "published_rate_nano_usd": sum(
            _integer(child.get("published_rate_nano_usd"), "published cost") for child in children
        ),
        "ledger_status": ledger_status,
        "sealed_test_accesses": 0,
        "scientific_status": (
            "compatibility-only-not-scientific-evidence"
            if experiment.stage == "training-canary"
            else "development-evidence-only-h3-unconfirmed"
        ),
    }
    write_content_artifact(root / "children.json", canonical_json({"children": children}))
    if experiment.stage == "development-pilot":
        pairs: list[JsonObject] = []
        by_identity = {
            (
                str(child["task_id"]),
                _integer(child.get("seed"), "child seed"),
                str(child["condition"]),
            ): child
            for child in children
        }
        for task_id in experiment.task_ids:
            for seed in experiment.search_seeds:
                off = by_identity[(task_id, seed, CONDITION_C)]
                on = by_identity[(task_id, seed, CONDITION_D)]
                pairs.append(
                    cast(
                        JsonObject,
                        {
                            "task_id": task_id,
                            "seed": seed,
                            "target_family": off["target_family"],
                            "condition_c_exact": off["exact_solved"],
                            "condition_d_exact": on["exact_solved"],
                            "exact_difference_d_minus_c": int(bool(on["exact_solved"]))
                            - int(bool(off["exact_solved"])),
                            "score_difference_d_minus_c": _integer(
                                on.get("best_score"), "condition D best score"
                            )
                            - _integer(off.get("best_score"), "condition C best score"),
                            "published_rate_difference_d_minus_c_nano_usd": _integer(
                                on.get("published_rate_nano_usd"), "condition D published cost"
                            )
                            - _integer(
                                off.get("published_rate_nano_usd"), "condition C published cost"
                            ),
                            "gross_program_savings_bits": (
                                _integer(
                                    off.get("best_exact_program_bits"),
                                    "condition C exact program bits",
                                )
                                - _integer(
                                    on.get("best_exact_program_bits"),
                                    "condition D exact program bits",
                                )
                                if isinstance(off.get("best_exact_program_bits"), int)
                                and isinstance(on.get("best_exact_program_bits"), int)
                                else None
                            ),
                        },
                    )
                )
        comparable_savings = [
            _integer(pair.get("gross_program_savings_bits"), "paired program savings")
            for pair in pairs
            if isinstance(pair.get("gross_program_savings_bits"), int)
        ]
        definition_cost = library_definition_cost(primitives.definitions)
        family_matrix: list[JsonObject] = []
        for family in sorted({str(pair["target_family"]) for pair in pairs}):
            family_savings = [
                _integer(pair.get("gross_program_savings_bits"), "family program savings")
                for pair in pairs
                if pair["target_family"] == family
                and isinstance(pair.get("gross_program_savings_bits"), int)
            ]
            family_matrix.append(
                {
                    "target_family": family,
                    "comparable_pair_count": len(family_savings),
                    "gross_program_length_savings_bits": sum(family_savings),
                    "definition_cost_allocated_to_cell_bits": 0,
                    "negative_transfer": sum(family_savings) < 0,
                }
            )
        analysis = cast(
            JsonObject,
            {
                "analysis_version": "phase5-live-development-analysis-v1",
                "data_role": "development-pilot",
                "confirmatory": False,
                "h3_confirmed": False,
                "paired_row_count": len(pairs),
                "comparable_exact_pair_count": len(comparable_savings),
                "pairs": pairs,
                "transfer_matrix": family_matrix,
                "aggregate_gross_program_savings_bits": sum(comparable_savings),
                "library_definition_cost_bits_charged_once": definition_cost,
                "aggregate_net_two_part_gain_bits": sum(comparable_savings) - definition_cost,
                "retrieval_precision": (
                    sum(
                        1
                        for child in children
                        if child["condition"] == CONDITION_D and child["retrieval_hit"] is True
                    )
                    / sum(1 for child in children if child["condition"] == CONDITION_D)
                ),
                "condition_c_exact_solved": sum(
                    1
                    for child in children
                    if child["condition"] == CONDITION_C and child["exact_solved"] is True
                ),
                "condition_d_exact_solved": sum(
                    1
                    for child in children
                    if child["condition"] == CONDITION_D and child["exact_solved"] is True
                ),
                "limitation": (
                    "development families were used by the primitive-promotion gate; this pilot "
                    "cannot independently confirm H3"
                ),
            },
        )
        summary["analysis_hash"] = write_content_artifact(
            root / "analysis.json", canonical_json(analysis)
        )
    else:
        summary["analysis_hash"] = None
    write_content_artifact(summary_path, canonical_json(summary))
    return summary


def replay_phase5_live_experiment(*, repository_root: Path, registry_path: Path) -> JsonObject:
    """Provider-disabled integrity replay over a completed canary or development pilot."""

    experiment = load_phase5_live_experiment(registry_path)
    _validate_freeze(repository_root, experiment)
    root = repository_root / experiment.output_root
    try:
        summary: object = json.loads(read_text_artifact(root / "summary.json"))
        children_value: object = json.loads(read_text_artifact(root / "children.json"))
    except (json.JSONDecodeError, PersistenceError) as exc:
        raise ReplayError("Phase 5 live replay artifacts are unavailable") from exc
    if (
        not isinstance(summary, dict)
        or not isinstance(children_value, dict)
        or not isinstance(children_value.get("children"), list)
        or summary.get("experiment_hash") != experiment.source_hash
    ):
        raise ReplayError("Phase 5 live replay root identity differs")
    children = cast(list[dict[str, object]], children_value["children"])
    if len(children) != experiment.child_count:
        raise ReplayError("Phase 5 live replay child count differs")
    for child in children:
        child_id = str(child.get("child_id"))
        child_root = root / "children" / child_id
        stored: object = json.loads(read_text_artifact(child_root / "summary.json"))
        if stored != child:
            raise ReplayError("Phase 5 live child summary differs")
        expected_result_hashes = child.get("result_hashes")
        if (
            not isinstance(expected_result_hashes, list)
            or len(expected_result_hashes) != experiment.requests_per_child
        ):
            raise ReplayError("Phase 5 live child result-hash index differs")
        for request_index in range(experiment.requests_per_child):
            request: object = json.loads(
                read_text_artifact(child_root / "requests" / f"request-{request_index:05d}.json")
            )
            response: object = json.loads(
                read_text_artifact(child_root / "responses" / f"request-{request_index:05d}.json")
            )
            result: object = json.loads(
                read_text_artifact(child_root / "results" / f"request-{request_index:05d}.json")
            )
            if (
                not isinstance(request, dict)
                or not isinstance(response, dict)
                or not isinstance(result, dict)
                or not isinstance(request.get("identity"), dict)
                or sha256_json(request["identity"]) != request.get("request_hash")
                or request.get("request_hash") != response.get("request_hash")
                or request.get("request_hash") != result.get("request_hash")
                or sha256_text(canonical_json(response)) != result.get("response_hash")
                or sha256_json(result) != expected_result_hashes[request_index]
            ):
                raise ReplayError("Phase 5 live request/response/result identity differs")
    aggregates = {
        "request_attempts": sum(
            _integer(child.get("request_attempts"), "replay request attempts") for child in children
        ),
        "physical_provider_calls": sum(
            _integer(child.get("physical_provider_calls"), "replay provider calls")
            for child in children
        ),
        "cache_hits": sum(
            _integer(child.get("cache_hits"), "replay cache hits") for child in children
        ),
        "valid_candidates": sum(
            _integer(child.get("valid_candidates"), "replay valid candidates") for child in children
        ),
        "invalid_candidates": sum(
            _integer(child.get("invalid_candidates"), "replay invalid candidates")
            for child in children
        ),
        "oracle_calls": sum(
            _integer(child.get("oracle_calls"), "replay oracle calls") for child in children
        ),
        "published_rate_nano_usd": sum(
            _integer(child.get("published_rate_nano_usd"), "replay published cost")
            for child in children
        ),
    }
    if any(summary.get(key) != value for key, value in aggregates.items()):
        raise ReplayError("Phase 5 live aggregate summary differs from child artifacts")
    if experiment.stage == "development-pilot":
        analysis_text = read_text_artifact(root / "analysis.json")
        if sha256_text(analysis_text) != summary.get("analysis_hash"):
            raise ReplayError("Phase 5 live development analysis hash differs")
    elif summary.get("analysis_hash") is not None:
        raise ReplayError("Phase 5 canary unexpectedly binds development analysis")
    return {
        "replay_version": "phase5-live-provider-disabled-replay-v1",
        "status": "verified-provider-disabled",
        "experiment_hash": experiment.source_hash,
        "child_count": len(children),
        "provider_calls": 0,
        "oracle_accesses": 0,
        "sealed_test_accesses": 0,
        "summary_hash": sha256_json(summary),
    }
