"""No-cost Phase 5 smoke, exact replay, transfer analysis, and budget forecast."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import cast

import yaml

from world_model_search.domain.types import OracleResponseMode, ProposalRole, SplitLabel
from world_model_search.dsl.ast import AstLimits, BitExpr
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.primitive_schema import primitive_candidate_batch_json_schema
from world_model_search.dsl.primitives import (
    PrimitiveDefinition,
    PrimitiveRegistry,
    decode_library,
    decode_program,
    empty_primitive_registry,
    encode_library,
    encode_program,
    expand_primitives,
    library_definition_cost,
    replace_subtree,
    write_primitive_registry,
)
from world_model_search.errors import ConfigurationError, PersistenceError, ReplayError
from world_model_search.evaluation.phase5_transfer import (
    Phase5TaskStore,
    generate_transfer_benchmark,
    load_transfer_public_task,
    load_transfer_registry,
    selector_core,
)
from world_model_search.memory.retrieval import RetrievalRecord, retrieve_memory
from world_model_search.memory.store import Phase5MemoryStore
from world_model_search.memory.types import (
    EvidenceFact,
    EvidencePolarity,
    MemoryApplicability,
    MemoryKind,
    MemorySnapshot,
    SafeMemoryItem,
    ValidationState,
)
from world_model_search.model.phase5_prompts import (
    Phase5ModelRequest,
    Phase5RequestBindings,
    assert_matched_prompt_isolation,
    render_phase5_prompt,
)
from world_model_search.model.policy import load_price_policy
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.persistence.artifacts import (
    read_text_artifact,
    write_content_artifact,
)
from world_model_search.phase5_versions import (
    PHASE5_ANALYSIS_VERSION,
    PHASE5_EXPERIMENT_SCHEMA_VERSION,
    PHASE5_EXPERIMENT_VERSION,
    PHASE5_EXPOSURE_POLICY_VERSION,
    PHASE5_REPLAY_VERSION,
    PHASE5_REPORT_VERSION,
)
from world_model_search.serialization import JsonObject, canonical_json, sha256_json, sha256_text

CONDITION_C = "condition-c-empty-memory-v1"
CONDITION_D = "condition-d-typed-memory-v1"


@dataclass(frozen=True, slots=True)
class Phase5Experiment:
    registry_source_hash: str
    experiment_id: str
    output_root: Path
    transfer_registry_path: Path
    search_seeds: tuple[int, ...]
    candidate_cap: int
    oracle_call_cap: int
    model_request_cap: int
    retrieval_max_items: int
    retrieval_max_bytes: int
    retrieval_max_tokens: int
    bootstrap_seed: int
    bootstrap_replicates: int
    exposure_policy: Path
    cash_policy: Path
    cash_ledger: Path
    live_input_token_bound_per_request: int
    live_requests_per_child: int
    live_child_count: int
    live_runtime_bound_seconds: int

    @property
    def content_hash(self) -> str:
        return sha256_json(self.to_value())

    def to_value(self) -> JsonObject:
        return {
            "experiment_schema_version": PHASE5_EXPERIMENT_SCHEMA_VERSION,
            "experiment_version": PHASE5_EXPERIMENT_VERSION,
            "registry_source_hash": self.registry_source_hash,
            "experiment_id": self.experiment_id,
            "output_root": str(self.output_root),
            "transfer_registry": str(self.transfer_registry_path),
            "search_seeds": list(self.search_seeds),
            "caps": {
                "candidate": self.candidate_cap,
                "oracle_call": self.oracle_call_cap,
                "model_request": self.model_request_cap,
            },
            "retrieval": {
                "max_items": self.retrieval_max_items,
                "max_bytes": self.retrieval_max_bytes,
                "max_tokens": self.retrieval_max_tokens,
            },
            "analysis": {
                "version": PHASE5_ANALYSIS_VERSION,
                "bootstrap_seed": self.bootstrap_seed,
                "bootstrap_replicates": self.bootstrap_replicates,
            },
            "spending": {
                "exposure_policy": str(self.exposure_policy),
                "cash_policy": str(self.cash_policy),
                "cash_ledger": str(self.cash_ledger),
                "live_input_token_bound_per_request": self.live_input_token_bound_per_request,
                "live_requests_per_child": self.live_requests_per_child,
                "live_child_count": self.live_child_count,
                "live_runtime_bound_seconds": self.live_runtime_bound_seconds,
            },
        }


def _mapping(value: object, keys: set[str], location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping")
    if set(value) != keys:
        raise ConfigurationError(f"{location} has missing or unknown keys")
    return value


def _integer(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{location} must be an integer >= {minimum}")
    return value


def _json_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersistenceError(f"{location} must be an integer")
    return value


def _path(value: object, location: str) -> Path:
    if not isinstance(value, str):
        raise ConfigurationError(f"{location} must be a path")
    result = Path(value)
    if result.is_absolute() or ".." in result.parts:
        raise ConfigurationError(f"{location} must be repository-relative")
    return result


def load_phase5_experiment(path: Path) -> Phase5Experiment:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("Phase 5 experiment registry is unavailable") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("Phase 5 experiment registry must be a mapping")
    if (
        value.get("experiment_schema_version") != PHASE5_EXPERIMENT_SCHEMA_VERSION
        or value.get("experiment_version") != PHASE5_EXPERIMENT_VERSION
    ):
        raise ConfigurationError("unsupported Phase 5 experiment registry")
    if value.get("status") != "deterministic-development-smoke":
        raise ConfigurationError("Phase 5 implementation accepts only the no-cost smoke registry")
    if value.get("conditions") != [CONDITION_C, CONDITION_D]:
        raise ConfigurationError("Phase 5 conditions must be fresh matched C and D")
    roles = _mapping(
        value.get("data_roles"),
        {"primitive_training", "primitive_promotion", "paired_evaluation", "confirmatory_test"},
        "data_roles",
    )
    if roles != {
        "primitive_training": "training",
        "primitive_promotion": "development",
        "paired_evaluation": "development",
        "confirmatory_test": "sealed",
    }:
        raise ConfigurationError("Phase 5 data roles are not split-safe")
    matched = _mapping(
        value.get("matched_contract"),
        {
            "base_search_algorithm",
            "scheduler",
            "candidate_cap",
            "oracle_call_cap",
            "model_request_cap",
            "stopping_rule",
            "prompt_template",
            "response_schema",
            "model",
            "endpoint",
            "service_tier",
            "reasoning_effort",
            "max_output_tokens",
        },
        "matched_contract",
    )
    expected_model = {
        "model": "gpt-5-mini-2025-08-07",
        "endpoint": "v1/responses",
        "service_tier": "default",
        "reasoning_effort": "low",
        "max_output_tokens": 2048,
    }
    if any(matched.get(key) != expected for key, expected in expected_model.items()):
        raise ConfigurationError(
            "Phase 5 model snapshot/settings differ from the declared contract"
        )
    if matched.get("base_search_algorithm") != "deterministic-typed-recoding-smoke-v1":
        raise ConfigurationError("Phase 5 no-cost smoke algorithm is not recognized")
    retrieval = _mapping(
        value.get("retrieval"), {"max_items", "max_bytes", "max_tokens"}, "retrieval"
    )
    analysis = _mapping(
        value.get("analysis"),
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
    if analysis.get("primary_endpoint") != "net-held-out-two-part-code-length-gain":
        raise ConfigurationError("Phase 5 primary endpoint is not the proposal's MDL gate")
    spending = _mapping(
        value.get("spending"),
        {
            "exposure_policy",
            "cash_policy",
            "cash_ledger",
            "live_input_token_bound_per_request",
            "live_requests_per_child",
            "live_child_count",
            "live_runtime_bound_seconds",
        },
        "spending",
    )
    authorization = _mapping(
        value.get("authorization"),
        {"live_model", "development_oracle_for_smoke", "sealed_test_oracle"},
        "authorization",
    )
    if authorization != {
        "live_model": False,
        "development_oracle_for_smoke": True,
        "sealed_test_oracle": False,
    }:
        raise ConfigurationError("Phase 5 smoke/test authority must fail closed")
    seeds = value.get("search_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
    ):
        raise ConfigurationError("Phase 5 search seeds are invalid")
    return Phase5Experiment(
        registry_source_hash=sha256_json(value),
        experiment_id=str(value["experiment_id"]),
        output_root=_path(value["output_root"], "output_root"),
        transfer_registry_path=_path(value["transfer_registry"], "transfer_registry"),
        search_seeds=tuple(seeds),
        candidate_cap=_integer(matched["candidate_cap"], "candidate_cap", minimum=1),
        oracle_call_cap=_integer(matched["oracle_call_cap"], "oracle_call_cap", minimum=1),
        model_request_cap=_integer(matched["model_request_cap"], "model_request_cap"),
        retrieval_max_items=_integer(retrieval["max_items"], "retrieval.max_items"),
        retrieval_max_bytes=_integer(retrieval["max_bytes"], "retrieval.max_bytes"),
        retrieval_max_tokens=_integer(retrieval["max_tokens"], "retrieval.max_tokens"),
        bootstrap_seed=_integer(analysis["bootstrap_seed"], "analysis.bootstrap_seed"),
        bootstrap_replicates=_integer(
            analysis["bootstrap_replicates"], "analysis.bootstrap_replicates", minimum=1
        ),
        exposure_policy=_path(spending["exposure_policy"], "exposure_policy"),
        cash_policy=_path(spending["cash_policy"], "cash_policy"),
        cash_ledger=_path(spending["cash_ledger"], "cash_ledger"),
        live_input_token_bound_per_request=_integer(
            spending["live_input_token_bound_per_request"],
            "live_input_token_bound_per_request",
            minimum=1,
        ),
        live_requests_per_child=_integer(
            spending["live_requests_per_child"], "live_requests_per_child", minimum=1
        ),
        live_child_count=_integer(spending["live_child_count"], "live_child_count", minimum=1),
        live_runtime_bound_seconds=_integer(
            spending["live_runtime_bound_seconds"],
            "live_runtime_bound_seconds",
            minimum=1,
        ),
    )


def is_phase5_experiment(path: Path) -> bool:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(value, dict) and value.get("experiment_version") == PHASE5_EXPERIMENT_VERSION


def _load_policy(repository_root: Path, path: Path) -> JsonObject:
    try:
        value: object = yaml.safe_load((repository_root / path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("Phase 5 exposure policy is unavailable") from exc
    if not isinstance(value, dict) or value.get("policy_version") != PHASE5_EXPOSURE_POLICY_VERSION:
        raise ConfigurationError("unsupported Phase 5 exposure policy")
    if value.get("status") != "pending-user-review":
        raise ConfigurationError("Phase 5 exposure policy status is not pending review")
    authorization = value.get("authorization")
    if not isinstance(authorization, dict) or any(authorization.values()):
        raise ConfigurationError("Phase 5 exposure policy must deny all live/test authority")
    return cast(JsonObject, value)


def _task_records(manifest: JsonObject, role: SplitLabel) -> list[dict[str, object]]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ConfigurationError("Phase 5 task index is invalid")
    result: list[dict[str, object]] = [
        cast(dict[str, object], item)
        for item in tasks
        if isinstance(item, dict) and item.get("split") == role.value
    ]
    return sorted(result, key=lambda item: str(item["task_id"]))


def _evidence_fact(hidden: object, *, label: str) -> EvidenceFact:
    from world_model_search.evaluation.phase5_transfer import Phase5HiddenTask

    if not isinstance(hidden, Phase5HiddenTask):
        raise TypeError("Phase 5 evidence requires evaluator task data")
    candidate_hash = sha256_json(
        {"candidate": canonical_json(hidden.reference_ast), "label": label}
    )
    evaluation_hash = sha256_json(
        {"semantic_hash": hidden.semantic_hash, "exact": True, "label": label}
    )
    run_hash = sha256_json(
        {"task_id": hidden.task_id, "family_id": hidden.family_id, "label": label}
    )
    return EvidenceFact(
        task_id=hidden.task_id,
        family_id=hidden.family_id,
        role=hidden.role,
        semantic_hash=hidden.semantic_hash,
        run_hash=run_hash,
        candidate_hash=candidate_hash,
        evaluation_hash=evaluation_hash,
        artifact_hash=sha256_json(
            {
                "task_id": hidden.task_id,
                "reference_semantics": list(hidden.oracle_bundle.ordered_semantics),
            }
        ),
    )


def _load_snapshot(path: Path) -> MemorySnapshot:
    try:
        value: object = json.loads(read_text_artifact(path))
    except json.JSONDecodeError as exc:
        raise PersistenceError("memory snapshot is corrupt") from exc
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise PersistenceError("memory snapshot fields are malformed")
    items: list[SafeMemoryItem] = []
    for item in value["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("applicability"), dict):
            raise PersistenceError("memory snapshot item is malformed")
        applicability = item["applicability"]
        items.append(
            SafeMemoryItem(
                str(item["record_id"]),
                MemoryKind(str(item["kind"])),
                str(item["text"]),
                str(item["scope"]),
                MemoryApplicability(
                    str(applicability["world_specification_version"]),
                    str(applicability["dsl_version"]),
                    bool(applicability["requires_exact_feedback"]),
                ),
            )
        )
    snapshot = MemorySnapshot(
        str(value["split_registry_hash"]),
        str(value["database_export_hash"]),
        tuple(items),
    )
    if snapshot.snapshot_hash != value.get("snapshot_hash"):
        raise PersistenceError("memory snapshot identity mismatch")
    return snapshot


def _recode(
    reference: BitExpr, definition: PrimitiveDefinition, registry: PrimitiveRegistry
) -> BitExpr:
    return replace_subtree(reference, definition.ast, definition.primitive_id)


def _empty_retrieval(
    task: object, snapshot: MemorySnapshot, experiment: Phase5Experiment
) -> RetrievalRecord:
    from world_model_search.domain.types import PublicTask

    if not isinstance(task, PublicTask):
        raise TypeError("retrieval task must be PublicTask")
    return retrieve_memory(
        task=task,
        snapshot=snapshot,
        public_search_state={"search_stage": "initial", "evaluations": 0},
        max_items=experiment.retrieval_max_items,
        max_bytes=experiment.retrieval_max_bytes,
        max_tokens=experiment.retrieval_max_tokens,
    )


def _forecast(
    repository_root: Path,
    experiment: Phase5Experiment,
    exposure_policy: JsonObject,
) -> JsonObject:
    price = load_price_policy(repository_root / experiment.cash_policy)
    maximum_output = 2048
    request_nano = price.price.maximum_cost(
        input_token_bound=experiment.live_input_token_bound_per_request,
        max_output_tokens=maximum_output,
    )
    total_requests = experiment.live_requests_per_child * experiment.live_child_count
    published = request_nano * total_requests
    ceilings = exposure_policy.get("published_exposure_ceilings_nano_usd")
    if not isinstance(ceilings, dict):
        raise ConfigurationError("Phase 5 exposure ceilings are malformed")
    one_request_cap = _json_integer(ceilings.get("one_request"), "one-request exposure cap")
    child_cap = _json_integer(ceilings.get("child"), "child exposure cap")
    development_cap = _json_integer(ceilings.get("development_stage"), "development exposure cap")
    phase5_cap = _json_integer(ceilings.get("phase5_total"), "Phase 5 exposure cap")
    if (
        request_nano > one_request_cap
        or request_nano * experiment.live_requests_per_child > child_cap
        or published > development_cap
        or published > phase5_cap
    ):
        raise ConfigurationError("Phase 5 forecast exceeds a published exposure partition")
    cash_status: JsonObject
    from world_model_search.model.ledger import ProjectLedger

    with ProjectLedger(repository_root / experiment.cash_ledger, price) as ledger:
        status = ledger.status()
        cash_status = cast(JsonObject, status["cash_budget"])
    cash_upper = _json_integer(cash_status.get("cash_upper_bound_nano_usd"), "cash upper bound")
    personal_cap = _json_integer(
        cash_status.get("personal_lifetime_cap_nano_usd"), "personal cash cap"
    )
    return {
        "forecast_version": "phase5-live-development-worst-case-forecast-v1",
        "input_token_bound_per_request": experiment.live_input_token_bound_per_request,
        "max_output_tokens_per_request": maximum_output,
        "requests_per_child": experiment.live_requests_per_child,
        "child_count": experiment.live_child_count,
        "total_requests": total_requests,
        "runtime_bound_seconds": experiment.live_runtime_bound_seconds,
        "published_rate_nano_usd_per_request": request_nano,
        "published_rate_nano_usd_per_child": request_nano * experiment.live_requests_per_child,
        "published_rate_nano_usd_total": published,
        "actual_cash_worst_case_upper_bound_nano_usd": cash_upper + published,
        "cash_ceiling_nano_usd": personal_cap,
        "fits_current_cash_headroom": cash_upper + published <= personal_cap,
        "authorization": "denied-pending-user-review-and-live-run-approval",
    }


def phase5_dry_run(*, repository_root: Path, experiment: Phase5Experiment) -> JsonObject:
    transfer = load_transfer_registry(repository_root / experiment.transfer_registry_path)
    benchmark = generate_transfer_benchmark(repository_root, transfer)
    exposure = _load_policy(repository_root, experiment.exposure_policy)
    forecast = _forecast(repository_root, experiment, exposure)
    return {
        "dry_run_version": "phase5-no-cost-dry-run-v1",
        "experiment_hash": experiment.content_hash,
        "transfer_registry_hash": transfer.content_hash,
        "transfer_manifest_hash": sha256_json(benchmark.manifest),
        "family_count": len(transfer.families),
        "task_count": len(cast(list[object], benchmark.manifest["tasks"])),
        "semantic_disjointness": benchmark.manifest["semantic_disjointness_proof"],
        "sealed_test_authorized": False,
        "model_request_cap": experiment.model_request_cap,
        "forecast": forecast,
    }


def _write(path: Path, value: object) -> str:
    return write_content_artifact(path, canonical_json(value))


def _artifact_hash(path: Path) -> str:
    return sha256_text(read_text_artifact(path))


def _phase5_code_hash(repository_root: Path) -> str:
    source_root = repository_root / "src/world_model_search"
    records = []
    for path in sorted(source_root.rglob("*.py")):
        records.append(
            {
                "path": str(path.relative_to(repository_root)),
                "content_hash": sha256_text(path.read_text(encoding="utf-8")),
            }
        )
    if not records:
        # Installed-package tests use a temporary experiment repository without source files.
        package_root = Path(__file__).parents[1]
        module_files = (
            Path(__file__),
            package_root / "memory/store.py",
            package_root / "dsl/primitives.py",
        )
        records = [
            {"path": path.name, "content_hash": sha256_text(path.read_text(encoding="utf-8"))}
            for path in module_files
        ]
    return sha256_json({"code_hash_version": "phase5-source-tree-v1", "files": records})


def assert_matched_condition_isolation(off: JsonObject, on: JsonObject) -> None:
    allowed = {
        "condition",
        "memory_snapshot_hash",
        "retrieval_record_hashes",
        "rendered_memory_hashes",
        "primitive_registry_hash",
    }
    if set(off) != set(on):
        raise ValueError("matched-isolation gate: condition manifest fields differ")
    if any(off[key] != on[key] for key in off.keys() - allowed):
        raise ValueError("matched-isolation gate: nonmemory condition manifest fields differ")


def run_phase5_smoke(
    *,
    repository_root: Path,
    registry_path: Path,
    allow_live_model: bool = False,
    interrupt_after_memory: bool = False,
) -> JsonObject:
    if allow_live_model:
        raise ConfigurationError("Phase 5 implementation turn forbids live model calls")
    experiment = load_phase5_experiment(registry_path)
    root = repository_root / experiment.output_root
    summary_path = root / "summary.json"
    if summary_path.exists():
        existing_summary = json.loads(read_text_artifact(summary_path))
        if (
            not isinstance(existing_summary, dict)
            or existing_summary.get("experiment_hash") != experiment.content_hash
        ):
            raise PersistenceError("completed Phase 5 smoke binds another registry")
        return cast(JsonObject, existing_summary)
    transfer = load_transfer_registry(repository_root / experiment.transfer_registry_path)
    benchmark = generate_transfer_benchmark(repository_root, transfer)
    exposure = _load_policy(repository_root, experiment.exposure_policy)
    forecast = _forecast(repository_root, experiment, exposure)
    plan_hash = sha256_json(
        {
            "experiment_hash": experiment.content_hash,
            "transfer_registry_hash": transfer.content_hash,
            "primary": "net-held-out-two-part-code-length-gain",
            "definition_cost_charged_once": True,
        }
    )
    code_config_hash = sha256_json(
        {
            "phase5_code_hash": _phase5_code_hash(repository_root),
            "experiment_hash": experiment.content_hash,
            "transfer_registry_hash": transfer.content_hash,
            "exposure_policy_hash": sha256_json(exposure),
        }
    )
    store_authority = Phase5TaskStore(benchmark.root)
    training_hidden = [
        store_authority.load(
            str(item["task_id"]),
            allowed_splits=frozenset({SplitLabel.TRAINING}),
            purpose="phase5-no-cost-training-evidence",
        )
        for item in _task_records(benchmark.manifest, SplitLabel.TRAINING)
    ]
    development_hidden = [
        store_authority.load(
            str(item["task_id"]),
            allowed_splits=frozenset({SplitLabel.DEVELOPMENT}),
            purpose="phase5-no-cost-development-promotion-and-smoke",
        )
        for item in _task_records(benchmark.manifest, SplitLabel.DEVELOPMENT)
    ]
    facts = {
        fact.evidence_id: fact
        for hidden in (*training_hidden, *development_hidden)
        for fact in (_evidence_fact(hidden, label="phase5-smoke-reference-v1"),)
    }
    memory_path = root / "memory" / "phase5-memory.sqlite3"
    definition = PrimitiveDefinition(selector_core())
    with Phase5MemoryStore(
        memory_path,
        split_registry_hash=transfer.content_hash,
        evidence_catalog=facts,
    ) as memory:
        for hidden in (*training_hidden, *development_hidden):
            memory.admit_evidence(
                facts[_evidence_fact(hidden, label="phase5-smoke-reference-v1").evidence_id]
            )
        source_ids = tuple(
            sorted(
                _evidence_fact(hidden, label="phase5-smoke-reference-v1").evidence_id
                for hidden in training_hidden
            )
        )
        record_id = memory.propose_record(
            kind=MemoryKind.PRIMITIVE_PROPOSAL,
            proposer_text=(
                "A typed zero-arity selector subtree recurred across independent training tasks; "
                "try its exact registered expansion when composing compact F0 programs."
            ),
            scope="global-f0",
            applicability=MemoryApplicability(
                "elementary-public-world-v1", "binary-ca-radius1-dsl-v1"
            ),
            support_evidence_ids=source_ids,
            provenance_hashes=tuple(sorted(facts[item].artifact_hash for item in source_ids)),
            definition_cost_bits=definition.base_definition_bits,
        )
        proposed_registry = PrimitiveRegistry(
            transfer.content_hash, plan_hash, source_ids, (definition,)
        )
        base_total = 0
        memory_total = 0
        correctness = True
        development_rows: list[JsonObject] = []
        for hidden in development_hidden:
            learned = _recode(hidden.reference_ast, definition, proposed_registry)
            expanded = expand_primitives(learned, proposed_registry)
            base_bits = encoded_length(hidden.reference_ast)
            memory_bits = len(encode_program(learned, proposed_registry))
            exact = expanded == hidden.reference_ast
            correctness = correctness and exact
            base_total += base_bits
            memory_total += memory_bits
            fact = _evidence_fact(hidden, label="phase5-smoke-reference-v1")
            memory.link_evidence(record_id, fact.evidence_id, EvidencePolarity.VALIDATION)
            development_rows.append(
                {
                    "task_id": hidden.task_id,
                    "target_family": hidden.family_id,
                    "base_bits": base_bits,
                    "memory_program_bits": memory_bits,
                    "gross_savings_bits": base_bits - memory_bits,
                    "correct": exact,
                }
            )
        definition_cost = library_definition_cost(proposed_registry.definitions)
        gross_gain = base_total - memory_total
        net_gain = gross_gain - definition_cost
        gate: JsonObject = {
            "base_total_bits": base_total,
            "memory_program_total_bits": memory_total,
            "gross_gain_bits": gross_gain,
            "definition_cost_bits": definition_cost,
            "net_gain_bits": net_gain,
            "strictly_positive_net_gain": net_gain > 0,
            "all_correct": correctness,
            "definition_charged_exactly_once": True,
        }
        if correctness and net_gain > 0:
            memory.transition(
                record_id,
                ValidationState.PROMOTED,
                reason="predeclared development transfer gate passed",
                gate_payload=gate,
            )
            registry = proposed_registry
            promotion_status = "promoted-development-evidence"
        else:
            memory.transition(
                record_id,
                ValidationState.REJECTED,
                reason="predeclared development transfer gate failed",
                gate_payload=gate,
            )
            registry = empty_primitive_registry(transfer.content_hash, plan_hash)
            promotion_status = "rejected-no-promotion"
        export = memory.deterministic_export()
        _write(root / "memory" / "export.json", export)
        snapshot = memory.freeze_snapshot(root / "memory" / "snapshot.json")
    write_primitive_registry(root / "primitives" / "registry.json", registry)
    if interrupt_after_memory:
        return {
            "summary_version": "phase5-no-cost-smoke-summary-v1",
            "experiment_id": experiment.experiment_id,
            "experiment_hash": experiment.content_hash,
            "status": "interrupted-after-frozen-memory",
            "provider_calls": 0,
            "sealed_test_accesses": 0,
            "memory_snapshot_hash": snapshot.snapshot_hash,
            "primitive_registry_hash": registry.registry_hash,
        }

    empty_snapshot = MemorySnapshot(
        transfer.content_hash,
        sha256_json(
            {
                "empty_memory_version": "phase5-condition-c-empty-memory-v1",
                "experiment_hash": experiment.content_hash,
            }
        ),
        (),
    )
    empty_registry = empty_primitive_registry(transfer.content_hash, plan_hash)
    rows: list[JsonObject] = []
    retrieval_records: list[JsonObject] = []
    request_records: list[JsonObject] = []
    run_started = perf_counter_ns()
    for hidden in development_hidden:
        task = load_transfer_public_task(benchmark.root, hidden.task_id).public_view()
        for seed in experiment.search_seeds:
            retrieval_c = _empty_retrieval(task, empty_snapshot, experiment)
            retrieval_d = _empty_retrieval(task, snapshot, experiment)
            prompt_c = render_phase5_prompt(
                task=task,
                role=ProposalRole.TRANSFER,
                requested_batch_size=1,
                retrieval=retrieval_c,
                primitives=empty_registry,
            )
            prompt_d = render_phase5_prompt(
                task=task,
                role=ProposalRole.TRANSFER,
                requested_batch_size=1,
                retrieval=retrieval_d,
                primitives=registry,
            )
            assert_matched_prompt_isolation(prompt_c, prompt_d)
            for condition, retrieval, condition_registry, prompt in (
                (CONDITION_C, retrieval_c, empty_registry, prompt_c),
                (CONDITION_D, retrieval_d, registry, prompt_d),
            ):
                schema = primitive_candidate_batch_json_schema(
                    role=ProposalRole.TRANSFER,
                    batch_size=1,
                    registry=condition_registry,
                )
                retrieval_hash = sha256_json(retrieval.to_value())
                bindings = Phase5RequestBindings(
                    transfer.content_hash,
                    snapshot.database_export_hash
                    if condition == CONDITION_D
                    else empty_snapshot.database_export_hash,
                    retrieval.snapshot_hash,
                    retrieval_hash,
                    condition_registry.registry_hash,
                    sha256_json(schema),
                    sha256_json(exposure),
                    code_config_hash,
                )
                request = Phase5ModelRequest(
                    "offline-no-model-v1",
                    "none",
                    "gpt-5-mini-2025-08-07",
                    "v1/responses",
                    "default",
                    "low",
                    2048,
                    1,
                    ProposalRole.TRANSFER,
                    prompt,
                    schema,
                    bindings,
                )
                request_records.append(
                    {
                        "task_id": hidden.task_id,
                        "seed": seed,
                        "condition": condition,
                        "request_hash": request.request_hash,
                        "request_identity": request.identity_value(),
                        "provider_calls": 0,
                        "cache_hits": 0,
                        "retries": 0,
                        "response_hash": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                )
                retrieval_records.append(
                    {
                        "task_id": hidden.task_id,
                        "seed": seed,
                        "condition": condition,
                        "retrieval_hash": retrieval_hash,
                        "retrieval": retrieval.to_value(),
                    }
                )
                started = perf_counter_ns()
                if condition == CONDITION_D and condition_registry.definitions:
                    learned_program = _recode(hidden.reference_ast, definition, condition_registry)
                    expanded = expand_primitives(learned_program, condition_registry)
                    program_bits = len(encode_program(learned_program, condition_registry))
                    decode_ok = (
                        expand_primitives(
                            decode_program(
                                encode_program(learned_program, condition_registry),
                                condition_registry,
                            ),
                            condition_registry,
                        )
                        == expanded
                    )
                else:
                    expanded = hidden.reference_ast
                    program_bits = encoded_length(hidden.reference_ast)
                    decode_ok = True
                evaluation = ExactDslOracle(
                    hidden.oracle_bundle,
                    limits=AstLimits(),
                    response_mode=OracleResponseMode.SCORE_ONLY,
                ).evaluate(expanded)
                runtime_ns = max(0, perf_counter_ns() - started)
                rows.append(
                    {
                        "task_id": hidden.task_id,
                        "target_family": hidden.family_id,
                        "seed": seed,
                        "condition": condition,
                        "correct": evaluation.result.exact and decode_ok,
                        "program_bits": program_bits,
                        "base_program_bits": encoded_length(hidden.reference_ast),
                        "candidate_evaluations": experiment.candidate_cap,
                        "oracle_calls": experiment.oracle_call_cap,
                        "model_requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "published_rate_nano_usd": 0,
                        "actual_cash_nano_usd": 0,
                        "runtime_ns": runtime_ns,
                        "retrieval_hit": record_id in retrieval.selected_record_ids,
                        "request_hash": request.request_hash,
                        "retrieval_hash": retrieval_hash,
                        "memory_snapshot_hash": retrieval.snapshot_hash,
                        "primitive_registry_hash": condition_registry.registry_hash,
                    }
                )
    runtime_ns = max(0, perf_counter_ns() - run_started)
    family_matrix: list[JsonObject] = []
    for target_family in sorted({str(row["target_family"]) for row in rows}):
        target_rows = [row for row in rows if row["target_family"] == target_family]
        base_bits = sum(
            _json_integer(row["program_bits"], "row program_bits")
            for row in target_rows
            if row["condition"] == CONDITION_C
        )
        learned_bits = sum(
            _json_integer(row["program_bits"], "row program_bits")
            for row in target_rows
            if row["condition"] == CONDITION_D
        )
        family_matrix.append(
            cast(
                JsonObject,
                {
                    "source_families": sorted({hidden.family_id for hidden in training_hidden}),
                    "target_family": target_family,
                    "gross_program_length_savings_bits": base_bits - learned_bits,
                    "definition_cost_allocated_to_cell_bits": 0,
                    "negative_transfer": learned_bits > base_bits,
                },
            )
        )
    paired_base = sum(
        _json_integer(row["program_bits"], "row program_bits")
        for row in rows
        if row["condition"] == CONDITION_C
    )
    paired_memory = sum(
        _json_integer(row["program_bits"], "row program_bits")
        for row in rows
        if row["condition"] == CONDITION_D
    )
    paired_gross = paired_base - paired_memory
    paired_net = paired_gross - library_definition_cost(registry.definitions)
    prompt_bytes: dict[str, int] = {CONDITION_C: 0, CONDITION_D: 0}
    for record in request_records:
        condition = str(record["condition"])
        identity = cast(dict[str, object], record["request_identity"])
        prompt_section = cast(dict[str, object], identity["prompt"])
        prompt_bytes[condition] += len(str(prompt_section["rendered_input_utf8"]).encode("utf-8"))
    analysis: JsonObject = {
        "analysis_version": PHASE5_ANALYSIS_VERSION,
        "data_role": "development-smoke",
        "confirmatory": False,
        "paired_row_count": len(rows) // 2,
        "all_correct": all(bool(row["correct"]) for row in rows),
        "base_total_bits": paired_base,
        "memory_program_total_bits": paired_memory,
        "gross_program_length_savings_bits": paired_gross,
        "library_definition_cost_bits_charged_once": library_definition_cost(registry.definitions),
        "net_held_out_two_part_gain_bits": paired_net,
        "memory_on_minus_off_exact_rate": 0.0,
        "retrieval_precision": (
            sum(bool(row["retrieval_hit"]) for row in rows if row["condition"] == CONDITION_D)
            / max(1, sum(row["condition"] == CONDITION_D for row in rows))
        ),
        "model_tokens": 0,
        "memory_off_prompt_utf8_bytes": prompt_bytes[CONDITION_C],
        "memory_on_prompt_utf8_bytes": prompt_bytes[CONDITION_D],
        "genuine_memory_prompt_overhead_bytes": (
            prompt_bytes[CONDITION_D] - prompt_bytes[CONDITION_C]
        ),
        "published_rate_nano_usd": 0,
        "actual_cash_nano_usd": 0,
        "exact_per_candidate_evaluation": {CONDITION_C: 1.0, CONDITION_D: 1.0},
        "performance_per_published_dollar": None,
        "runtime_ns": runtime_ns,
        "multiplicity": "holm-predeclared-secondary-v1-not-applicable-to-deterministic-smoke",
        "uncertainty": "family-stratified-interval-not-estimated-for-mechanism-smoke",
        "promotion_status": promotion_status,
    }
    shared_condition_manifest: JsonObject = {
        "manifest_version": "phase5-condition-manifest-v1",
        "experiment_hash": experiment.content_hash,
        "transfer_registry_hash": transfer.content_hash,
        "task_ids": [hidden.task_id for hidden in development_hidden],
        "task_order": [hidden.task_id for hidden in development_hidden],
        "search_seeds": list(experiment.search_seeds),
        "base_search_algorithm": "deterministic-typed-recoding-smoke-v1",
        "scheduler": "uniform-sorted-task-v1",
        "candidate_cap": experiment.candidate_cap,
        "oracle_call_cap": experiment.oracle_call_cap,
        "model_request_cap": experiment.model_request_cap,
        "stopping_rule": "one-exact-recoding-candidate-v1",
        "prompt_template": "phase5-public-task-with-explicit-memory-v1",
        "model": "gpt-5-mini-2025-08-07",
        "endpoint": "v1/responses",
        "service_tier": "default",
        "reasoning_effort": "low",
        "max_output_tokens": 2048,
        "budget_policy_hash": sha256_json(exposure),
        "code_config_hash": code_config_hash,
    }
    off_retrievals = [item for item in retrieval_records if item["condition"] == CONDITION_C]
    on_retrievals = [item for item in retrieval_records if item["condition"] == CONDITION_D]
    condition_off = cast(
        JsonObject,
        {
            **shared_condition_manifest,
            "condition": CONDITION_C,
            "memory_snapshot_hash": empty_snapshot.snapshot_hash,
            "retrieval_record_hashes": [item["retrieval_hash"] for item in off_retrievals],
            "rendered_memory_hashes": [
                sha256_text(str(cast(dict[str, object], item["retrieval"])["rendered_memory_utf8"]))
                for item in off_retrievals
            ],
            "primitive_registry_hash": empty_registry.registry_hash,
        },
    )
    condition_on = cast(
        JsonObject,
        {
            **shared_condition_manifest,
            "condition": CONDITION_D,
            "memory_snapshot_hash": snapshot.snapshot_hash,
            "retrieval_record_hashes": [item["retrieval_hash"] for item in on_retrievals],
            "rendered_memory_hashes": [
                sha256_text(str(cast(dict[str, object], item["retrieval"])["rendered_memory_utf8"]))
                for item in on_retrievals
            ],
            "primitive_registry_hash": registry.registry_hash,
        },
    )
    assert_matched_condition_isolation(condition_off, condition_on)
    manifest: JsonObject = {
        "manifest_version": "phase5-smoke-manifest-v1",
        "experiment_id": experiment.experiment_id,
        "experiment_hash": experiment.content_hash,
        "transfer_registry_hash": transfer.content_hash,
        "transfer_manifest_hash": sha256_json(benchmark.manifest),
        "memory_database_schema": 1,
        "memory_database_export_hash": snapshot.database_export_hash,
        "memory_snapshot_hash": snapshot.snapshot_hash,
        "primitive_language": "phase5-zero-arity-typed-primitives-v1",
        "primitive_registry_hash": registry.registry_hash,
        "primitive_definition_code": encode_library(registry.definitions),
        "analysis_plan_hash": plan_hash,
        "prompt_version": "phase5-single-template-memory-block-v1",
        "model_settings": {
            "model": "gpt-5-mini-2025-08-07",
            "endpoint": "v1/responses",
            "service_tier": "default",
            "reasoning_effort": "low",
            "max_output_tokens": 2048,
            "provider_calls": 0,
        },
        "caps": {
            "candidate": experiment.candidate_cap,
            "oracle_call": experiment.oracle_call_cap,
            "model_request": experiment.model_request_cap,
        },
        "budget_policy_hash": sha256_json(exposure),
        "cash_policy_hash": load_price_policy(
            repository_root / experiment.cash_policy
        ).content_hash,
        "code_config_hash": code_config_hash,
        "sealed_test_accesses": 0,
        "development_oracle_accesses": len(development_hidden),
        "training_oracle_accesses": len(training_hidden),
    }
    _write(root / "manifest.json", manifest)
    _write(root / "raw-paired-rows.json", {"rows": rows})
    _write(root / "retrieval-records.json", {"records": retrieval_records})
    _write(root / "request-records.json", {"records": request_records})
    _write(root / "primitive-development-gate.json", {"gate": gate, "rows": development_rows})
    _write(
        root / "transfer-matrix.json",
        {
            "matrix_version": "phase5-source-target-transfer-matrix-v1",
            "cells": family_matrix,
            "library_definition_cost_bits_separate": library_definition_cost(registry.definitions),
            "aggregate_gross_savings_bits": paired_gross,
            "aggregate_net_gain_bits_after_one_definition_charge": paired_net,
        },
    )
    _write(root / "analysis.json", analysis)
    _write(root / "forecast.json", forecast)
    _write(root / "condition-c-manifest.json", condition_off)
    _write(root / "condition-d-manifest.json", condition_on)
    artifact_hashes = {
        name: _artifact_hash(root / name)
        for name in (
            "manifest.json",
            "raw-paired-rows.json",
            "retrieval-records.json",
            "request-records.json",
            "primitive-development-gate.json",
            "transfer-matrix.json",
            "analysis.json",
            "forecast.json",
            "condition-c-manifest.json",
            "condition-d-manifest.json",
            "memory/export.json",
            "memory/snapshot.json",
            "primitives/registry.json",
        )
    }
    report = cast(
        JsonObject,
        {
            "report_version": PHASE5_REPORT_VERSION,
            "scientific_scope": "structured-transfer-within-f0-only",
            "data_role": "development-smoke-not-confirmatory",
            "result": analysis,
            "transfer_matrix": family_matrix,
            "primitive_gate": gate,
            "scoped_rejected_invalidated": []
            if promotion_status.startswith("promoted")
            else [record_id],
            "negative_transfer_cells": [
                cell for cell in family_matrix if bool(cell["negative_transfer"])
            ],
            "retrieval_misses": sum(
                not bool(row["retrieval_hit"]) for row in rows if row["condition"] == CONDITION_D
            ),
            "forecast": forecast,
            "artifact_hashes": artifact_hashes,
        },
    )
    _write(root / "report.json", report)
    artifact_hashes["report.json"] = _artifact_hash(root / "report.json")
    summary = cast(
        JsonObject,
        {
            "summary_version": "phase5-no-cost-smoke-summary-v1",
            "experiment_id": experiment.experiment_id,
            "experiment_hash": experiment.content_hash,
            "status": "completed-no-cost-development-smoke",
            "scientific_status": (
                "primitive-gate-development-evidence-paired-smoke-only-h3-unconfirmed"
            ),
            "promotion_status": promotion_status,
            "net_gain_bits": paired_net,
            "provider_calls": 0,
            "sealed_test_accesses": 0,
            "artifact_hashes": artifact_hashes,
        },
    )
    _write(summary_path, summary)
    return summary


def replay_phase5_smoke(*, repository_root: Path, registry_path: Path) -> JsonObject:
    """Verify all provider-independent records and reproduce analysis/report hashes."""

    experiment = load_phase5_experiment(registry_path)
    root = repository_root / experiment.output_root
    summary_raw = json.loads(read_text_artifact(root / "summary.json"))
    if (
        not isinstance(summary_raw, dict)
        or summary_raw.get("experiment_hash") != experiment.content_hash
    ):
        raise ReplayError("Phase 5 summary identity mismatch")
    summary = cast(JsonObject, summary_raw)
    artifact_hashes = summary.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ReplayError("Phase 5 summary has no artifact index")
    for name, expected in artifact_hashes.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ReplayError("Phase 5 artifact index is malformed")
        if _artifact_hash(root / name) != expected:
            raise ReplayError(f"Phase 5 artifact hash mismatch: {name}")
    transfer = load_transfer_registry(repository_root / experiment.transfer_registry_path)
    benchmark = generate_transfer_benchmark(repository_root, transfer)
    manifest = json.loads(read_text_artifact(root / "manifest.json"))
    if not isinstance(manifest, dict) or manifest.get("transfer_manifest_hash") != sha256_json(
        benchmark.manifest
    ):
        raise ReplayError("Phase 5 transfer benchmark identity mismatch")
    snapshot = _load_snapshot(root / "memory" / "snapshot.json")
    if snapshot.snapshot_hash != manifest.get("memory_snapshot_hash"):
        raise ReplayError("Phase 5 replay memory snapshot mismatch")
    from world_model_search.dsl.primitives import load_primitive_registry

    registry = load_primitive_registry(root / "primitives" / "registry.json")
    if registry.registry_hash != manifest.get("primitive_registry_hash"):
        raise ReplayError("Phase 5 replay primitive registry mismatch")
    if decode_library(str(manifest["primitive_definition_code"])) != registry.definitions:
        raise ReplayError("Phase 5 replay primitive code does not decode")
    requests_raw = json.loads(read_text_artifact(root / "request-records.json"))
    retrievals_raw = json.loads(read_text_artifact(root / "retrieval-records.json"))
    rows_raw = json.loads(read_text_artifact(root / "raw-paired-rows.json"))
    if not all(isinstance(value, dict) for value in (requests_raw, retrievals_raw, rows_raw)):
        raise ReplayError("Phase 5 replay row artifacts are malformed")
    requests = requests_raw.get("records")
    retrievals = retrievals_raw.get("records")
    rows = rows_raw.get("rows")
    if not all(isinstance(value, list) for value in (requests, retrievals, rows)):
        raise ReplayError("Phase 5 replay records are unavailable")
    retrieval_index = {
        str(record["retrieval_hash"]): record for record in retrievals if isinstance(record, dict)
    }
    for record in requests:
        if not isinstance(record, dict) or record.get("provider_calls") != 0:
            raise ReplayError("Phase 5 replay encountered provider activity")
        identity = record.get("request_identity")
        if not isinstance(identity, dict) or sha256_json(identity) != record.get("request_hash"):
            raise ReplayError("Phase 5 request identity mismatch")
        bindings = identity.get("bindings")
        if not isinstance(bindings, dict):
            raise ReplayError("Phase 5 request bindings are missing")
        retrieval_hash = str(bindings.get("retrieval_record_hash"))
        retrieval_record = retrieval_index.get(retrieval_hash)
        if retrieval_record is None:
            raise ReplayError("Phase 5 request retrieval binding is missing")
        retrieval_value = retrieval_record.get("retrieval")
        if not isinstance(retrieval_value, dict) or sha256_json(retrieval_value) != retrieval_hash:
            raise ReplayError("Phase 5 retrieval record identity mismatch")
    all_correct = all(isinstance(row, dict) and row.get("correct") is True for row in rows)
    if not all_correct:
        raise ReplayError("Phase 5 recorded correctness does not replay")
    base = sum(
        int(row["program_bits"])
        for row in rows
        if isinstance(row, dict) and row.get("condition") == CONDITION_C
    )
    memory = sum(
        int(row["program_bits"])
        for row in rows
        if isinstance(row, dict) and row.get("condition") == CONDITION_D
    )
    net = base - memory - library_definition_cost(registry.definitions)
    analysis = json.loads(read_text_artifact(root / "analysis.json"))
    if not isinstance(analysis, dict) or analysis.get("net_held_out_two_part_gain_bits") != net:
        raise ReplayError("Phase 5 analysis reproduction diverged")
    return {
        "replay_version": PHASE5_REPLAY_VERSION,
        "experiment_id": experiment.experiment_id,
        "status": "verified-provider-disabled",
        "provider_calls": 0,
        "sealed_test_accesses": 0,
        "row_count": len(rows),
        "net_gain_bits": net,
        "summary_hash": _artifact_hash(root / "summary.json"),
        "report_hash": _artifact_hash(root / "report.json"),
    }
