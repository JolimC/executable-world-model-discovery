"""Crash-safe Phase 4 A/B/C language-model search lifecycle."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from world_model_search.config import AppConfig
from world_model_search.domain.types import (
    Candidate,
    CandidateSummary,
    OracleFeedback,
    OracleResponseMode,
    OracleResult,
    ProposalRole,
    SearchEvent,
    SplitLabel,
    Task,
)
from world_model_search.dsl.ast import AstLimits, BitExpr
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.interpreter import semantic_hash
from world_model_search.dsl.json_schema import (
    DslCandidateDocument,
    ast_canonical_json,
)
from world_model_search.errors import BudgetExhaustedError, ConfigurationError, PersistenceError
from world_model_search.model.backends import (
    LiveOptIn,
    OfflineResumeBackend,
    OpenAIResponsesBackend,
    ScriptedBackend,
)
from world_model_search.model.cache import ExactResponseCache
from world_model_search.model.ledger import ProjectLedger
from world_model_search.model.policy import PricePolicy, load_price_policy
from world_model_search.model.prompts import ParentScoreFeedback
from world_model_search.model.schema import BatchEnvelopeError
from world_model_search.model.types import (
    ModelBackend,
    ModelDispatchError,
    ModelError,
    ModelErrorCategory,
    ModelRequest,
    ModelResponse,
)
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.persistence.artifacts import (
    read_text_artifact,
    write_content_artifact,
    write_json_exclusive,
)
from world_model_search.persistence.manifest import build_manifest, utc_now
from world_model_search.persistence.phase4_database import Phase4Database
from world_model_search.phase4_versions import (
    PHASE4_EVENT_SCHEMA_VERSION,
    PHASE4_MANIFEST_SCHEMA_VERSION,
    PHASE4_REPORT_VERSION,
)
from world_model_search.proposer.llm import LLMParsedResponse, LLMProposer
from world_model_search.scheduler.uniform import UniformScheduler
from world_model_search.search.archive import (
    ArchiveDecision,
    MapElitesArchive,
    SingleIncumbent,
)
from world_model_search.search.phase3 import initialization_candidates
from world_model_search.search.phase4_types import (
    Phase4BudgetState,
    Phase4Condition,
    RequestState,
    phase4_candidate,
)
from world_model_search.serialization import (
    JsonObject,
    JsonValue,
    canonical_json,
    parse_json_object,
    sha256_json,
    sha256_text,
)
from world_model_search.tasks import (
    HiddenTaskBundle,
    HiddenTaskStore,
    OracleTaskAccess,
    benchmark_root_for_config,
    load_public_task,
)


@dataclass(frozen=True, slots=True)
class Phase4Outcome:
    run_id: str
    status: str
    completed_steps: int
    event_payload_hashes: tuple[str, ...]
    run_directory: Path


@dataclass(frozen=True, slots=True)
class Phase4Authority:
    mode: str
    allowed_splits: frozenset[SplitLabel]
    frozen_task_ids: tuple[str, ...] = ()
    freeze_hash: str | None = None

    @classmethod
    def ordinary(cls) -> Phase4Authority:
        return cls(
            "phase4-training-development-v1",
            frozenset({SplitLabel.TRAINING, SplitLabel.DEVELOPMENT}),
        )

    @classmethod
    def locked_test(cls, *, frozen_task_ids: tuple[str, ...], freeze_hash: str) -> Phase4Authority:
        if not frozen_task_ids or len(frozen_task_ids) != len(set(frozen_task_ids)):
            raise ConfigurationError("locked test authority needs unique frozen task IDs")
        if len(freeze_hash) != 64:
            raise ConfigurationError("locked test authority needs a complete freeze hash")
        return cls(
            "phase4-locked-test-once-v1",
            frozenset({SplitLabel.TEST}),
            frozen_task_ids,
            freeze_hash,
        )

    def to_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "mode": self.mode,
                "allowed_splits": sorted(split.value for split in self.allowed_splits),
                "frozen_task_ids": list(self.frozen_task_ids),
                "freeze_hash": self.freeze_hash,
            },
        )


@dataclass(frozen=True, slots=True)
class PreparedPhase4:
    task: Task
    hidden: HiddenTaskBundle
    store: HiddenTaskStore
    limits: AstLimits
    authority: Phase4Authority
    price_policy: PricePolicy


Mechanism = MapElitesArchive | SingleIncumbent


def _backend(config: AppConfig, *, allow_live_model: bool) -> ModelBackend:
    if config.model is None:
        raise ConfigurationError("Phase 4 model settings are missing")
    if config.model.backend_id == "scripted-deterministic-v1":
        return ScriptedBackend()
    if config.model.backend_id == "openai-responses-sdk-v1":
        try:
            return OpenAIResponsesBackend(opt_in=LiveOptIn.resolve(allow_live_model))
        except ModelDispatchError:
            raise ConfigurationError(
                "live model requires --allow-live-model, WMS_ALLOW_LIVE_MODEL=1, and OPENAI_API_KEY"
            ) from None
    raise ConfigurationError("unknown Phase 4 model backend")


def prepare_phase4(
    *, repository_root: Path, config: AppConfig, authority: Phase4Authority, purpose: str
) -> PreparedPhase4:
    if (
        config.schema_version != 4
        or config.dsl is None
        or config.phase4_budget is None
        or config.phase4_policy is None
    ):
        raise ConfigurationError("Phase 4 requires a complete schema-4 configuration")
    benchmark_root = benchmark_root_for_config(repository_root, config)
    task = load_public_task(benchmark_root, config.run.task_id)
    if task.split != config.run.split or task.split not in authority.allowed_splits:
        raise ConfigurationError("Phase 4 task split is not authorized")
    if authority.mode == "phase4-locked-test-once-v1" and task.task_id not in set(
        authority.frozen_task_ids
    ):
        raise ConfigurationError("test task is absent from the frozen Phase 4 authority")
    if config.phase4_budget.oracle_call_cap < len(initialization_candidates()):
        raise ConfigurationError("Phase 4 must charge all seven shared initial candidates")
    store = HiddenTaskStore(benchmark_root)
    hidden = store.load(task.task_id, allowed_splits=authority.allowed_splits, purpose=purpose)
    return PreparedPhase4(
        task=task,
        hidden=hidden,
        store=store,
        limits=AstLimits(config.dsl.max_depth, config.dsl.max_nodes, config.dsl.max_cases),
        authority=authority,
        price_policy=load_price_policy(repository_root / config.phase4_policy.price_policy),
    )


def _mechanism(config: AppConfig, task: Task) -> Mechanism:
    condition = Phase4Condition(str(config.run.condition_id))
    if condition is Phase4Condition.DIVERSE:
        if config.archive is None:
            raise ConfigurationError("diverse Phase 4 condition has no archive settings")
        return MapElitesArchive(task.public_view(), reserve_size=config.archive.reserve_size)
    return SingleIncumbent(task.public_view())


def _result_from_json(data: str) -> OracleResult:
    raw = parse_json_object(data)
    response = raw.get("response")
    if not isinstance(response, dict):
        raise PersistenceError("recorded Phase 4 oracle feedback is malformed")
    summary = response.get("summary")
    if not isinstance(summary, list) or not all(isinstance(item, str) for item in summary):
        raise PersistenceError("recorded Phase 4 feedback summary is malformed")

    def integer(name: str) -> int:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise PersistenceError(f"recorded Phase 4 {name} is malformed")
        return value

    return OracleResult(
        type_valid=bool(raw["type_valid"]),
        total=bool(raw["total"]),
        local_errors=integer("local_errors"),
        local_cases=integer("local_cases"),
        rollout_pass=bool(raw["rollout_pass"]),
        exact=bool(raw["exact"]),
        ast_bits=integer("ast_bits"),
        residual_bits=integer("residual_bits"),
        runtime_ns=integer("runtime_ns"),
        response=OracleFeedback(
            OracleResponseMode(str(response["mode"])),
            cast(tuple[str, ...], tuple(summary)),
            str(response["counterexample"]) if response.get("counterexample") is not None else None,
        ),
    )


def _budget_counter(counters: object, name: str) -> int:
    value = counters.get(name, 0) if isinstance(counters, dict) else 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersistenceError("Phase 4 event budget counters are malformed")
    return value


def _mapping_int(mapping: dict[str, object], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersistenceError(f"persisted request field {name!r} must be an integer")
    return value


def _request_from_artifact(value: object, *, expected_hash: object) -> ModelRequest:
    if (
        not isinstance(value, dict)
        or set(value) != {"artifact_version", "request_hash", "identity"}
        or value.get("artifact_version") != "phase4-model-request-v1"
        or not isinstance(value.get("request_hash"), str)
    ):
        raise PersistenceError("recorded model request artifact is malformed")
    try:
        request = ModelRequest.from_identity_value(value.get("identity"))
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistenceError("recorded model request identity is malformed") from exc
    if request.request_hash != value["request_hash"] or request.request_hash != expected_hash:
        raise PersistenceError("recorded model request hash diverged")
    return request


def _response_from_artifact(value: object, *, request: ModelRequest) -> ModelResponse:
    if (
        not isinstance(value, dict)
        or set(value)
        != {"artifact_version", "provider_id", "request_hash", "response", "diagnostics"}
        or value.get("artifact_version") != "phase4-model-response-v1"
        or value.get("provider_id") != request.provider_id
        or value.get("request_hash") != request.request_hash
    ):
        raise PersistenceError("recorded model response artifact is malformed")
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {"provider_latency_ns"}:
        raise PersistenceError("recorded model response diagnostics are malformed")
    latency = diagnostics.get("provider_latency_ns")
    if latency is not None and (
        isinstance(latency, bool) or not isinstance(latency, int) or latency < 0
    ):
        raise PersistenceError("recorded provider latency is malformed")
    response = _response_from_value(value.get("response"))
    if response.request_hash != request.request_hash:
        raise PersistenceError("recorded model response hash identity diverged")
    return response


def _candidate_from_row(row: sqlite3.Row, limits: AstLimits) -> Candidate:
    mapping = dict(row)
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
        raise PersistenceError("Phase 4 candidate parents are malformed")
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


def _restore(
    database: Phase4Database, mechanism: Mechanism, limits: AstLimits
) -> tuple[dict[str, Candidate], set[str], set[str]]:
    candidates = {
        str(row["candidate_id"]): _candidate_from_row(row, limits) for row in database.candidates()
    }
    transitions = {
        int(row["evaluation_index"]): str(row["decision_json"]) for row in database.transitions()
    }
    canonical_seen: set[str] = set()
    semantic_seen: set[str] = set()
    for row in database.evaluations():
        candidate = candidates[str(row["candidate_id"])]
        result = _result_from_json(str(row["result_json"]))
        decision = mechanism.insert(candidate, result)
        if transitions[int(row["evaluation_index"])] != canonical_json(decision.to_value()):
            raise PersistenceError("Phase 4 mechanism diverged during resume")
        if not isinstance(candidate.ast, BitExpr) or candidate.semantic_hash is None:
            raise PersistenceError("Phase 4 recorded candidate is untyped")
        canonical_seen.add(ast_canonical_json(candidate.ast))
        semantic_seen.add(candidate.semantic_hash)
    return candidates, canonical_seen, semantic_seen


def _feedback(candidate_id: str, result: OracleResult) -> ParentScoreFeedback:
    return ParentScoreFeedback(
        candidate_id,
        result.type_valid,
        result.total,
        result.local_errors,
        result.local_cases,
        result.exact,
        result.ast_bits,
        result.residual_bits,
        result.ast_bits + result.residual_bits,
    )


def _event_payload(
    *,
    evaluation_index: int | None,
    request_index: int | None,
    item_ordinal: int | None,
    candidate: Candidate | None,
    result: OracleResult | None,
    decision: ArchiveDecision | None,
    rejection_reason: str | None,
    budget: Phase4BudgetState,
) -> JsonObject:
    return {
        "schema_version": PHASE4_EVENT_SCHEMA_VERSION,
        "evaluation_index": evaluation_index,
        "request_index": request_index,
        "item_ordinal": item_ordinal,
        "candidate": (
            {
                "candidate_id": candidate.candidate_id,
                "ordered_parent_ids": list(candidate.parent_ids),
                "proposer_id": candidate.proposer_id,
                "operator_id": candidate.operator_id,
                "context_hash": candidate.context_hash,
                "payload_hash": candidate.payload_hash,
                "ast_bits": encoded_length(candidate.ast)
                if isinstance(candidate.ast, BitExpr)
                else None,
            }
            if candidate is not None
            else None
        ),
        "oracle_result": result.deterministic_payload() if result is not None else None,
        "archive_decision": decision.to_value() if decision is not None else None,
        "rejection_reason": rejection_reason,
        "budget": budget.to_value(),
    }


class Phase4RunEngine:
    def __init__(
        self,
        *,
        repository_root: Path,
        run_directory: Path,
        config: AppConfig,
        prepared: PreparedPhase4,
        backend: ModelBackend,
        allow_new_dispatch: bool = True,
    ) -> None:
        if (
            config.model is None
            or config.cache is None
            or config.retry is None
            or config.phase4_budget is None
            or config.phase4_policy is None
        ):
            raise ConfigurationError("Phase 4 engine settings are incomplete")
        if (
            backend.backend_id != config.model.backend_id
            or backend.provider_id != config.model.provider_id
        ):
            raise ConfigurationError(
                "Phase 4 backend identity differs from the frozen model config"
            )
        self.repository_root = repository_root
        self.run_directory = run_directory
        self.config = config
        self.prepared = prepared
        self.backend = backend
        self.allow_new_dispatch = allow_new_dispatch
        namespace = config.cache.namespace
        if config.phase4_policy.stage == "locked-test":
            namespace = f"{namespace}-{run_directory.name}"
        self.cache = ExactResponseCache(repository_root / config.cache.root, namespace)
        self.proposer = LLMProposer(
            backend=backend,
            resolved_model=config.model.resolved_model,
            endpoint=config.model.endpoint,
            service_tier=config.model.service_tier,
            settings=config.model.request_settings(),
            limits=prepared.limits,
            allowed_macros=frozenset(config.dsl.allowed_macros) if config.dsl else frozenset(),
            cache=self.cache,
        )
        self.oracle = ExactDslOracle(
            prepared.hidden,
            limits=prepared.limits,
            response_mode=config.oracle.response_mode,
        )
        self.ledger: ProjectLedger | None = None
        if config.model.provider_id == "openai":
            ledger_path = repository_root / config.phase4_policy.ledger
            if not ledger_path.exists() and self._paid_artifacts_exist():
                raise PersistenceError(
                    "project ledger is missing while paid response artifacts exist; reconcile first"
                )
            self.ledger = ProjectLedger(ledger_path, prepared.price_policy)

    def _paid_artifacts_exist(self) -> bool:
        artifacts = self.repository_root / "artifacts"
        if not artifacts.exists():
            return False
        for path in artifacts.glob("**/responses/*.json"):
            try:
                if '"provider_id":"openai"' in path.read_text(encoding="utf-8"):
                    return True
            except OSError:
                continue
        return False

    def execute(self, *, interrupt_after: int | None = None) -> Phase4Outcome:
        if interrupt_after is not None and interrupt_after < 1:
            raise ConfigurationError("interrupt_after must be >= 1")
        mechanism = _mechanism(self.config, self.prepared.task)
        with Phase4Database(self.run_directory / "run.sqlite3") as database:
            state = database.state()
            terminal_statuses = {
                "completed",
                "cost-cap-exhausted",
                "usage-uncertain",
                "failed",
            }
            if (
                state.status in terminal_statuses
                and (self.run_directory / "results.json").is_file()
            ):
                return self._outcome(state.run_id, state.status, database.events())
            candidates, canonical_seen, semantic_seen = _restore(
                database, mechanism, self.prepared.limits
            )
            budget = database.budget()
            if state.status in terminal_statuses:
                return self._finalize(
                    database=database,
                    budget=budget,
                    mechanism=mechanism,
                    status=state.status,
                )
            database.set_status("running", utc_now())
            while state.next_evaluation < len(initialization_candidates()):
                index = state.next_evaluation
                source = initialization_candidates()[index]
                context_hash = sha256_json(
                    {
                        "phase4_initialization": "public-dsl-baselines-v1",
                        "task": self.prepared.task.public_view(),
                        "index": index,
                    }
                )
                candidate = self._candidate(
                    source=source,
                    parents=(),
                    operator_id="public-baseline-initialization",
                    context_hash=context_hash,
                )
                if not isinstance(candidate.ast, BitExpr):
                    raise PersistenceError("Phase 4 initialization candidate is not a DSL AST")
                evaluated = self.oracle.evaluate(candidate.ast)
                decision = mechanism.insert(candidate, evaluated.result)
                budget = budget.updated(oracle_invocations=1, evaluated_candidates=1)
                event = SearchEvent.create(
                    sequence=state.next_event,
                    event_type="phase4_initialization_evaluated",
                    logical_cost=budget.oracle_invocations,
                    payload=_event_payload(
                        evaluation_index=index,
                        request_index=None,
                        item_ordinal=None,
                        candidate=candidate,
                        result=evaluated.result,
                        decision=decision,
                        rejection_reason=None,
                        budget=budget,
                    ),
                    audit_timestamp=utc_now(),
                )
                database.append_evaluation(
                    candidate=candidate,
                    source_ast=source,
                    result=evaluated.result,
                    decision=decision,
                    oracle_version=self.config.oracle.oracle_id,
                    event=event,
                    budget=budget,
                    initialization_index=index,
                )
                candidates[candidate.candidate_id] = candidate
                canonical_seen.add(ast_canonical_json(candidate.ast))
                if candidate.semantic_hash is not None:
                    semantic_seen.add(candidate.semantic_hash)
                state = database.state()
                if interrupt_after is not None and state.next_evaluation >= interrupt_after:
                    database.set_status("interrupted", utc_now())
                    return self._outcome(state.run_id, "interrupted", database.events())

            incomplete = self._incomplete_request(database)
            if incomplete is not None:
                budget, stopped = self._resume_response(
                    database=database,
                    row=incomplete,
                    mechanism=mechanism,
                    candidates=candidates,
                    canonical_seen=canonical_seen,
                    semantic_seen=semantic_seen,
                    budget=budget,
                    interrupt_after=interrupt_after,
                )
                if stopped:
                    status = database.state().status
                    if status != "interrupted":
                        return self._finalize(
                            database=database,
                            budget=budget,
                            mechanism=mechanism,
                            status=status,
                        )
                    return self._outcome(state.run_id, status, database.events())

            if not self.allow_new_dispatch and not budget.exhausted:
                database.set_status("interrupted", utc_now())
                return self._outcome(database.state().run_id, "interrupted", database.events())

            while not budget.exhausted:
                batch_size = min(
                    self.config.proposer.batch_size,
                    budget.remaining_proposal_items,
                    budget.remaining_oracle_calls,
                )
                if batch_size < 1:
                    break
                selection, parent, parent_result = self._select_context(
                    mechanism=mechanism, database=database, budget=budget
                )
                role = ProposalRole.EXPLOIT
                call_index = budget.logical_model_calls
                request = self.proposer.build_request(
                    task=self.prepared.task.public_view(),
                    role=role,
                    batch_size=batch_size,
                    parent=parent,
                    feedback=(
                        _feedback(parent.candidate_id, parent_result)
                        if parent is not None and parent_result is not None
                        else None
                    ),
                )
                request = self._indexed_request(request, call_index)
                max_output = self.config.model.max_output_tokens if self.config.model else 0
                try:
                    budget.preflight(
                        input_token_bound=request.conservative_input_token_bound,
                        max_output_tokens=max_output,
                    )
                except ValueError:
                    break
                budget = budget.updated(
                    logical_model_calls=1,
                    scheduler_selections=int(parent is not None),
                )
                success = False
                if self.config.retry is None:
                    raise AssertionError("Phase 4 retry settings are unavailable")
                for retry_index in range(self.config.retry.max_retries + 1):
                    if budget.remaining_model_requests < 1:
                        break
                    try:
                        budget.preflight(
                            input_token_bound=request.conservative_input_token_bound,
                            max_output_tokens=max_output,
                        )
                    except ValueError:
                        break
                    try:
                        parsed, budget, stop, retry_allowed = self._dispatch_request(
                            database=database,
                            request=request,
                            logical_call_index=call_index,
                            retry_index=retry_index,
                            selection=selection,
                            parent=parent,
                            budget=budget,
                        )
                    except BudgetExhaustedError:
                        budget = database.budget()
                        return self._finalize(
                            database=database,
                            budget=budget,
                            mechanism=mechanism,
                            status="cost-cap-exhausted",
                        )
                    if stop:
                        return self._finalize(
                            database=database,
                            budget=budget,
                            mechanism=mechanism,
                            status=database.state().status,
                        )
                    if parsed is None:
                        if retry_allowed:
                            continue
                        break
                    request_index = database.state().next_request - 1
                    budget, interrupted = self._process_items(
                        database=database,
                        request_index=request_index,
                        parsed=parsed,
                        mechanism=mechanism,
                        parents=(parent,) if parent is not None else (),
                        candidates=candidates,
                        canonical_seen=canonical_seen,
                        semantic_seen=semantic_seen,
                        budget=budget,
                        start_ordinal=0,
                        interrupt_after=interrupt_after,
                    )
                    success = True
                    if interrupted:
                        database.set_status("interrupted", utc_now())
                        return self._outcome(state.run_id, "interrupted", database.events())
                    database.complete_request(request_index)
                    break
                if not success:
                    break
            return self._finalize(
                database=database, budget=budget, mechanism=mechanism, status="completed"
            )

    def _finalize(
        self,
        *,
        database: Phase4Database,
        budget: Phase4BudgetState,
        mechanism: Mechanism,
        status: str,
    ) -> Phase4Outcome:
        results = phase4_results(
            database=database,
            budget=budget,
            mechanism=mechanism,
            config=self.config,
            status=status,
        )
        analysis_hash = write_phase4_analysis(
            run_directory=self.run_directory,
            database=database,
            results=results,
            accesses=self.prepared.store.accesses,
            config=self.config,
            policy=self.prepared.price_policy,
        )
        results["analysis_manifest_hash"] = analysis_hash
        results["deterministic_summary_hash"] = sha256_json(results)
        write_content_artifact(self.run_directory / "results.json", canonical_json(results))
        database.set_status(status, utc_now())
        return self._outcome(database.state().run_id, status, database.events())

    def _indexed_request(self, request: ModelRequest, call_index: int) -> ModelRequest:
        settings = dict(request.settings)
        settings["independent_sample_index"] = call_index
        return ModelRequest(
            request.backend_id,
            request.provider_id,
            request.resolved_model,
            request.endpoint,
            request.service_tier,
            request.prompt_template,
            request.prompt_version,
            request.rendered_input,
            request.structured_schema_name,
            request.structured_schema_version,
            request.structured_schema,
            request.role,
            request.requested_batch_size,
            settings,
        )

    def _candidate(
        self,
        *,
        source: BitExpr,
        parents: tuple[CandidateSummary, ...],
        operator_id: str,
        context_hash: str,
    ) -> Candidate:
        document = DslCandidateDocument(source)
        canonical = document.ast
        return phase4_candidate(
            task_id=self.prepared.task.task_id,
            ast=canonical,
            parent_ids=tuple(parent.candidate_id for parent in parents),
            operator_id=operator_id,
            context_hash=context_hash,
            payload_hash=sha256_text(document.to_json()),
            semantic_hash=semantic_hash(canonical, limits=self.prepared.limits),
        )

    def _select_context(
        self, *, mechanism: Mechanism, database: Phase4Database, budget: Phase4BudgetState
    ) -> tuple[JsonObject, CandidateSummary | None, OracleResult | None]:
        condition = Phase4Condition(str(self.config.run.condition_id))
        if condition is Phase4Condition.DIRECT:
            return (
                {
                    "scheduler_version": "not-applicable-direct-independent-v1",
                    "selected_branch_id": None,
                    "eligible_branch_ids": [],
                },
                None,
                None,
            )
        decision = UniformScheduler().select(
            mechanism.branch_ids(),
            master_seed=self.config.run.seed,
            selection_counter=budget.logical_model_calls,
            remaining_proposal_attempts=budget.remaining_model_requests,
            remaining_oracle_calls=budget.remaining_oracle_calls,
        )
        if isinstance(mechanism, MapElitesArchive):
            coordinate = mechanism.coordinate_for_branch(decision.selected_branch_id)
            pool = mechanism.candidate_summaries(coordinate=coordinate)
        else:
            pool = mechanism.candidate_summaries()
        if not pool:
            raise PersistenceError("selected Phase 4 branch has no primary parent")
        parent = pool[0]
        result = _result_from_json(
            str(database.candidate_result(parent.candidate_id)["result_json"])
        )
        return decision.to_value(), parent, result

    def _dispatch_request(
        self,
        *,
        database: Phase4Database,
        request: ModelRequest,
        logical_call_index: int,
        retry_index: int,
        selection: JsonObject,
        parent: CandidateSummary | None,
        budget: Phase4BudgetState,
    ) -> tuple[LLMParsedResponse | None, Phase4BudgetState, bool, bool]:
        if self.config.phase4_policy is None or self.config.model is None:
            raise AssertionError("Phase 4 request settings are unavailable")
        if self.config.phase4_budget is None:
            raise AssertionError("Phase 4 budget settings are unavailable")
        if self.config.retry is None:
            raise AssertionError("Phase 4 retry settings are unavailable")
        cache_hit = self.cache.get(request)
        max_cost = self.prepared.price_policy.price.maximum_cost(
            input_token_bound=request.conservative_input_token_bound,
            max_output_tokens=self.config.model.max_output_tokens,
        )
        if max_cost > self.prepared.price_policy.request_cap_nano_usd:
            raise PersistenceError("request worst-case estimate exceeds the $0.01 ceiling")
        request_index = database.state().next_request
        reservation_id = sha256_text(
            f"phase4-reservation-v1\0{self.run_directory.name}\0{request_index}\0{request.request_hash}"
        )
        paid = self.config.model.provider_id == "openai" and cache_hit is None
        reserved = max_cost if paid else 0
        if paid:
            if self.ledger is None:
                raise PersistenceError("paid request has no project ledger")
            self.ledger.reserve(
                reservation_id=reservation_id,
                run_id=self.run_directory.name,
                stage=self.config.phase4_policy.stage,
                request_hash=request.request_hash,
                amount_nano_usd=reserved,
                child_cap_nano_usd=self.config.phase4_budget.child_nano_usd_cap,
            )
        prompt_name = f"prompts/request-{request_index:05d}.json"
        request_name = f"requests/request-{request_index:05d}.json"
        prompt_hash = write_content_artifact(
            self.run_directory / prompt_name, request.rendered_input
        )
        request_artifact: JsonObject = {
            "artifact_version": "phase4-model-request-v1",
            "request_hash": request.request_hash,
            "identity": request.identity_value(),
        }
        write_content_artifact(self.run_directory / request_name, canonical_json(request_artifact))
        record: JsonObject = {
            "logical_call_index": logical_call_index,
            "retry_index": retry_index,
            "condition_id": str(self.config.run.condition_id),
            "role": request.role.value,
            "selected_branch_id": selection.get("selected_branch_id"),
            "ordered_parent_ids": [parent.candidate_id] if parent is not None else [],
            "scheduler": selection,
            "prompt_artifact": prompt_name,
            "prompt_hash": prompt_hash,
            "request_artifact": request_name,
            "request_hash": request.request_hash,
            "backend_id": request.backend_id,
            "provider_id": request.provider_id,
            "resolved_model": request.resolved_model,
            "endpoint": request.endpoint,
            "service_tier": request.service_tier,
            "settings": request.settings,
            "batch_size": request.requested_batch_size,
            "cache_namespace": self.cache.namespace,
            "cache_key": request.request_hash,
            "cache_hit": cache_hit is not None,
            "reservation_id": reservation_id if paid else None,
            "reserved_nano_usd": reserved,
            "price_entry": self.prepared.price_policy.to_value()["price"],
        }
        database.prepare_request(record, budget=budget, timestamp=utc_now())
        response: ModelResponse
        if cache_hit is not None:
            response = cache_hit
        else:
            database.mark_dispatched(request_index)
            try:
                response = self.backend.dispatch(request)
            except ModelDispatchError as exc:
                return self._record_failure(
                    database=database,
                    request_index=request_index,
                    retry_index=retry_index,
                    error=exc.error,
                    reservation_id=reservation_id if paid else None,
                    reserved=reserved,
                    budget=budget,
                )
        response_name = f"responses/request-{request_index:05d}.json"
        response_artifact: JsonObject = {
            "artifact_version": "phase4-model-response-v1",
            "provider_id": request.provider_id,
            "request_hash": request.request_hash,
            "response": response.deterministic_value(),
            "diagnostics": {"provider_latency_ns": response.provider_latency_ns},
        }
        response_hash = write_content_artifact(
            self.run_directory / response_name, canonical_json(response_artifact)
        )
        usage = response.usage
        estimated = self.prepared.price_policy.price.cost(usage) if paid else 0
        released = 0
        if paid:
            assert self.ledger is not None
            _, released = self.ledger.reconcile(
                reservation_id=reservation_id,
                actual_nano_usd=estimated,
                usage_record={
                    "run_id": self.run_directory.name,
                    "request_index": request_index,
                    "request_hash": request.request_hash,
                    "usage": usage.to_value(),
                    "actual_nano_usd": estimated,
                    "price_policy_hash": self.prepared.price_policy.content_hash,
                    "response_hash": response_hash,
                },
            )
        budget = budget.updated(
            model_request_attempts=1,
            physical_provider_calls=int(cache_hit is None),
            exact_cache_hits=int(cache_hit is not None),
            retries=int(retry_index > 0),
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            total_tokens=usage.total_tokens,
            actual_nano_usd=estimated,
            released_nano_usd=released,
        )
        error: JsonObject | None
        try:
            parsed = self.proposer.parse_response(request, response)
            request_state = RequestState.RESPONDED
            error = None
            item_count: int | None = len(parsed.batch.items)
            if cache_hit is None:
                self.cache.put(request, response)
        except BatchEnvelopeError as exc:
            parsed = None
            request_state = RequestState.SCHEMA_FAILURE
            error = {
                "error_schema_version": 1,
                "category": "malformed-response",
                "retryable": True,
                "usage_uncertain": False,
                "detail": str(exc),
            }
            item_count = None
        database.finalize_request(
            request_index=request_index,
            state=request_state,
            provider_request_id=response.provider_request_id,
            response_artifact=response_name,
            response_hash=response_hash,
            error=error,
            usage=usage.to_value(),
            actual_nano_usd=estimated,
            released_nano_usd=released,
            uncertain_nano_usd=0,
            item_count=item_count,
            budget=budget,
        )
        retry_allowed = (
            parsed is None
            and retry_index < self.config.retry.max_retries
            and "malformed-response" in self.config.retry.retryable_categories
        )
        return parsed, budget, False, retry_allowed

    def _record_failure(
        self,
        *,
        database: Phase4Database,
        request_index: int,
        retry_index: int,
        error: ModelError,
        reservation_id: str | None,
        reserved: int,
        budget: Phase4BudgetState,
    ) -> tuple[None, Phase4BudgetState, bool, bool]:
        failure_name = f"responses/request-{request_index:05d}-failure.json"
        failure_artifact: JsonObject = {
            "artifact_version": "phase4-model-failure-v1",
            "request_index": request_index,
            "error": error.to_value(),
        }
        failure_hash = write_content_artifact(
            self.run_directory / failure_name, canonical_json(failure_artifact)
        )
        uncertain = 0
        released = 0
        if reservation_id is not None and self.ledger is not None:
            if error.usage_uncertain:
                uncertain = self.ledger.mark_uncertain(
                    reservation_id=reservation_id,
                    failure_record={
                        "run_id": self.run_directory.name,
                        "request_index": request_index,
                        "error": error.to_value(),
                        "uncertain_nano_usd": reserved,
                        "response_hash": failure_hash,
                    },
                )
            else:
                _, released = self.ledger.reconcile(
                    reservation_id=reservation_id,
                    actual_nano_usd=0,
                    usage_record={
                        "run_id": self.run_directory.name,
                        "request_index": request_index,
                        "error": error.to_value(),
                        "actual_nano_usd": 0,
                        "response_hash": failure_hash,
                    },
                )
        usage = error.usage
        budget = budget.updated(
            model_request_attempts=1,
            physical_provider_calls=1,
            retries=int(retry_index > 0),
            input_tokens=usage.input_tokens if usage else 0,
            cached_input_tokens=usage.cached_input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            reasoning_tokens=usage.reasoning_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            uncertain_nano_usd=uncertain,
            released_nano_usd=released,
        )
        state = RequestState.USAGE_UNCERTAIN if error.usage_uncertain else RequestState.FAILED
        database.finalize_request(
            request_index=request_index,
            state=state,
            provider_request_id=error.provider_request_id,
            response_artifact=failure_name,
            response_hash=failure_hash,
            error=error.to_value(),
            usage=usage.to_value() if usage else None,
            actual_nano_usd=0,
            released_nano_usd=released,
            uncertain_nano_usd=uncertain,
            item_count=None,
            budget=budget,
        )
        if error.usage_uncertain:
            database.set_status("usage-uncertain", utc_now())
            return None, budget, True, False
        if self.config.retry is None:
            raise AssertionError("Phase 4 retry settings are unavailable")
        retry_allowed = (
            error.retryable
            and error.category.value in self.config.retry.retryable_categories
            and retry_index < self.config.retry.max_retries
        )
        return None, budget, False, retry_allowed

    def _process_items(
        self,
        *,
        database: Phase4Database,
        request_index: int,
        parsed: LLMParsedResponse,
        mechanism: Mechanism,
        parents: tuple[CandidateSummary, ...],
        candidates: dict[str, Candidate],
        canonical_seen: set[str],
        semantic_seen: set[str],
        budget: Phase4BudgetState,
        start_ordinal: int,
        interrupt_after: int | None,
    ) -> tuple[Phase4BudgetState, bool]:
        proposal_by_ordinal = {proposal.ordinal: proposal for proposal in parsed.proposals}
        for item in parsed.batch.items[start_ordinal:]:
            state = database.state()
            if not item.accepted:
                budget = budget.updated(proposal_items=1, invalid_items=1)
                reason = item.rejection_reason or "invalid candidate item"
                event = SearchEvent.create(
                    sequence=state.next_event,
                    event_type="phase4_model_item_rejected",
                    logical_cost=budget.oracle_invocations,
                    payload=_event_payload(
                        evaluation_index=None,
                        request_index=request_index,
                        item_ordinal=item.ordinal,
                        candidate=None,
                        result=None,
                        decision=None,
                        rejection_reason=reason,
                        budget=budget,
                    ),
                    audit_timestamp=utc_now(),
                )
                database.append_rejected_item(
                    request_index=request_index,
                    ordinal=item.ordinal,
                    submitted_document=item.submitted_document,
                    rejection_reason=reason,
                    event=event,
                    budget=budget,
                )
                continue
            proposal = proposal_by_ordinal[item.ordinal]
            canonical_key = ast_canonical_json(proposal.canonical_ast)
            semantics = semantic_hash(proposal.canonical_ast, limits=self.prepared.limits)
            canonical_duplicate = canonical_key in canonical_seen
            semantic_duplicate = semantics in semantic_seen
            payload_hash = sha256_text(canonical_json(proposal.submitted_document))
            candidate = phase4_candidate(
                task_id=self.prepared.task.task_id,
                ast=proposal.canonical_ast,
                parent_ids=tuple(parent.candidate_id for parent in parents),
                operator_id=proposal.operator_id,
                context_hash=str(database.request(request_index)["prompt_hash"]),
                payload_hash=payload_hash,
                semantic_hash=semantics,
            )
            evaluated = self.oracle.evaluate(proposal.canonical_ast)
            decision = mechanism.insert(candidate, evaluated.result)
            budget = budget.updated(
                proposal_items=1,
                valid_items=1,
                canonical_duplicates=int(canonical_duplicate),
                semantic_duplicates=int(semantic_duplicate),
                oracle_invocations=1,
                evaluated_candidates=1,
            )
            event = SearchEvent.create(
                sequence=state.next_event,
                event_type="phase4_model_item_evaluated",
                logical_cost=budget.oracle_invocations,
                payload=_event_payload(
                    evaluation_index=state.next_evaluation,
                    request_index=request_index,
                    item_ordinal=item.ordinal,
                    candidate=candidate,
                    result=evaluated.result,
                    decision=decision,
                    rejection_reason=None,
                    budget=budget,
                ),
                audit_timestamp=utc_now(),
            )
            database.append_evaluation(
                candidate=candidate,
                source_ast=proposal.source_ast,
                result=evaluated.result,
                decision=decision,
                oracle_version=self.config.oracle.oracle_id,
                event=event,
                budget=budget,
                request_index=request_index,
                item_ordinal=item.ordinal,
                submitted_document=item.submitted_document,
                canonical_duplicate=canonical_duplicate,
                semantic_duplicate=semantic_duplicate,
            )
            candidates[candidate.candidate_id] = candidate
            canonical_seen.add(canonical_key)
            semantic_seen.add(semantics)
            if interrupt_after is not None and database.state().next_evaluation >= interrupt_after:
                return budget, True
        return budget, False

    def _incomplete_request(self, database: Phase4Database) -> sqlite3.Row | None:
        requests = database.requests()
        if not requests:
            return None
        last = requests[-1]
        if last["state"] == "responded" and last["next_item_ordinal"] <= last["item_count"]:
            return last
        if last["state"] in {
            "pending",
            "dispatched",
            "schema-failure",
            "failed",
            "usage-uncertain",
        }:
            return last
        return None

    def _resume_response(
        self,
        *,
        database: Phase4Database,
        row: sqlite3.Row,
        mechanism: Mechanism,
        candidates: dict[str, Candidate],
        canonical_seen: set[str],
        semantic_seen: set[str],
        budget: Phase4BudgetState,
        interrupt_after: int | None,
    ) -> tuple[Phase4BudgetState, bool]:
        mapping = dict(row)
        if mapping["state"] in {"schema-failure", "failed", "usage-uncertain"}:
            request_name = mapping.get("request_artifact")
            response_name = mapping.get("response_artifact")
            response_hash = mapping.get("response_hash")
            if (
                not isinstance(request_name, str)
                or not isinstance(response_name, str)
                or not isinstance(response_hash, str)
            ):
                raise PersistenceError("finalized recovery artifacts are missing")
            request_artifact = parse_json_object(
                read_text_artifact(self.run_directory / request_name)
            )
            request = _request_from_artifact(
                request_artifact, expected_hash=mapping.get("request_hash")
            )
            response_text = read_text_artifact(self.run_directory / response_name)
            if sha256_text(response_text) != response_hash:
                raise PersistenceError("finalized recovery artifact hash mismatch")
            retry_allowed = self._malformed_retry_allowed(mapping)
            if mapping["state"] in {"failed", "usage-uncertain"}:
                failure_artifact = parse_json_object(response_text)
                try:
                    error = ModelError.from_value(failure_artifact.get("error"))
                except ValueError as exc:
                    raise PersistenceError("finalized failure record is malformed") from exc
                if mapping["state"] == "usage-uncertain":
                    database.set_status("usage-uncertain", utc_now())
                    return budget, True
                if self.config.retry is None:
                    raise AssertionError("Phase 4 retry settings are unavailable")
                retry_allowed = (
                    error.retryable
                    and error.category.value in self.config.retry.retryable_categories
                    and _mapping_int(mapping, "retry_index") < self.config.retry.max_retries
                )
            return self._continue_recovered_sequence(
                database=database,
                mapping=mapping,
                request=request,
                retry_allowed=retry_allowed,
                mechanism=mechanism,
                candidates=candidates,
                canonical_seen=canonical_seen,
                semantic_seen=semantic_seen,
                budget=budget,
                interrupt_after=interrupt_after,
            )
        if mapping["state"] in {"pending", "dispatched"}:
            request_name = mapping.get("request_artifact")
            if not isinstance(request_name, str):
                raise PersistenceError("prepared Phase 4 request artifact is missing")
            request_artifact = parse_json_object(
                read_text_artifact(self.run_directory / request_name)
            )
            request = _request_from_artifact(
                request_artifact, expected_hash=mapping.get("request_hash")
            )
            response_name = f"responses/request-{int(mapping['request_index']):05d}.json"
            response_path = self.run_directory / response_name
            failure_name = f"responses/request-{int(mapping['request_index']):05d}-failure.json"
            failure_path = self.run_directory / failure_name
            if response_path.is_file() and failure_path.is_file():
                raise PersistenceError("request has conflicting durable response artifacts")
            if mapping["state"] == "pending":
                if not self.allow_new_dispatch:
                    database.set_status("interrupted", utc_now())
                    return budget, True
                cache_hit = bool(mapping["cache_hit"])
                response = self.cache.get(request) if cache_hit else None
                if cache_hit and response is None:
                    raise PersistenceError("prepared exact cache hit disappeared before resume")
                if response is None:
                    database.mark_dispatched(int(mapping["request_index"]))
                    try:
                        response = self.backend.dispatch(request)
                    except ModelDispatchError as exc:
                        _, budget, stop, retry_allowed = self._record_failure(
                            database=database,
                            request_index=int(mapping["request_index"]),
                            retry_index=int(mapping["retry_index"]),
                            error=exc.error,
                            reservation_id=(
                                str(mapping["reservation_id"])
                                if mapping.get("reservation_id") is not None
                                else None
                            ),
                            reserved=int(mapping["reserved_nano_usd"]),
                            budget=budget,
                        )
                        if stop:
                            return budget, True
                        return self._continue_recovered_sequence(
                            database=database,
                            mapping=mapping,
                            request=request,
                            retry_allowed=retry_allowed,
                            mechanism=mechanism,
                            candidates=candidates,
                            canonical_seen=canonical_seen,
                            semantic_seen=semantic_seen,
                            budget=budget,
                            interrupt_after=interrupt_after,
                        )
                response_artifact: JsonObject = {
                    "artifact_version": "phase4-model-response-v1",
                    "provider_id": request.provider_id,
                    "request_hash": request.request_hash,
                    "response": response.deterministic_value(),
                    "diagnostics": {"provider_latency_ns": response.provider_latency_ns},
                }
                response_hash = write_content_artifact(
                    response_path, canonical_json(response_artifact)
                )
                parsed, budget = self._finalize_recovered_response(
                    database=database,
                    mapping=mapping,
                    request=request,
                    response=response,
                    response_name=response_name,
                    response_hash=response_hash,
                    budget=budget,
                )
                if parsed is None:
                    return self._continue_recovered_sequence(
                        database=database,
                        mapping=mapping,
                        request=request,
                        retry_allowed=self._malformed_retry_allowed(mapping),
                        mechanism=mechanism,
                        candidates=candidates,
                        canonical_seen=canonical_seen,
                        semantic_seen=semantic_seen,
                        budget=budget,
                        interrupt_after=interrupt_after,
                    )
                mapping = dict(database.request(int(mapping["request_index"])))
            elif response_path.is_file():
                response_text = read_text_artifact(response_path)
                response_artifact = parse_json_object(response_text)
                response = _response_from_artifact(response_artifact, request=request)
                parsed, budget = self._finalize_recovered_response(
                    database=database,
                    mapping=mapping,
                    request=request,
                    response=response,
                    response_name=response_name,
                    response_hash=sha256_text(response_text),
                    budget=budget,
                )
                if parsed is None:
                    return self._continue_recovered_sequence(
                        database=database,
                        mapping=mapping,
                        request=request,
                        retry_allowed=self._malformed_retry_allowed(mapping),
                        mechanism=mechanism,
                        candidates=candidates,
                        canonical_seen=canonical_seen,
                        semantic_seen=semantic_seen,
                        budget=budget,
                        interrupt_after=interrupt_after,
                    )
                mapping = dict(database.request(int(mapping["request_index"])))
            elif failure_path.is_file():
                failure_text = read_text_artifact(failure_path)
                failure_artifact = parse_json_object(failure_text)
                if (
                    set(failure_artifact) != {"artifact_version", "request_index", "error"}
                    or failure_artifact.get("artifact_version") != "phase4-model-failure-v1"
                    or failure_artifact.get("request_index") != mapping["request_index"]
                ):
                    raise PersistenceError("durable model failure artifact is malformed")
                try:
                    error = ModelError.from_value(failure_artifact.get("error"))
                except ValueError as exc:
                    raise PersistenceError("durable model failure record is malformed") from exc
                _, budget, stop, retry_allowed = self._record_failure(
                    database=database,
                    request_index=_mapping_int(mapping, "request_index"),
                    retry_index=_mapping_int(mapping, "retry_index"),
                    error=error,
                    reservation_id=(
                        str(mapping["reservation_id"])
                        if mapping.get("reservation_id") is not None
                        else None
                    ),
                    reserved=_mapping_int(mapping, "reserved_nano_usd"),
                    budget=budget,
                )
                if stop:
                    return budget, True
                return self._continue_recovered_sequence(
                    database=database,
                    mapping=mapping,
                    request=request,
                    retry_allowed=retry_allowed,
                    mechanism=mechanism,
                    candidates=candidates,
                    canonical_seen=canonical_seen,
                    semantic_seen=semantic_seen,
                    budget=budget,
                    interrupt_after=interrupt_after,
                )
            else:
                # A dispatched request without a durable response is never duplicated.
                _, budget, _, _ = self._record_failure(
                    database=database,
                    request_index=_mapping_int(mapping, "request_index"),
                    retry_index=_mapping_int(mapping, "retry_index"),
                    error=ModelError(
                        ModelErrorCategory.UNKNOWN,
                        retryable=False,
                        usage_uncertain=True,
                    ),
                    reservation_id=(
                        str(mapping["reservation_id"])
                        if mapping.get("reservation_id") is not None
                        else None
                    ),
                    reserved=_mapping_int(mapping, "reserved_nano_usd"),
                    budget=budget,
                )
                return budget, True
        recorded_response_name = mapping.get("response_artifact")
        request_name = mapping.get("request_artifact")
        if not isinstance(recorded_response_name, str) or not isinstance(request_name, str):
            raise PersistenceError("resumable Phase 4 response artifacts are missing")
        response_artifact = parse_json_object(
            read_text_artifact(self.run_directory / recorded_response_name)
        )
        request_artifact = parse_json_object(read_text_artifact(self.run_directory / request_name))
        request = _request_from_artifact(
            request_artifact, expected_hash=mapping.get("request_hash")
        )
        response = _response_from_artifact(response_artifact, request=request)
        parsed = self.proposer.parse_response(request, response)
        parent_ids = json.loads(str(mapping["ordered_parent_ids_json"]))
        parents = tuple(
            CandidateSummary(candidate_id, candidates[candidate_id].ast)
            for candidate_id in parent_ids
        )
        budget, interrupted = self._process_items(
            database=database,
            request_index=int(mapping["request_index"]),
            parsed=parsed,
            mechanism=mechanism,
            parents=parents,
            candidates=candidates,
            canonical_seen=canonical_seen,
            semantic_seen=semantic_seen,
            budget=budget,
            start_ordinal=int(mapping["next_item_ordinal"]),
            interrupt_after=interrupt_after,
        )
        if interrupted:
            database.set_status("interrupted", utc_now())
            return budget, True
        database.complete_request(int(mapping["request_index"]))
        return budget, False

    def _malformed_retry_allowed(self, mapping: dict[str, object]) -> bool:
        if self.config.retry is None:
            raise AssertionError("Phase 4 retry settings are unavailable")
        return (
            _mapping_int(mapping, "retry_index") < self.config.retry.max_retries
            and "malformed-response" in self.config.retry.retryable_categories
        )

    def _continue_recovered_sequence(
        self,
        *,
        database: Phase4Database,
        mapping: dict[str, object],
        request: ModelRequest,
        retry_allowed: bool,
        mechanism: Mechanism,
        candidates: dict[str, Candidate],
        canonical_seen: set[str],
        semantic_seen: set[str],
        budget: Phase4BudgetState,
        interrupt_after: int | None,
    ) -> tuple[Phase4BudgetState, bool]:
        if not retry_allowed:
            database.set_status("completed", utc_now())
            return budget, True
        if not self.allow_new_dispatch:
            database.set_status("interrupted", utc_now())
            return budget, True
        if self.config.retry is None or self.config.model is None:
            raise AssertionError("Phase 4 retry/model settings are unavailable")
        parent_ids = json.loads(str(mapping["ordered_parent_ids_json"]))
        selection = json.loads(str(mapping["scheduler_json"]))
        if (
            not isinstance(parent_ids, list)
            or not all(isinstance(candidate_id, str) for candidate_id in parent_ids)
            or not isinstance(selection, dict)
            or not all(isinstance(key, str) for key in selection)
        ):
            raise PersistenceError("recovered retry context is malformed")
        parents = tuple(
            CandidateSummary(candidate_id, candidates[candidate_id].ast)
            for candidate_id in parent_ids
        )
        parent = parents[0] if parents else None
        retry_start = _mapping_int(mapping, "retry_index") + 1
        for retry_index in range(retry_start, self.config.retry.max_retries + 1):
            if budget.remaining_model_requests < 1:
                break
            try:
                budget.preflight(
                    input_token_bound=request.conservative_input_token_bound,
                    max_output_tokens=self.config.model.max_output_tokens,
                )
            except ValueError:
                break
            try:
                parsed, budget, stop, may_retry = self._dispatch_request(
                    database=database,
                    request=request,
                    logical_call_index=_mapping_int(mapping, "logical_call_index"),
                    retry_index=retry_index,
                    selection=cast(JsonObject, selection),
                    parent=parent,
                    budget=budget,
                )
            except BudgetExhaustedError:
                budget = database.budget()
                database.set_status("cost-cap-exhausted", utc_now())
                return budget, True
            if stop:
                return budget, True
            if parsed is None:
                if may_retry:
                    continue
                break
            request_index = database.state().next_request - 1
            budget, interrupted = self._process_items(
                database=database,
                request_index=request_index,
                parsed=parsed,
                mechanism=mechanism,
                parents=parents,
                candidates=candidates,
                canonical_seen=canonical_seen,
                semantic_seen=semantic_seen,
                budget=budget,
                start_ordinal=0,
                interrupt_after=interrupt_after,
            )
            if interrupted:
                database.set_status("interrupted", utc_now())
                return budget, True
            database.complete_request(request_index)
            return budget, False
        database.set_status("completed", utc_now())
        return budget, True

    def _finalize_recovered_response(
        self,
        *,
        database: Phase4Database,
        mapping: dict[str, object],
        request: ModelRequest,
        response: ModelResponse,
        response_name: str,
        response_hash: str,
        budget: Phase4BudgetState,
    ) -> tuple[LLMParsedResponse | None, Phase4BudgetState]:
        paid = isinstance(mapping.get("reservation_id"), str)
        estimated = self.prepared.price_policy.price.cost(response.usage) if paid else 0
        released = 0
        if paid:
            if self.ledger is None:
                raise PersistenceError("recovered paid response has no project ledger")
            reservation_id = str(mapping["reservation_id"])
            _, released = self.ledger.reconcile(
                reservation_id=reservation_id,
                actual_nano_usd=estimated,
                usage_record={
                    "run_id": self.run_directory.name,
                    "request_index": _mapping_int(mapping, "request_index"),
                    "request_hash": request.request_hash,
                    "usage": response.usage.to_value(),
                    "actual_nano_usd": estimated,
                    "price_policy_hash": self.prepared.price_policy.content_hash,
                    "response_hash": response_hash,
                },
            )
        usage = response.usage
        cache_hit = bool(mapping["cache_hit"])
        budget = budget.updated(
            model_request_attempts=1,
            physical_provider_calls=int(not cache_hit),
            exact_cache_hits=int(cache_hit),
            retries=int(_mapping_int(mapping, "retry_index") > 0),
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            total_tokens=usage.total_tokens,
            actual_nano_usd=estimated,
            released_nano_usd=released,
        )
        error: JsonObject | None
        try:
            parsed = self.proposer.parse_response(request, response)
            state = RequestState.RESPONDED
            error = None
            item_count: int | None = len(parsed.batch.items)
            if not cache_hit:
                self.cache.put(request, response)
        except BatchEnvelopeError as exc:
            parsed = None
            state = RequestState.SCHEMA_FAILURE
            error = {
                "error_schema_version": 1,
                "category": "malformed-response",
                "retryable": True,
                "usage_uncertain": False,
                "detail": str(exc),
            }
            item_count = None
        database.finalize_request(
            request_index=_mapping_int(mapping, "request_index"),
            state=state,
            provider_request_id=response.provider_request_id,
            response_artifact=response_name,
            response_hash=response_hash,
            error=error,
            usage=usage.to_value(),
            actual_nano_usd=estimated,
            released_nano_usd=released,
            uncertain_nano_usd=0,
            item_count=item_count,
            budget=budget,
        )
        return parsed, budget

    def _outcome(self, run_id: str, status: str, events: tuple[SearchEvent, ...]) -> Phase4Outcome:
        return Phase4Outcome(
            run_id,
            status,
            len(events),
            tuple(event.payload_hash for event in events),
            self.run_directory,
        )


def _response_from_value(value: object) -> ModelResponse:
    try:
        return ModelResponse.from_deterministic_value(value)
    except (TypeError, ValueError) as exc:
        raise PersistenceError("recorded model response is malformed") from exc


def phase4_results(
    *,
    database: Phase4Database,
    budget: Phase4BudgetState,
    mechanism: Mechanism,
    config: AppConfig,
    status: str = "completed",
) -> JsonObject:
    exact_calls: list[int] = []
    exact_bits: list[int] = []
    two_part: list[int] = []
    for row in database.evaluations():
        result = _result_from_json(str(row["result_json"]))
        call = int(row["evaluation_index"]) + 1
        two_part.append(result.ast_bits + result.residual_bits)
        if result.exact:
            exact_calls.append(call)
            exact_bits.append(result.ast_bits)
    first_exact = min(exact_calls) if exact_calls else None
    auc_numerator = budget.oracle_call_cap - first_exact + 1 if first_exact is not None else 0
    request_states = Counter(str(row["state"]) for row in database.requests())
    metrics: JsonObject = {
        "normalized_exact_auc": auc_numerator / budget.oracle_call_cap,
        "exact_auc_numerator": auc_numerator,
        "exact_auc_denominator": budget.oracle_call_cap,
        "final_exact_solved": bool(exact_calls),
        "calls_to_first_exact": first_exact,
        "best_exact_ast_bits": min(exact_bits) if exact_bits else None,
        "best_two_part_bits": min(two_part) if two_part else None,
        "archive_coverage": len(mechanism.cells) if isinstance(mechanism, MapElitesArchive) else 0,
        "distinct_candidate_semantics": len(
            {str(row["semantic_hash"]) for row in database.candidates()}
        ),
        "valid_proposal_rate": budget.valid_items / budget.proposal_items
        if budget.proposal_items
        else 0.0,
        "invalid_proposal_rate": budget.invalid_items / budget.proposal_items
        if budget.proposal_items
        else 0.0,
        "canonical_duplicate_rate": budget.canonical_duplicates / budget.valid_items
        if budget.valid_items
        else 0.0,
        "semantic_duplicate_rate": budget.semantic_duplicates / budget.valid_items
        if budget.valid_items
        else 0.0,
        "request_states": dict(sorted(request_states.items())),
    }
    return {
        "schema_version": 4,
        "evidence_class": "fake"
        if config.model and config.model.provider_id == "scripted"
        else "live",
        "status": status,
        "condition_id": config.run.condition_id,
        "task_id": config.run.task_id,
        "search_seed": config.run.seed,
        "metrics": metrics,
        "budget": budget.to_value(),
        "event_payload_hashes": [event.payload_hash for event in database.events()],
    }


def write_phase4_analysis(
    *,
    run_directory: Path,
    database: Phase4Database,
    results: JsonObject,
    accesses: tuple[OracleTaskAccess, ...],
    config: AppConfig,
    policy: PricePolicy,
) -> str:
    analysis = run_directory / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    def record(name: str, value: object) -> None:
        text = value if isinstance(value, str) else canonical_json(value)
        normalized = text.rstrip("\n")
        write_content_artifact(analysis / name, normalized)
        files[name] = sha256_text(normalized)

    requests = [dict(row) for row in database.requests()]
    request_manifest = {
        "request_manifest_version": "phase4-request-manifest-v1",
        "requests": [
            {
                "request_index": row["request_index"],
                "logical_call_index": row["logical_call_index"],
                "retry_index": row["retry_index"],
                "state": row["state"],
                "request_hash": row["request_hash"],
                "prompt_hash": row["prompt_hash"],
                "response_hash": row["response_hash"],
                "cache_hit": bool(row["cache_hit"]),
                "provider_request_id": row["provider_request_id"],
                "prompt_artifact": row["prompt_artifact"],
                "request_artifact": row["request_artifact"],
                "response_artifact": row["response_artifact"],
            }
            for row in requests
        ],
    }
    record("request-manifest.json", request_manifest)
    record("budget-reconciliation.json", results["budget"])
    record(
        "cost-reconciliation.json",
        {
            "price_policy_hash": policy.content_hash,
            "price_entry": policy.to_value()["price"],
            "requests": [
                {
                    "request_index": row["request_index"],
                    "reserved_nano_usd": row["reserved_nano_usd"],
                    "actual_nano_usd": row["actual_nano_usd"],
                    "released_nano_usd": row["released_nano_usd"],
                    "uncertain_nano_usd": row["uncertain_nano_usd"],
                    "usage": json.loads(row["usage_json"]) if row["usage_json"] else None,
                }
                for row in requests
            ],
        },
    )
    curve = ["oracle_calls,total_tokens,nano_usd,best_exact"]
    solved = 0
    for row in database.evaluations():
        result = _result_from_json(str(row["result_json"]))
        solved = max(solved, int(result.exact))
        budget_at_event = next(
            (
                event.payload.get("budget")
                for event in database.events()
                if event.payload.get("evaluation_index") == row["evaluation_index"]
            ),
            None,
        )
        counters = budget_at_event.get("counters", {}) if isinstance(budget_at_event, dict) else {}

        total_tokens = _budget_counter(counters, "total_tokens")
        actual = _budget_counter(counters, "actual_nano_usd")
        uncertain = _budget_counter(counters, "uncertain_nano_usd")
        curve.append(
            f"{int(row['evaluation_index']) + 1},{total_tokens},{actual + uncertain},{solved}"
        )
    record("exact-curves.csv", "\n".join(curve) + "\n")
    record(
        "lineage.json",
        {
            "lineage_version": "phase4-lineage-v1",
            "candidates": [dict(row) for row in database.candidates()],
            "edges": [dict(row) for row in database.lineage()],
        },
    )
    record(
        "proposal-diagnostics.json",
        {
            "items": [dict(row) for row in database.items()],
            "request_states": dict(sorted(Counter(row["state"] for row in requests).items())),
        },
    )
    record(
        "access-ledger.json",
        {
            "authority_mode": (
                "phase4-locked-test-once-v1"
                if config.run.split is SplitLabel.TEST
                else "phase4-training-development-v1"
            ),
            "accesses": [
                {"task_id": item.task_id, "split": item.split.value, "purpose": item.purpose}
                for item in accesses
            ],
            "test_consumed": config.run.split is SplitLabel.TEST,
            "test_oracle_accesses": sum(item.split is SplitLabel.TEST for item in accesses),
        },
    )
    record(
        "failure-analysis.json",
        {
            "provider_failures": [
                {
                    "request_index": row["request_index"],
                    "state": row["state"],
                    "error": row["error_json"],
                }
                for row in requests
                if row["state"] in {"failed", "usage-uncertain", "schema-failure"}
            ],
            "limitations": [
                "F0 only; proposal first-release F1/F2 families are not implemented",
                "fake evidence is executable evidence, not scientific H1/H2 evidence",
                "live provider outputs are replayable records but not bitwise regenerable",
            ],
        },
    )
    provider_latency: list[int] = []
    for request_record in requests:
        response_name = request_record["response_artifact"]
        if not isinstance(response_name, str) or response_name.endswith("-failure.json"):
            continue
        response_artifact = parse_json_object(read_text_artifact(run_directory / response_name))
        diagnostics = response_artifact.get("diagnostics")
        latency = diagnostics.get("provider_latency_ns") if isinstance(diagnostics, dict) else None
        if isinstance(latency, int) and not isinstance(latency, bool):
            provider_latency.append(latency)
    oracle_runtime = sum(int(row["runtime_ns"]) for row in database.evaluations())
    runtime_diagnostics: JsonObject = {
        "diagnostic_version": "phase4-nondeterministic-runtime-v1",
        "provider_latency_ns": cast(list[JsonValue], provider_latency),
        "provider_latency_ns_total": sum(provider_latency),
        "oracle_runtime_ns_total": oracle_runtime,
        "excluded_from_deterministic_metrics": True,
    }
    write_content_artifact(
        analysis / "runtime-diagnostics.json", canonical_json(runtime_diagnostics)
    )
    manifest: JsonObject = {
        "analysis_artifact_version": PHASE4_REPORT_VERSION,
        "source": "committed-phase4-records-only",
        "files": cast(JsonObject, files),
        "non_deterministic_diagnostic_files": ["runtime-diagnostics.json"],
    }
    text = canonical_json(manifest)
    write_content_artifact(analysis / "manifest.json", text)
    return sha256_text(text)


def start_phase4_run(
    *,
    repository_root: Path,
    config: AppConfig,
    config_source: str,
    run_id: str,
    interrupt_after: int | None,
    allow_live_model: bool,
    authority: Phase4Authority | None = None,
    backend: ModelBackend | None = None,
) -> Phase4Outcome:
    from world_model_search.search.loop import validate_run_id

    selected = validate_run_id(run_id)
    run_directory = repository_root / config.run.root / selected
    if run_directory.exists():
        raise PersistenceError(f"run already exists: {selected}")
    resolved_backend = backend or _backend(config, allow_live_model=allow_live_model)
    selected_authority = authority or Phase4Authority.ordinary()
    prepared = prepare_phase4(
        repository_root=repository_root,
        config=config,
        authority=selected_authority,
        purpose=(
            "phase4-locked-test-once"
            if selected_authority.mode == "phase4-locked-test-once-v1"
            else "phase4-recorded-llm-run"
        ),
    )
    manifest = build_manifest(
        repository_root=repository_root,
        run_id=selected,
        config=config,
        config_source=config_source,
        task=prepared.task,
    )
    manifest["phase4_authority"] = selected_authority.to_value()
    manifest["phase4_authority_hash"] = sha256_json(selected_authority.to_value())
    run_directory.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(run_directory / "manifest.json", manifest)
    if config.phase4_budget is None:
        raise AssertionError("validated Phase 4 config has no budget")
    budget = Phase4BudgetState(
        config.phase4_budget.model_request_cap,
        config.phase4_budget.input_token_cap,
        config.phase4_budget.output_token_cap,
        config.phase4_budget.total_token_cap,
        config.phase4_budget.proposal_item_cap,
        config.phase4_budget.oracle_call_cap,
        config.phase4_budget.child_nano_usd_cap,
    )
    with Phase4Database(run_directory / "run.sqlite3") as database:
        database.initialize(
            run_id=selected,
            task=prepared.task,
            timestamp=utc_now(),
            budget=budget,
        )
    return Phase4RunEngine(
        repository_root=repository_root,
        run_directory=run_directory,
        config=config,
        prepared=prepared,
        backend=resolved_backend,
    ).execute(interrupt_after=interrupt_after)


def authority_from_manifest(manifest: JsonObject) -> Phase4Authority:
    raw = manifest.get("phase4_authority")
    if not isinstance(raw, dict):
        raise PersistenceError("Phase 4 manifest has no authority")
    splits = raw.get("allowed_splits")
    tasks = raw.get("frozen_task_ids")
    if not isinstance(splits, list) or not isinstance(tasks, list):
        raise PersistenceError("Phase 4 authority is malformed")
    authority = Phase4Authority(
        str(raw.get("mode")),
        frozenset(SplitLabel(str(value)) for value in splits),
        tuple(str(value) for value in tasks),
        str(raw["freeze_hash"]) if raw.get("freeze_hash") is not None else None,
    )
    if sha256_json(authority.to_value()) != manifest.get("phase4_authority_hash"):
        raise PersistenceError("Phase 4 authority hash mismatch")
    return authority


def resume_phase4_run(
    *,
    repository_root: Path,
    run_directory: Path,
    config: AppConfig,
    manifest: JsonObject,
    interrupt_after: int | None,
    allow_live_model: bool,
) -> Phase4Outcome:
    if manifest.get("manifest_schema_version") != PHASE4_MANIFEST_SCHEMA_VERSION:
        raise PersistenceError("Phase 4 configuration requires manifest schema 5")
    allow_new_dispatch = True
    if config.model is None:
        raise PersistenceError("Phase 4 manifest has no model settings")
    opt_in = LiveOptIn.resolve(allow_live_model)
    if config.model.provider_id == "openai" and not (
        opt_in.cli_allowed and opt_in.environment_allowed
    ):
        backend: ModelBackend = OfflineResumeBackend(
            backend_id=config.model.backend_id,
            provider_id=config.model.provider_id,
        )
        allow_new_dispatch = False
    else:
        backend = _backend(config, allow_live_model=allow_live_model)
    authority = authority_from_manifest(manifest)
    prepared = prepare_phase4(
        repository_root=repository_root,
        config=config,
        authority=authority,
        purpose="phase4-recorded-resume",
    )
    return Phase4RunEngine(
        repository_root=repository_root,
        run_directory=run_directory,
        config=config,
        prepared=prepared,
        backend=backend,
        allow_new_dispatch=allow_new_dispatch,
    ).execute(interrupt_after=interrupt_after)
