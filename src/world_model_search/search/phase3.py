"""Phase 3 mutation search lifecycle with exact budgets, resume, and lineage data."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import cast

from world_model_search.config import AppConfig
from world_model_search.domain.types import (
    Candidate,
    CandidateSummary,
    OracleFeedback,
    OracleResponseMode,
    OracleResult,
    ProposalBudget,
    ProposalContext,
    SearchEvent,
    SplitLabel,
    Task,
)
from world_model_search.dsl.ast import AstLimits, At, BitExpr, Const, Majority, Parity
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.interpreter import semantic_hash
from world_model_search.dsl.json_schema import (
    DslCandidateDocument,
    ast_canonical_json,
    ast_to_value,
)
from world_model_search.dsl.versions import (
    PHASE3_EVENT_SCHEMA_VERSION,
    PHASE3_INITIALIZATION_VERSION,
    PHASE3_MANIFEST_SCHEMA_VERSION,
    PHASE3_PROPOSAL_ARTIFACT_VERSION,
    PREFIX_CODE_VERSION,
)
from world_model_search.errors import ConfigurationError, PersistenceError
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.persistence.artifacts import write_content_artifact, write_json_exclusive
from world_model_search.persistence.database import RunDatabase
from world_model_search.persistence.manifest import build_manifest, utc_now
from world_model_search.scheduler.uniform import SchedulerDecision, UniformScheduler
from world_model_search.search.archive import (
    ArchiveDecision,
    InsertionOutcome,
    MapElitesArchive,
    SingleIncumbent,
)
from world_model_search.search.operators import (
    DEFAULT_OPERATOR_INVENTORY,
    AttemptOutcome,
    CounterRng,
    MutationProposer,
    OperatorAttempt,
    OperatorId,
)
from world_model_search.search.phase3_types import (
    BudgetState,
    SearchCondition,
    phase3_candidate,
)
from world_model_search.serialization import (
    JsonObject,
    canonical_json,
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
class Phase3Outcome:
    run_id: str
    status: str
    completed_steps: int
    event_payload_hashes: tuple[str, ...]
    run_directory: Path


@dataclass(frozen=True, slots=True)
class Phase3Authority:
    mode: str
    allowed_splits: frozenset[SplitLabel]
    frozen_task_ids: tuple[str, ...] = ()
    freeze_hash: str | None = None

    @classmethod
    def ordinary(cls) -> Phase3Authority:
        return cls(
            mode="ordinary-training-development-v1",
            allowed_splits=frozenset({SplitLabel.TRAINING, SplitLabel.DEVELOPMENT}),
        )

    @classmethod
    def locked_validation(
        cls, *, frozen_task_ids: tuple[str, ...], freeze_hash: str
    ) -> Phase3Authority:
        if not frozen_task_ids or len(frozen_task_ids) != len(set(frozen_task_ids)):
            raise ConfigurationError("validation authority requires unique frozen task IDs")
        return cls(
            mode="phase3-locked-validation-once-v1",
            allowed_splits=frozenset({SplitLabel.VALIDATION}),
            frozen_task_ids=frozen_task_ids,
            freeze_hash=freeze_hash,
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
class PreparedPhase3:
    task: Task
    hidden: HiddenTaskBundle
    store: HiddenTaskStore
    limits: AstLimits
    authority: Phase3Authority


Mechanism = MapElitesArchive | SingleIncumbent


def initialization_candidates() -> tuple[BitExpr, ...]:
    """Small target-independent baseline shared and charged in both conditions."""

    return (
        Const(0),
        Const(1),
        At(-1),
        At(0),
        At(1),
        Parity((-1, 0, 1)),
        Majority((-1, 0, 1)),
    )


def prepare_phase3(
    *, repository_root: Path, config: AppConfig, authority: Phase3Authority, purpose: str
) -> PreparedPhase3:
    if config.schema_version != 3 or config.dsl is None or config.budget is None:
        raise ConfigurationError("Phase 3 requires a complete schema-3 configuration")
    benchmark_root = benchmark_root_for_config(repository_root, config)
    task = load_public_task(benchmark_root, config.run.task_id)
    if task.split != config.run.split:
        raise ConfigurationError("configured split does not match the public task artifact")
    if task.split not in authority.allowed_splits:
        raise ConfigurationError("task split is not authorized for this Phase 3 operation")
    if authority.mode == "phase3-locked-validation-once-v1" and task.task_id not in set(
        authority.frozen_task_ids
    ):
        raise ConfigurationError("validation task is absent from the frozen authority list")
    if config.budget.oracle_call_cap < len(initialization_candidates()):
        raise ConfigurationError("oracle budget cannot charge the complete shared initialization")
    store = HiddenTaskStore(benchmark_root)
    hidden = store.load(task.task_id, allowed_splits=authority.allowed_splits, purpose=purpose)
    limits = AstLimits(
        max_depth=config.dsl.max_depth,
        max_nodes=config.dsl.max_nodes,
        max_cases=config.dsl.max_cases,
    )
    return PreparedPhase3(task=task, hidden=hidden, store=store, limits=limits, authority=authority)


def _mechanism(config: AppConfig, task: Task) -> Mechanism:
    if config.run.condition_id == SearchCondition.DIVERSE.value:
        if config.archive is None:
            raise ConfigurationError("diverse condition has no archive settings")
        return MapElitesArchive(task.public_view(), reserve_size=config.archive.reserve_size)
    if config.run.condition_id == SearchCondition.INCUMBENT.value:
        return SingleIncumbent(task.public_view())
    raise ConfigurationError("unknown Phase 3 condition")


def _result_from_json(data: str) -> OracleResult:
    try:
        raw: object = json.loads(data)
    except json.JSONDecodeError as exc:
        raise PersistenceError("recorded oracle result is invalid JSON") from exc
    if not isinstance(raw, dict):
        raise PersistenceError("recorded oracle result is not an object")
    response = raw.get("response")
    if not isinstance(response, dict):
        raise PersistenceError("recorded oracle feedback is malformed")
    try:
        mode = OracleResponseMode(response["mode"])
        summary_raw = response["summary"]
        if not isinstance(summary_raw, list) or not all(
            isinstance(item, str) for item in summary_raw
        ):
            raise ValueError
        feedback = OracleFeedback(
            mode=mode,
            summary=tuple(summary_raw),
            counterexample=response.get("counterexample")
            if isinstance(response.get("counterexample"), str)
            else None,
        )
        return OracleResult(
            type_valid=bool(raw["type_valid"]),
            total=bool(raw["total"]),
            local_errors=int(raw["local_errors"]),
            local_cases=int(raw["local_cases"]),
            rollout_pass=bool(raw["rollout_pass"]),
            exact=bool(raw["exact"]),
            ast_bits=int(raw["ast_bits"]),
            residual_bits=int(raw["residual_bits"]),
            runtime_ns=int(raw["runtime_ns"]),
            response=feedback,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistenceError("recorded oracle result fields are malformed") from exc


def _candidate_from_row(row: sqlite3.Row, limits: AstLimits) -> Candidate:
    mapping = dict(row)
    try:
        ast_value: object = json.loads(mapping["canonical_ast_json"])
        document_value = {
            "candidate_schema_version": 1,
            "dsl_version": "binary-ca-radius1-dsl-v1",
            "ast": ast_value,
        }
        ast = DslCandidateDocument.from_json(canonical_json(document_value), limits=limits).ast
        parent_raw: object = json.loads(mapping["parent_ids_json"])
        if not isinstance(parent_raw, list) or not all(
            isinstance(item, str) for item in parent_raw
        ):
            raise ValueError
        return Candidate(
            candidate_id=str(mapping["candidate_id"]),
            task_id=str(mapping["task_id"]),
            ast=ast,
            parent_ids=tuple(parent_raw),
            proposer_id=str(mapping["proposer_id"]),
            operator_id=str(mapping["operator_id"]),
            context_hash=str(mapping["context_hash"]),
            payload_hash=str(mapping["payload_hash"]),
            semantic_hash=str(mapping["semantic_hash"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistenceError("candidate row fields are malformed") from exc


def restore_mechanism(
    database: RunDatabase, mechanism: Mechanism, limits: AstLimits
) -> tuple[set[str], set[str]]:
    candidates = {
        row["candidate_id"]: _candidate_from_row(row, limits)
        for row in database.candidate_records()
    }
    canonical_seen: set[str] = set()
    semantic_seen: set[str] = set()
    transition_by_attempt = {
        row["attempt_index"]: row for row in database.phase3_transition_records()
    }
    for evaluation in database.evaluation_records():
        candidate = candidates[evaluation["candidate_id"]]
        result = _result_from_json(evaluation["result_json"])
        decision = mechanism.insert(candidate, result)
        recorded = transition_by_attempt.get(evaluation["attempt_index"])
        if recorded is None or canonical_json(decision.to_value()) != recorded["decision_json"]:
            raise PersistenceError("recorded archive/incumbent transition diverged during resume")
        if not isinstance(candidate.ast, BitExpr) or candidate.semantic_hash is None:
            raise PersistenceError("recorded Phase 3 candidate is untyped")
        canonical_seen.add(ast_canonical_json(candidate.ast))
        semantic_seen.add(candidate.semantic_hash)
    return canonical_seen, semantic_seen


def _parent_candidates(
    mechanism: Mechanism,
    decision: SchedulerDecision,
    *,
    seed: int,
    attempt_index: int,
    operator_id: OperatorId,
) -> tuple[CandidateSummary, ...]:
    if isinstance(mechanism, MapElitesArchive):
        coordinate = mechanism.coordinate_for_branch(decision.selected_branch_id)
        primary_pool = mechanism.candidate_summaries(coordinate=coordinate)
        all_pool = mechanism.candidate_summaries()
    else:
        primary_pool = mechanism.candidate_summaries()
        all_pool = primary_pool
    if not primary_pool:
        raise PersistenceError("selected search branch has no parents")
    parent_rng = CounterRng(seed, "parent-choice", attempt_index)
    first = primary_pool[parent_rng.integer("primary", len(primary_pool))]
    if operator_id is not OperatorId.CROSSOVER:
        return (first,)
    # Explicit unavoidable difference: archive can select another stored lineage; the incumbent
    # deterministically uses ordered self-crossover when it has no second candidate.
    second = all_pool[parent_rng.integer("secondary", len(all_pool))] if all_pool else first
    return (first, second)


def _public_context_value(context: ProposalContext) -> JsonObject:
    task_value = json.loads(canonical_json(context.task))
    if not isinstance(task_value, dict):
        raise AssertionError("public task did not serialize as an object")
    return {
        "task": task_value,
        "parents": [
            {
                "candidate_id": parent.candidate_id,
                "ast": ast_to_value(parent.ast)
                if isinstance(parent.ast, BitExpr)
                else {"phase0_opaque": True},
            }
            for parent in context.parents
        ],
        "feedback": json.loads(canonical_json(context.feedback)),
    }


def _scheduler(
    mechanism: Mechanism,
    task_id: str,
    budget: BudgetState,
    *,
    seed: int,
    attempt_index: int,
    initializing: bool,
) -> SchedulerDecision:
    branch_ids = (f"{task_id}:initialization",) if initializing else mechanism.branch_ids()
    return UniformScheduler().select(
        branch_ids,
        master_seed=seed,
        selection_counter=attempt_index,
        remaining_proposal_attempts=budget.remaining_proposal_attempts,
        remaining_oracle_calls=budget.remaining_oracle_calls,
    )


def _operator_value(attempt: OperatorAttempt | None, *, baseline_index: int | None) -> JsonObject:
    if attempt is None:
        return {
            "operator_id": "public-baseline-initialization",
            "operator_version": PHASE3_INITIALIZATION_VERSION,
            "outcome": AttemptOutcome.EMITTED.value,
            "selected_paths": [],
            "choices": {"baseline_index": baseline_index},
            "rejection_reason": None,
            "crossover_arity": 0,
        }
    return {
        "operator_id": attempt.operator_id.value,
        "operator_version": attempt.operator_version,
        "outcome": attempt.outcome.value,
        "selected_paths": [list(path) for path in attempt.selected_paths],
        "choices": attempt.choices,
        "rejection_reason": attempt.rejection_reason,
        "crossover_arity": attempt.crossover_arity,
    }


def _budget_after_evaluation(
    budget: BudgetState,
    *,
    operator_attempt: bool,
    canonical_duplicate: bool,
    semantic_duplicate: bool,
    decision: ArchiveDecision,
) -> BudgetState:
    increments = {
        "proposal_attempts": 1,
        "operator_attempts": int(operator_attempt),
        "parsed_proposals": 1,
        "type_valid_proposals": 1,
        "canonical_proposals": 1,
        "canonical_duplicates": int(canonical_duplicate),
        "semantic_duplicates": int(semantic_duplicate),
        "semantically_distinct_proposals": int(not semantic_duplicate),
        "oracle_invocations": 1,
        "scheduler_selections": 1,
        "evaluated_candidates": 1,
        "archive_insertions": int(decision.outcome is InsertionOutcome.INSERTED),
        "archive_replacements": int(decision.outcome is InsertionOutcome.REPLACED),
        "archive_reserves": int(decision.outcome is InsertionOutcome.RESERVED),
        "archive_duplicates": int(decision.outcome is InsertionOutcome.DUPLICATE),
        "archive_rejections": int(decision.outcome is InsertionOutcome.REJECTED),
    }
    return budget.updated(**increments)


def _attempt_event_payload(
    *,
    attempt_index: int,
    task_id: str,
    scheduler: SchedulerDecision,
    operator: JsonObject,
    artifact_hash: str,
    artifact_name: str,
    candidate: Candidate | None,
    canonical_duplicate: bool,
    semantic_duplicate: bool,
    result: OracleResult | None,
    decision: ArchiveDecision | None,
    budget: BudgetState,
) -> JsonObject:
    candidate_value: JsonObject | None = None
    if candidate is not None:
        candidate_value = {
            "candidate_id": candidate.candidate_id,
            "payload_hash": candidate.payload_hash,
            "ordered_parent_ids": list(candidate.parent_ids),
            "proposer_id": candidate.proposer_id,
            "operator_id": candidate.operator_id,
            "context_hash": candidate.context_hash,
            "semantic_hash": candidate.semantic_hash,
            "ast_bits": encoded_length(candidate.ast)
            if isinstance(candidate.ast, BitExpr)
            else None,
            "coding_version": PREFIX_CODE_VERSION,
            "canonical_duplicate": canonical_duplicate,
            "semantic_duplicate": semantic_duplicate,
        }
    return {
        "schema_version": PHASE3_EVENT_SCHEMA_VERSION,
        "attempt_index": attempt_index,
        "task_id": task_id,
        "scheduler": scheduler.to_value(),
        "proposal": {
            "artifact_name": artifact_name,
            "artifact_hash": artifact_hash,
            "operator": operator,
        },
        "candidate": candidate_value,
        "oracle_result": result.deterministic_payload() if result is not None else None,
        "archive_decision": decision.to_value() if decision is not None else None,
        "budget_state": budget.to_value(),
    }


class Phase3RunEngine:
    def __init__(
        self,
        *,
        run_directory: Path,
        config: AppConfig,
        prepared: PreparedPhase3,
    ) -> None:
        if config.budget is None:
            raise ConfigurationError("Phase 3 budget settings are missing")
        self.run_directory = run_directory
        self.config = config
        self.prepared = prepared
        self.proposer = MutationProposer()
        self.oracle = ExactDslOracle(
            prepared.hidden,
            limits=prepared.limits,
            response_mode=config.oracle.response_mode,
        )

    def execute(self, *, interrupt_after: int | None = None) -> Phase3Outcome:
        if interrupt_after is not None and interrupt_after < 1:
            raise ConfigurationError("interrupt_after must be >= 1")
        mechanism = _mechanism(self.config, self.prepared.task)
        with RunDatabase(self.run_directory / "run.sqlite3") as database:
            state = database.state()
            if state.status == "completed":
                return self._outcome(state.run_id, state.status, database.events())
            canonical_seen, semantic_seen = restore_mechanism(
                database, mechanism, self.prepared.limits
            )
            budget = database.phase3_budget()
            if budget.proposal_attempts != state.next_step:
                raise PersistenceError("resume step and charged proposal attempts diverged")
            database.set_status("running", utc_now())
            try:
                while not budget.exhausted:
                    attempt_cpu_started = process_time_ns()
                    attempt_elapsed_started = perf_counter_ns()
                    oracle_cpu_ns = 0
                    oracle_elapsed_ns = 0
                    attempt_index = budget.proposal_attempts
                    initial = attempt_index < len(initialization_candidates())
                    scheduler = _scheduler(
                        mechanism,
                        self.prepared.task.task_id,
                        budget,
                        seed=self.config.run.seed,
                        attempt_index=attempt_index,
                        initializing=initial,
                    )
                    operator_attempt: OperatorAttempt | None = None
                    parents: tuple[CandidateSummary, ...] = ()
                    context: ProposalContext
                    source: BitExpr | None
                    canonical: BitExpr | None
                    if initial:
                        source = initialization_candidates()[attempt_index]
                        canonical = source
                        operator_id = "public-baseline-initialization"
                        context = ProposalContext(
                            task=self.prepared.task.public_view(), parents=(), feedback=()
                        )
                    else:
                        choice_rng = CounterRng(
                            self.config.run.seed, "operator-choice", attempt_index
                        )
                        inventory_index = choice_rng.weighted_index(
                            "inventory", tuple(item.weight for item in DEFAULT_OPERATOR_INVENTORY)
                        )
                        selected_operator = DEFAULT_OPERATOR_INVENTORY[inventory_index].operator_id
                        parents = _parent_candidates(
                            mechanism,
                            scheduler,
                            seed=self.config.run.seed,
                            attempt_index=attempt_index,
                            operator_id=selected_operator,
                        )
                        typed_parents = tuple(
                            parent.ast for parent in parents if isinstance(parent.ast, BitExpr)
                        )
                        if len(typed_parents) != len(parents):
                            raise PersistenceError("parent context contains a non-DSL candidate")
                        context = ProposalContext(
                            task=self.prepared.task.public_view(), parents=parents, feedback=()
                        )
                        proposals = self.proposer.propose(
                            context,
                            ProposalBudget(
                                max_candidates=1,
                                start_index=attempt_index,
                                proposer_seed=self.config.run.seed,
                                operator_id=selected_operator.value,
                            ),
                        )
                        if len(proposals) != 1:
                            raise PersistenceError(
                                "the deterministic mutation proposer must return one attempt"
                            )
                        operator_attempt = proposals[0]
                        source = operator_attempt.source_ast
                        canonical = operator_attempt.canonical_ast
                        operator_id = selected_operator.value
                    operator_value = _operator_value(
                        operator_attempt, baseline_index=attempt_index if initial else None
                    )
                    candidate: Candidate | None = None
                    result: OracleResult | None = None
                    decision: ArchiveDecision | None = None
                    canonical_duplicate = False
                    semantic_duplicate = False
                    if (
                        canonical is not None
                        and source is not None
                        and (
                            operator_attempt is None
                            or operator_attempt.outcome is AttemptOutcome.EMITTED
                        )
                    ):
                        document = DslCandidateDocument(ast=canonical)
                        document_json = document.to_json()
                        payload_hash = sha256_text(document_json)
                        candidate_semantic = semantic_hash(canonical, limits=self.prepared.limits)
                        canonical_key = ast_canonical_json(canonical)
                        canonical_duplicate = canonical_key in canonical_seen
                        semantic_duplicate = candidate_semantic in semantic_seen
                        candidate = phase3_candidate(
                            task_id=self.prepared.task.task_id,
                            ast=canonical,
                            parent_ids=tuple(parent.candidate_id for parent in parents),
                            proposer_id="mutation",
                            operator_id=operator_id,
                            context_hash=context.content_hash,
                            payload_hash=payload_hash,
                            coding_version=PREFIX_CODE_VERSION,
                            semantic_hash=candidate_semantic,
                        )
                        oracle_cpu_started = process_time_ns()
                        evaluated = self.oracle.evaluate(canonical)
                        oracle_cpu_ns = max(0, process_time_ns() - oracle_cpu_started)
                        result = evaluated.result
                        oracle_elapsed_ns = result.runtime_ns
                        decision = mechanism.insert(candidate, result)
                        budget = _budget_after_evaluation(
                            budget,
                            operator_attempt=operator_attempt is not None,
                            canonical_duplicate=canonical_duplicate,
                            semantic_duplicate=semantic_duplicate,
                            decision=decision,
                        )
                        canonical_seen.add(canonical_key)
                        semantic_seen.add(candidate_semantic)
                        candidate_document_value: JsonObject | None = document.to_value()
                    else:
                        outcome = (
                            operator_attempt.outcome
                            if operator_attempt is not None
                            else AttemptOutcome.REJECTED
                        )
                        budget = budget.updated(
                            proposal_attempts=1,
                            operator_attempts=int(operator_attempt is not None),
                            invalid_outputs=int(outcome is AttemptOutcome.REJECTED),
                            noop_outputs=int(outcome is AttemptOutcome.NO_OP),
                            scheduler_selections=1,
                        )
                        candidate_document_value = None
                    artifact: JsonObject = {
                        "artifact_version": PHASE3_PROPOSAL_ARTIFACT_VERSION,
                        "attempt_index": attempt_index,
                        "task_id": self.prepared.task.task_id,
                        "public_context": _public_context_value(context),
                        "public_context_hash": context.content_hash,
                        "ordered_parent_ids": [parent.candidate_id for parent in parents],
                        "operator": operator_value,
                        "candidate_document": candidate_document_value,
                        "submitted_source_ast": ast_to_value(source)
                        if source is not None
                        else None,
                    }
                    artifact_text = canonical_json(artifact)
                    artifact_name = f"proposals/attempt-{attempt_index:05d}.json"
                    artifact_hash = write_content_artifact(
                        self.run_directory / artifact_name, artifact_text
                    )
                    event = SearchEvent.create(
                        sequence=attempt_index,
                        event_type="phase3_proposal_attempt",
                        logical_cost=budget.oracle_invocations,
                        payload=_attempt_event_payload(
                            attempt_index=attempt_index,
                            task_id=self.prepared.task.task_id,
                            scheduler=scheduler,
                            operator=operator_value,
                            artifact_hash=artifact_hash,
                            artifact_name=artifact_name,
                            candidate=candidate,
                            canonical_duplicate=canonical_duplicate,
                            semantic_duplicate=semantic_duplicate,
                            result=result,
                            decision=decision,
                            budget=budget,
                        ),
                        audit_timestamp=utc_now(),
                    )
                    attempt_cpu_ns = max(0, process_time_ns() - attempt_cpu_started)
                    attempt_elapsed_ns = max(0, perf_counter_ns() - attempt_elapsed_started)
                    database.append_phase3_step(
                        attempt_index=attempt_index,
                        candidate=candidate,
                        source_ast=source,
                        artifact_name=artifact_name,
                        artifact_hash=artifact_hash,
                        scheduler=scheduler,
                        operator_json=operator_value,
                        attempt_outcome=str(operator_value["outcome"]),
                        canonical_duplicate=canonical_duplicate,
                        semantic_duplicate=semantic_duplicate,
                        result=result,
                        oracle_version=self.config.oracle.oracle_id,
                        decision=decision,
                        budget=budget,
                        event=event,
                        next_step=attempt_index + 1,
                        attempt_cpu_ns=attempt_cpu_ns,
                        oracle_cpu_ns=oracle_cpu_ns,
                        attempt_elapsed_ns=attempt_elapsed_ns,
                        oracle_elapsed_ns=oracle_elapsed_ns,
                    )
                    if interrupt_after is not None and budget.proposal_attempts >= interrupt_after:
                        database.set_status("interrupted", utc_now())
                        return self._outcome(state.run_id, "interrupted", database.events())
            except KeyboardInterrupt:
                database.set_status("interrupted", utc_now())
                raise
            results = phase3_results(
                database.events(), budget=budget, mechanism=mechanism, config=self.config
            )
            analysis_hash = write_phase3_analysis(
                run_directory=self.run_directory,
                database=database,
                results=results,
                accesses=self.prepared.store.accesses,
                config=self.config,
            )
            results["analysis_manifest_hash"] = analysis_hash
            results["deterministic_summary_hash"] = sha256_json(results)
            write_content_artifact(self.run_directory / "results.json", canonical_json(results))
            database.set_status("completed", utc_now())
            return self._outcome(state.run_id, "completed", database.events())

    def _outcome(self, run_id: str, status: str, events: tuple[SearchEvent, ...]) -> Phase3Outcome:
        return Phase3Outcome(
            run_id=run_id,
            status=status,
            completed_steps=len(events),
            event_payload_hashes=tuple(event.payload_hash for event in events),
            run_directory=self.run_directory,
        )


def phase3_results(
    events: tuple[SearchEvent, ...],
    *,
    budget: BudgetState,
    mechanism: Mechanism,
    config: AppConfig,
) -> JsonObject:
    exact_calls: list[int] = []
    exact_bits: list[int] = []
    two_part: list[int] = []
    for event in events:
        payload = event.payload
        result = payload.get("oracle_result")
        state = payload.get("budget_state")
        if not isinstance(result, dict) or not isinstance(state, dict):
            continue
        counters = state.get("counters")
        if not isinstance(counters, dict):
            raise PersistenceError("event budget counters are malformed")
        call = counters.get("oracle_invocations")
        ast_bits = result.get("ast_bits")
        residual_bits = result.get("residual_bits")
        if (
            not isinstance(call, int)
            or not isinstance(ast_bits, int)
            or not isinstance(residual_bits, int)
        ):
            raise PersistenceError("event metric fields are malformed")
        two_part.append(ast_bits + residual_bits)
        if result.get("exact") is True:
            exact_calls.append(call)
            exact_bits.append(ast_bits)
    first_exact = min(exact_calls) if exact_calls else None
    auc_numerator = budget.oracle_call_cap - first_exact + 1 if first_exact is not None else 0
    transition_counts: Counter[str] = Counter()
    for event in events:
        raw_decision = event.payload.get("archive_decision")
        if isinstance(raw_decision, dict):
            transition_counts[str(raw_decision.get("outcome"))] += 1
    metrics: JsonObject = {
        "normalized_exact_auc": auc_numerator / budget.oracle_call_cap,
        "exact_auc_numerator": auc_numerator,
        "exact_auc_denominator": budget.oracle_call_cap,
        "final_exact_solved": bool(exact_calls),
        "calls_to_first_exact": first_exact,
        "best_exact_ast_bits": min(exact_bits) if exact_bits else None,
        "best_two_part_bits": min(two_part) if two_part else None,
        "archive_coverage": len(mechanism.cells) if isinstance(mechanism, MapElitesArchive) else 0,
        "distinct_candidate_semantics": budget.semantically_distinct_proposals,
        "valid_proposal_rate": (
            budget.parsed_proposals / budget.proposal_attempts if budget.proposal_attempts else 0.0
        ),
        "semantic_duplicate_rate": (
            budget.semantic_duplicates / budget.parsed_proposals if budget.parsed_proposals else 0.0
        ),
        "transition_outcomes": dict(sorted(transition_counts.items())),
        "proposal_budget_utilization": budget.proposal_attempts / budget.proposal_attempt_cap,
        "oracle_budget_utilization": budget.oracle_invocations / budget.oracle_call_cap,
    }
    return {
        "schema_version": 3,
        "status": "completed",
        "condition_id": config.run.condition_id,
        "task_id": config.run.task_id,
        "search_seed": config.run.seed,
        "metrics": metrics,
        "budget": budget.to_value(),
        "event_payload_hashes": [event.payload_hash for event in events],
    }


def write_phase3_analysis(
    *,
    run_directory: Path,
    database: RunDatabase,
    results: JsonObject,
    accesses: tuple[OracleTaskAccess, ...],
    config: AppConfig,
) -> str:
    """Derive deterministic individual-run artifacts only from committed records."""

    analysis = run_directory / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    events = database.events()
    candidates = database.candidate_records()
    edges = database.phase3_lineage_records()
    transitions = database.phase3_transition_records()
    nodes = [
        {
            "candidate_id": row["candidate_id"],
            "task_id": row["task_id"],
            "operator_id": row["operator_id"],
            "ordered_parent_ids": json.loads(row["parent_ids_json"]),
            "ast": json.loads(row["canonical_ast_json"]),
            "ast_bits": row["ast_bits"],
            "first_attempt_index": row["first_attempt_index"],
        }
        for row in candidates
    ]
    lineage = cast(
        JsonObject,
        {
            "lineage_version": "phase3-lineage-dag-v1",
            "selection_rule": "best-exact-else-final-best-then-candidate-id-v1",
            "nodes": nodes,
            "edges": [dict(row) for row in edges],
            "transitions": [json.loads(row["decision_json"]) for row in transitions],
        },
    )
    files: dict[str, str] = {}

    def record(name: str, content: str) -> None:
        normalized = content.rstrip("\n")
        write_content_artifact(analysis / name, normalized)
        files[name] = sha256_text(normalized)

    record("lineage.json", canonical_json(lineage))
    dot_lines = ["digraph phase3_lineage {", "  rankdir=LR;"]
    for node in nodes:
        label = f"{str(node['candidate_id'])[:8]}\\n{node['operator_id']}"
        dot_lines.append(f'  "{node["candidate_id"]}" [label="{label}"];')
    for edge in edges:
        dot_lines.append(
            f'  "{edge["parent_candidate_id"]}" -> "{edge["child_candidate_id"]}" '
            f'[label="{edge["parent_order"]}"];'
        )
    dot_lines.append("}")
    record("lineage.dot", "\n".join(dot_lines) + "\n")
    curve_lines = ["oracle_calls,best_exact"]
    coverage_lines = ["oracle_calls,archive_coverage"]
    solved = 0
    coordinates: set[str] = set()
    for event in events:
        payload = event.payload
        result = payload.get("oracle_result")
        state = payload.get("budget_state")
        decision = payload.get("archive_decision")
        if not isinstance(result, dict) or not isinstance(state, dict):
            continue
        counters = state.get("counters")
        if not isinstance(counters, dict) or not isinstance(
            counters.get("oracle_invocations"), int
        ):
            raise PersistenceError("analysis event budget is malformed")
        calls = counters["oracle_invocations"]
        solved = max(solved, int(result.get("exact") is True))
        if isinstance(decision, dict) and decision.get("role") in {"elite", "reserve"}:
            coordinate = decision.get("coordinate")
            if isinstance(coordinate, dict):
                coordinates.add(canonical_json(coordinate))
        curve_lines.append(f"{calls},{solved}")
        coverage_lines.append(f"{calls},{len(coordinates)}")
    record("exact-curve.csv", "\n".join(curve_lines) + "\n")
    record("archive-coverage.csv", "\n".join(coverage_lines) + "\n")
    attempts = database.phase3_attempt_records()
    operator_counts = Counter(json.loads(row["operator_json"])["operator_id"] for row in attempts)
    record(
        "operator-diagnostics.json",
        canonical_json(
            {
                "operator_attempts": dict(sorted(operator_counts.items())),
                "attempt_outcomes": dict(
                    sorted(Counter(row["outcome"] for row in attempts).items())
                ),
                "canonical_duplicates": sum(row["canonical_duplicate"] for row in attempts),
                "semantic_duplicates": sum(row["semantic_duplicate"] for row in attempts),
            }
        ),
    )
    record("budget-reconciliation.json", canonical_json(results["budget"]))
    record(
        "access-ledger.json",
        canonical_json(
            {
                "authority_mode": (
                    "phase3-locked-validation-once-v1"
                    if config.run.split is SplitLabel.VALIDATION
                    else "ordinary-training-development-v1"
                ),
                "accesses": [
                    {"task_id": item.task_id, "split": item.split.value, "purpose": item.purpose}
                    for item in accesses
                ],
                "validation_consumed": config.run.split is SplitLabel.VALIDATION,
                "test_oracle_accesses": sum(item.split is SplitLabel.TEST for item in accesses),
            }
        ),
    )
    timing_rows = database.phase3_diagnostic_records()
    evaluation_rows = database.evaluation_records()
    attempt_cpu_ns = sum(int(row["attempt_cpu_ns"]) for row in timing_rows)
    oracle_cpu_ns = sum(int(row["oracle_cpu_ns"]) for row in timing_rows)
    attempt_elapsed_ns = sum(int(row["attempt_elapsed_ns"]) for row in timing_rows)
    recorded_oracle_elapsed_ns = sum(int(row["runtime_ns"]) for row in evaluation_rows)
    runtime_diagnostics: JsonObject = {
        "diagnostic_version": "phase3-runtime-diagnostics-v1",
        "deterministic_replay_input": False,
        "timed_attempts": len(timing_rows),
        "evaluated_candidates": len(evaluation_rows),
        "attempt_cpu_ns": attempt_cpu_ns if timing_rows else None,
        "oracle_cpu_ns": oracle_cpu_ns if timing_rows else None,
        "non_oracle_cpu_ns": (max(0, attempt_cpu_ns - oracle_cpu_ns) if timing_rows else None),
        "attempt_elapsed_ns": attempt_elapsed_ns if timing_rows else None,
        "oracle_elapsed_ns": recorded_oracle_elapsed_ns,
        "language_model_calls": 0,
        "language_model_tokens": 0,
        "timing_scope": (
            "per-attempt-process-and-monotonic-clocks-v1"
            if timing_rows
            else "legacy-oracle-wall-clock-only-v1"
        ),
    }
    write_content_artifact(
        analysis / "runtime-diagnostics.json", canonical_json(runtime_diagnostics)
    )
    manifest = cast(
        JsonObject,
        {
            "analysis_artifact_version": "phase3-individual-analysis-v1",
            "source": "committed-phase3-records-only",
            "files": files,
            "non_deterministic_diagnostic_files": ["runtime-diagnostics.json"],
        },
    )
    manifest_text = canonical_json(manifest)
    write_content_artifact(analysis / "manifest.json", manifest_text)
    return sha256_text(manifest_text)


def start_phase3_run(
    *,
    repository_root: Path,
    config: AppConfig,
    config_source: str,
    run_id: str,
    interrupt_after: int | None,
    authority: Phase3Authority | None = None,
) -> Phase3Outcome:
    from world_model_search.search.loop import validate_run_id

    selected_authority = authority or Phase3Authority.ordinary()
    selected_id = validate_run_id(run_id)
    run_directory = repository_root / config.run.root / selected_id
    if run_directory.exists():
        raise PersistenceError(f"run already exists: {selected_id}")
    prepared = prepare_phase3(
        repository_root=repository_root,
        config=config,
        authority=selected_authority,
        purpose=(
            "phase3-locked-validation-gate"
            if selected_authority.mode == "phase3-locked-validation-once-v1"
            else "phase3-recorded-mutation-run"
        ),
    )
    manifest = build_manifest(
        repository_root=repository_root,
        run_id=selected_id,
        config=config,
        config_source=config_source,
        task=prepared.task,
    )
    manifest["phase3_authority"] = selected_authority.to_value()
    manifest["phase3_authority_hash"] = sha256_json(selected_authority.to_value())
    run_directory.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(run_directory / "manifest.json", manifest)
    if config.budget is None:
        raise AssertionError("validated Phase 3 config has no budget")
    with RunDatabase(run_directory / "run.sqlite3") as database:
        database.initialize_phase3(
            selected_id,
            prepared.task,
            utc_now(),
            proposal_attempt_cap=config.budget.proposal_attempt_cap,
            oracle_call_cap=config.budget.oracle_call_cap,
        )
    return Phase3RunEngine(run_directory=run_directory, config=config, prepared=prepared).execute(
        interrupt_after=interrupt_after
    )


def authority_from_manifest(manifest: JsonObject) -> Phase3Authority:
    raw = manifest.get("phase3_authority")
    if not isinstance(raw, dict):
        raise PersistenceError("Phase 3 manifest has no authority record")
    mode = raw.get("mode")
    splits = raw.get("allowed_splits")
    tasks = raw.get("frozen_task_ids")
    freeze_hash = raw.get("freeze_hash")
    if (
        not isinstance(mode, str)
        or not isinstance(splits, list)
        or not all(isinstance(item, str) for item in splits)
        or not isinstance(tasks, list)
        or not all(isinstance(item, str) for item in tasks)
    ):
        raise PersistenceError("Phase 3 authority record is malformed")
    try:
        split_values = cast(list[str], splits)
        task_values = cast(list[str], tasks)
        authority = Phase3Authority(
            mode=mode,
            allowed_splits=frozenset(SplitLabel(item) for item in split_values),
            frozen_task_ids=tuple(task_values),
            freeze_hash=freeze_hash if isinstance(freeze_hash, str) else None,
        )
    except ValueError as exc:
        raise PersistenceError("Phase 3 authority split is invalid") from exc
    if sha256_json(authority.to_value()) != manifest.get("phase3_authority_hash"):
        raise PersistenceError("Phase 3 authority hash mismatch")
    return authority


def resume_phase3_run(
    *,
    repository_root: Path,
    run_directory: Path,
    run_id: str,
    config: AppConfig,
    manifest: JsonObject,
    interrupt_after: int | None,
) -> Phase3Outcome:
    if manifest.get("manifest_schema_version") != PHASE3_MANIFEST_SCHEMA_VERSION:
        raise PersistenceError("Phase 3 configuration requires run manifest schema 4")
    authority = authority_from_manifest(manifest)
    prepared = prepare_phase3(
        repository_root=repository_root,
        config=config,
        authority=authority,
        purpose=(
            "phase3-locked-validation-resume"
            if authority.mode == "phase3-locked-validation-once-v1"
            else "phase3-recorded-mutation-run"
        ),
    )
    return Phase3RunEngine(run_directory=run_directory, config=config, prepared=prepared).execute(
        interrupt_after=interrupt_after
    )
