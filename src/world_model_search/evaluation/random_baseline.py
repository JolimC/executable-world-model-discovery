"""No-model target-blind random DSL baseline with separate local cost accounting."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns

from world_model_search.config import AppConfig
from world_model_search.domain.types import ProposalBudget, ProposalContext, SplitLabel
from world_model_search.dsl.ast import AstLimits
from world_model_search.errors import ConfigurationError
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.proposer.random_dsl import RANDOM_DSL_VERSION, RandomDslProposer
from world_model_search.serialization import JsonObject
from world_model_search.tasks import HiddenTaskStore, benchmark_root_for_config, load_public_task


def run_random_baseline(
    *, repository_root: Path, config: AppConfig, candidate_count: int
) -> JsonObject:
    if config.schema_version != 4 or config.dsl is None:
        raise ConfigurationError("random DSL baseline requires a Phase 4 typed configuration")
    if config.run.split not in {SplitLabel.TRAINING, SplitLabel.DEVELOPMENT}:
        raise ConfigurationError("random DSL baseline is restricted to training/development")
    if not 1 <= candidate_count <= 100_000:
        raise ConfigurationError("random DSL candidate count must be in [1, 100000]")
    root = benchmark_root_for_config(repository_root, config)
    task = load_public_task(root, config.run.task_id)
    hidden = HiddenTaskStore(root).load(
        task.task_id,
        allowed_splits=frozenset({SplitLabel.TRAINING, SplitLabel.DEVELOPMENT}),
        purpose="phase4-target-blind-random-baseline",
    )
    limits = AstLimits(config.dsl.max_depth, config.dsl.max_nodes, config.dsl.max_cases)
    context = ProposalContext(task.public_view())
    construction_start = perf_counter_ns()
    documents = RandomDslProposer().propose(
        context,
        ProposalBudget(
            max_candidates=candidate_count,
            start_index=0,
            proposer_seed=config.run.seed,
        ),
    )
    construction_ns = max(0, perf_counter_ns() - construction_start)
    oracle = ExactDslOracle(hidden, limits=limits, response_mode=config.oracle.response_mode)
    evaluation_start = perf_counter_ns()
    results = tuple(oracle.evaluate(document.ast).result for document in documents)
    evaluation_ns = max(0, perf_counter_ns() - evaluation_start)
    exact_bits = [result.ast_bits for result in results if result.exact]
    return {
        "baseline_schema_version": 1,
        "baseline": RANDOM_DSL_VERSION,
        "task_id": task.task_id,
        "split": task.split.value,
        "search_seed": config.run.seed,
        "target_blind_contract": "ignores-task-demonstrations-parents-and-oracle-feedback-v1",
        "candidate_constructions": len(documents),
        "proposal_items": len(documents),
        "oracle_calls": len(results),
        "model_requests": 0,
        "model_tokens": 0,
        "model_nano_usd": 0,
        "exact_candidates": len(exact_bits),
        "best_exact_ast_bits": min(exact_bits) if exact_bits else None,
        "diagnostics": {
            "construction_cpu_elapsed_ns": construction_ns,
            "oracle_cpu_elapsed_ns": evaluation_ns,
        },
        "comparison_note": "contextual baseline; not token-matched and not an H1/H2 condition",
    }
