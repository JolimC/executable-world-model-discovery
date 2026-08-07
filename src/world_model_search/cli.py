"""The `wms` Phase 0 command-line surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from world_model_search.config import load_config
from world_model_search.domain.types import SplitLabel
from world_model_search.dsl.ast import AstLimits
from world_model_search.dsl.json_schema import CandidateJsonError, DslCandidateDocument
from world_model_search.errors import (
    CandidateValidationError,
    ConfigurationError,
    PhaseUnavailableError,
    WorldModelSearchError,
)
from world_model_search.evaluation.report import create_recorded_report
from world_model_search.logging import configure_logging
from world_model_search.oracle.exact import ExactDslOracle
from world_model_search.replay import replay_run
from world_model_search.search.loop import resume_run, start_run
from world_model_search.serialization import canonical_json
from world_model_search.tasks import (
    HiddenTaskStore,
    benchmark_root_for_config,
    generate_benchmark,
    generate_phase2_benchmark,
    load_public_task,
)

DEFAULT_RUNS_ROOT = Path("artifacts/runs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wms", description="Executable world-model discovery")
    commands = parser.add_subparsers(dest="command", required=True)

    tasks = commands.add_parser("tasks", help="task artifact commands")
    task_commands = tasks.add_subparsers(dest="task_command", required=True)
    task_generate = task_commands.add_parser("generate", help="generate tasks (Phase 1)")
    task_generate.add_argument("--config", type=Path, required=True)

    oracle = commands.add_parser("oracle", help="oracle commands")
    oracle_commands = oracle.add_subparsers(dest="oracle_command", required=True)
    oracle_verify = oracle_commands.add_parser("verify", help="verify a typed Phase 2 candidate")
    oracle_verify.add_argument("--task", required=True)
    oracle_verify.add_argument("--candidate", type=Path, required=True)
    oracle_verify.add_argument("--config", type=Path, default=Path("configs/phase2-smoke.yaml"))

    solve = commands.add_parser("solve", help="start or resume a run")
    source = solve.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--resume", metavar="RUN_ID")
    solve.add_argument("--proposer")
    solve.add_argument("--run-id")
    solve.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    solve.add_argument(
        "--interrupt-after",
        type=int,
        help="deliberately stop after this total number of evaluations",
    )

    benchmark = commands.add_parser("benchmark", help="run an experiment registry (later phase)")
    benchmark.add_argument("--experiment", type=Path, required=True)

    replay = commands.add_parser("replay", help="replay a completed run from recorded proposals")
    replay.add_argument("--run", required=True)
    replay.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)

    report = commands.add_parser("report", help="report from frozen run data")
    report.add_argument("--run", required=True)
    report.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    report.add_argument("--out", type=Path, required=True)
    return parser


def _unavailable(message: str) -> NoReturn:
    raise PhaseUnavailableError(message)


def _dispatch(arguments: argparse.Namespace, repository_root: Path) -> int:
    if arguments.command == "tasks":
        config = load_config(arguments.config)
        configure_logging(config.logging.level)
        generated = generate_benchmark(repository_root, config)
        phase2_generated = generate_phase2_benchmark(repository_root, config)
        print(
            canonical_json(
                {
                    "benchmark_root": generated.root,
                    "manifest": generated.root / "manifest.json",
                    "validation_report": generated.root / "validation-report.json",
                    "phase2_benchmark_root": phase2_generated.root,
                    "phase2_manifest": phase2_generated.root / "manifest.json",
                }
            )
        )
        return 0
    if arguments.command == "oracle":
        config = load_config(arguments.config)
        if config.schema_version != 2 or config.dsl is None:
            raise ConfigurationError("oracle verify requires a schema-2 Phase 2 configuration")
        try:
            candidate_text = arguments.candidate.read_text(encoding="utf-8")
        except OSError as exc:
            raise CandidateValidationError("candidate file is unavailable") from exc
        limits = AstLimits(
            max_depth=config.dsl.max_depth,
            max_nodes=config.dsl.max_nodes,
            max_cases=config.dsl.max_cases,
        )
        try:
            document = DslCandidateDocument.from_json(
                candidate_text,
                limits=limits,
                allowed_macros=frozenset(config.dsl.allowed_macros),
            )
        except CandidateJsonError as exc:
            raise CandidateValidationError(str(exc)) from exc
        benchmark_root = benchmark_root_for_config(repository_root, config)
        public_task = load_public_task(benchmark_root, arguments.task)
        allowed = frozenset({SplitLabel.TRAINING, SplitLabel.DEVELOPMENT})
        store = HiddenTaskStore(benchmark_root)
        hidden = store.load(
            public_task.task_id,
            allowed_splits=allowed,
            purpose="phase2-cli-verification",
        )
        evaluated = ExactDslOracle(
            hidden,
            limits=limits,
            response_mode=config.oracle.response_mode,
        ).evaluate(document.ast)
        print(
            canonical_json(
                {
                    "verification_schema_version": 1,
                    "task_id": public_task.task_id,
                    "oracle_version": config.oracle.oracle_id,
                    "result": evaluated.result.deterministic_payload(),
                    "diagnostics": {"runtime_ns": evaluated.result.runtime_ns},
                }
            )
        )
        return 0 if evaluated.result.exact else 1
    if arguments.command == "benchmark":
        _unavailable("benchmark execution begins after Phase 0")
    if arguments.command == "solve":
        if arguments.config is not None:
            config = load_config(arguments.config)
            if arguments.proposer is not None and arguments.proposer != config.proposer.proposer_id:
                raise ConfigurationError(
                    f"--proposer must be '{config.proposer.proposer_id}' for this configuration"
                )
            if arguments.runs_root != DEFAULT_RUNS_ROOT:
                raise ConfigurationError("--runs-root is only used together with --resume")
            configure_logging(config.logging.level)
            outcome = start_run(
                repository_root=repository_root,
                config=config,
                config_source=str(arguments.config),
                run_id=arguments.run_id,
                interrupt_after=arguments.interrupt_after,
            )
        else:
            if arguments.run_id is not None or arguments.proposer is not None:
                raise ConfigurationError("--run-id and --proposer cannot be used with --resume")
            configure_logging("INFO")
            outcome = resume_run(
                repository_root=repository_root,
                runs_root=arguments.runs_root,
                run_id=arguments.resume,
                interrupt_after=arguments.interrupt_after,
            )
        print(
            canonical_json(
                {
                    "run_id": outcome.run_id,
                    "status": outcome.status,
                    "completed_steps": outcome.completed_steps,
                    "event_payload_hashes": outcome.event_payload_hashes,
                    "run_directory": outcome.run_directory,
                }
            )
        )
        return 0
    if arguments.command == "replay":
        configure_logging("INFO")
        replay = replay_run(
            repository_root=repository_root,
            runs_root=arguments.runs_root,
            run_id=arguments.run,
        )
        print(canonical_json(replay))
        return 0
    if arguments.command == "report":
        configure_logging("INFO")
        json_path, markdown_path = create_recorded_report(
            repository_root=repository_root,
            runs_root=arguments.runs_root,
            run_id=arguments.run,
            output_directory=arguments.out,
        )
        print(
            canonical_json(
                {
                    "run_id": arguments.run,
                    "source": "frozen-run-artifacts-only",
                    "json_report": json_path,
                    "markdown_report": markdown_path,
                }
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        return _dispatch(arguments, Path.cwd())
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except WorldModelSearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
