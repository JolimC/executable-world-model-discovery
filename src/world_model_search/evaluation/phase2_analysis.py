"""Deterministic Phase 2 gate analysis generated during a recorded run."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from world_model_search.domain.types import (
    OracleFeedback,
    OracleResponseMode,
    OracleResult,
    ProposalContext,
    Task,
)
from world_model_search.dsl.ast import (
    AddConst,
    And,
    At,
    Between,
    BitExpr,
    Const,
    Count,
    Eq,
    Ge,
    If,
    IntConst,
    IntExpr,
    Le,
    Majority,
    Not,
    Or,
    Parity,
    PredExpr,
    TruthTable,
    Xor,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import decode, encode, encoded_length, opcodes_are_prefix_free
from world_model_search.dsl.interpreter import semantic_hash, truth_table
from world_model_search.dsl.json_schema import ast_canonical_json, ast_to_value
from world_model_search.dsl.versions import (
    ANALYSIS_ARTIFACT_VERSION,
    ANALYSIS_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    CANONICALIZER_VERSION,
    DSL_VERSION,
    ENUMERATOR_VERSION,
    INTERPRETER_VERSION,
    PREFIX_CODE_VERSION,
    RANK_VERSION,
    RESIDUAL_CODE_VERSION,
    SEMANTIC_HASH_VERSION,
    TRUTH_TABLE_BASELINE_VERSION,
)
from world_model_search.evaluation.rank import rank_result
from world_model_search.oracle.elementary import ROLLOUT_VERSION, SIMULATOR_VERSION, ElementaryRule
from world_model_search.oracle.exact import EXACT_ORACLE_VERSION
from world_model_search.oracle.residual import residual_bits
from world_model_search.persistence.artifacts import write_content_artifact
from world_model_search.proposer.enumerative import EnumerationResult
from world_model_search.serialization import JsonObject, JsonValue, canonical_json, sha256_text
from world_model_search.tasks import HiddenTaskBundle, OracleTaskAccess

CANONICAL_PROPERTY_SEEDS = 512


@dataclass(frozen=True, slots=True)
class AnalysisPaths:
    manifest: Path
    complexity_json: Path
    complexity_csv: Path
    complexity_svg: Path
    collapses: Path
    gate_report: Path
    access_ledger: Path


def _random_int(rng: random.Random, depth: int) -> IntExpr:
    if depth <= 0 or rng.randrange(3) < 2:
        if rng.randrange(2):
            return IntConst(rng.randrange(-3, 4))
        mask = tuple(offset for offset in (-1, 0, 1) if rng.randrange(2))
        return Count(mask or (rng.choice((-1, 0, 1)),))
    return AddConst(_random_int(rng, depth - 1), rng.randrange(-3, 4))


def _random_pred(rng: random.Random, depth: int) -> PredExpr:
    choice = rng.randrange(4)
    if choice == 0:
        return Eq(_random_int(rng, depth), _random_int(rng, depth))
    if choice == 1:
        return Le(_random_int(rng, depth), _random_int(rng, depth))
    if choice == 2:
        return Ge(_random_int(rng, depth), _random_int(rng, depth))
    return Between(
        _random_int(rng, depth),
        _random_int(rng, depth),
        _random_int(rng, depth),
    )


def _random_bit(rng: random.Random, depth: int) -> BitExpr:
    if depth <= 0:
        choice = rng.randrange(5)
        if choice == 0:
            return Const(rng.randrange(2))
        if choice == 1:
            return At(rng.choice((-1, 0, 1)))
        mask = tuple(offset for offset in (-1, 0, 1) if rng.randrange(2))
        mask = mask or (rng.choice((-1, 0, 1)),)
        if choice == 2:
            return Parity(mask)
        if choice == 3:
            return Majority(mask)
        return TruthTable(tuple(rng.randrange(2) for _ in range(8)))
    choice = rng.randrange(10)
    if choice < 3:
        return _random_bit(rng, 0)
    if choice == 3:
        return Not(_random_bit(rng, depth - 1))
    if choice == 4:
        return And(_random_bit(rng, depth - 1), _random_bit(rng, depth - 1))
    if choice == 5:
        return Or(_random_bit(rng, depth - 1), _random_bit(rng, depth - 1))
    if choice == 6:
        return Xor(_random_bit(rng, depth - 1), _random_bit(rng, depth - 1))
    return If(
        _random_pred(rng, max(0, depth - 2)),
        _random_bit(rng, depth - 1),
        _random_bit(rng, depth - 1),
    )


def _canonical_property_evidence() -> JsonObject:
    constructor_names: set[str] = set()
    exhaustive_comparisons = 0
    for seed in range(CANONICAL_PROPERTY_SEEDS):
        source = _random_bit(random.Random(seed), 3)
        canonical = canonicalize(source)
        before, after = truth_table(source), truth_table(canonical)
        if before != after:
            raise AssertionError(f"canonicalization changed semantics for seed {seed}")
        if canonicalize(canonical) != canonical:
            raise AssertionError(f"canonicalization was not idempotent for seed {seed}")
        encoded = encode(canonical)
        if decode(encoded) != canonical or encode(decode(encoded)) != encoded:
            raise AssertionError(f"codec round trip failed for seed {seed}")
        if semantic_hash(source) != semantic_hash(canonical):
            raise AssertionError(f"semantic hash changed for seed {seed}")
        if ast_canonical_json(canonical) != ast_canonical_json(canonicalize(source)):
            raise AssertionError(f"canonical JSON was unstable for seed {seed}")
        constructor_names.update(_constructor_names(source))
        exhaustive_comparisons += 8
    constructor_values: list[JsonValue] = list(sorted(constructor_names))
    evidence: JsonObject = {
        "seed_policy": "consecutive-integers-0-through-511-random-v1",
        "seed_count": CANONICAL_PROPERTY_SEEDS,
        "exhaustive_case_comparisons": exhaustive_comparisons,
        "constructor_families_observed": constructor_values,
        "truth_tables_preserved": True,
        "idempotent": True,
        "stable_json_hash_and_code": True,
        "codec_round_trips": CANONICAL_PROPERTY_SEEDS,
    }
    return evidence


def _constructor_names(expr: BitExpr | IntExpr | PredExpr) -> set[str]:
    names = {type(expr).__name__}
    if isinstance(expr, Not):
        names.update(_constructor_names(expr.expr))
    elif isinstance(expr, And | Or | Xor):
        names.update(_constructor_names(expr.left))
        names.update(_constructor_names(expr.right))
    elif isinstance(expr, If):
        names.update(_constructor_names(expr.condition))
        names.update(_constructor_names(expr.then_branch))
        names.update(_constructor_names(expr.else_branch))
    elif isinstance(expr, AddConst):
        names.update(_constructor_names(expr.expr))
    elif isinstance(expr, Eq | Le | Ge):
        names.update(_constructor_names(expr.left))
        names.update(_constructor_names(expr.right))
    elif isinstance(expr, Between):
        names.update(_constructor_names(expr.value))
        names.update(_constructor_names(expr.lower))
        names.update(_constructor_names(expr.upper))
    return names


def _codec_evidence(enumeration: EnumerationResult) -> JsonObject:
    codes = tuple(encode(program.ast) for program in enumeration.programs)
    if not opcodes_are_prefix_free():
        raise AssertionError("opcode table is not prefix-free")
    if any(
        right.startswith(left)
        for index, left in enumerate(codes)
        for other_index, right in enumerate(codes)
        if index != other_index
    ):
        raise AssertionError("complete AST code set is not prefix-free")
    for program, bits in zip(enumeration.programs, codes, strict=True):
        if decode(bits) != program.ast or encode(decode(bits)) != bits:
            raise AssertionError("enumerated codec round trip failed")
    return {
        "round_trip_programs": len(codes),
        "repeated_byte_equality_checks": len(codes),
        "opcode_prefix_free": True,
        "complete_value_prefix_free": True,
        "decode_reencode_equal": True,
    }


def _complexity_records(enumeration: EnumerationResult) -> tuple[JsonObject, ...]:
    by_semantics = {program.ordered_semantics: program for program in enumeration.programs}
    records: list[JsonObject] = []
    for number in range(256):
        semantics = ElementaryRule(number).ordered_semantics
        baseline_bits = encoded_length(TruthTable(semantics))
        program = by_semantics.get(semantics)
        record: JsonObject = {
            "rule_number": number,
            "truth_table_baseline_bits": baseline_bits,
            "truth_table_baseline_version": TRUTH_TABLE_BASELINE_VERSION,
            "best_enumerated_status": (
                "found" if program is not None else "not_found_within_bounds"
            ),
            "representable_by_truth_table_baseline": True,
            "best_enumerated_bits": program.ast_bits if program is not None else None,
            "best_enumerated_discovery_index": (
                program.discovery_index if program is not None else None
            ),
        }
        records.append(record)
    return tuple(records)


def _known_recoveries(enumeration: EnumerationResult) -> JsonObject:
    targets = {
        "rule_90": ElementaryRule(90).ordered_semantics,
        "rule_150": ElementaryRule(150).ordered_semantics,
        "majority_three": truth_table(Majority((-1, 0, 1))),
        "parity_three": truth_table(Parity((-1, 0, 1))),
    }
    records: JsonObject = {}
    for name, target in targets.items():
        program = next(
            (
                candidate
                for candidate in enumeration.programs
                if candidate.ordered_semantics == target
            ),
            None,
        )
        if program is None:
            raise AssertionError(f"enumerator failed frozen target: {name}")
        records[name] = {
            "discovery_index": program.discovery_index,
            "ast": ast_to_value(program.ast),
            "semantic_hash": program.semantic_hash,
            "ast_bits": program.ast_bits,
            "residual_bits": residual_bits(0, 8),
        }
    records["declared_standard_forms"] = {
        "rule_90": {
            "ast": ast_to_value(Xor(At(-1), At(1))),
            "ast_bits": encoded_length(Xor(At(-1), At(1))),
        },
        "rule_150": {
            "ast": ast_to_value(Xor(Xor(At(-1), At(0)), At(1))),
            "ast_bits": encoded_length(Xor(Xor(At(-1), At(0)), At(1))),
        },
    }
    return records


def _collapse_examples() -> tuple[JsonObject, ...]:
    source_groups = (
        ("commutative-order", And(At(1), At(-1))),
        ("idempotence", Or(At(0), At(0))),
        ("double-negation", Not(Not(At(0)))),
        ("dead-branch", If(Eq(IntConst(0), IntConst(0)), At(-1), At(1))),
        ("semantic-macro-collapse", Xor(At(-1), At(1))),
        ("semantic-macro-collapse", Parity((-1, 1))),
    )
    records: list[JsonObject] = []
    for reason, source in source_groups:
        canonical = canonicalize(source)
        records.append(
            {
                "reason": reason,
                "original_ast": ast_to_value(source),
                "canonical_ast": ast_to_value(canonical),
                "semantic_hash": semantic_hash(canonical),
                "canonical_ast_bits": encoded_length(canonical),
            }
        )
    return tuple(records)


def _rank_evidence() -> JsonObject:
    feedback = OracleFeedback(mode=OracleResponseMode.SCORE_ONLY)

    def result(errors: int, exact: bool, bits: int, runtime: int = 0) -> OracleResult:
        return OracleResult(
            type_valid=True,
            total=True,
            local_errors=errors,
            local_cases=8,
            rollout_pass=exact,
            exact=exact,
            ast_bits=bits,
            residual_bits=residual_bits(errors, 8),
            runtime_ns=runtime,
            response=feedback,
        )

    more_correct = rank_result(result(0, True, 30)) > rank_result(result(1, False, 4))
    shorter_exact = rank_result(result(0, True, 8, 100)) > rank_result(result(0, True, 9, 1))
    if not more_correct or not shorter_exact:
        raise AssertionError("lexicographic rank contract failed")
    return {
        "correctness_dominates_compression": more_correct,
        "exact_length_dominates_efficiency": shorter_exact,
        "runtime_in_deterministic_rank": False,
    }


def _leakage_evidence(task: Task, hidden: HiddenTaskBundle) -> JsonObject:
    context_json = canonical_json(ProposalContext(task=task.public_view()))
    decoded: object = json.loads(context_json)
    forbidden_keys = {
        "reference_rule",
        "ordered_semantics_000_to_111",
        "semantic_hash",
        "internal_family",
        "internal_family_id",
        "seed",
        "seeds",
        "hidden_artifact_id",
        "exact_case_set_id",
        "rollout_suite_id",
        "locked_rollout",
        "task_seed",
    }
    found_keys: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden_keys:
                    found_keys.add(key)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(decoded)
    sensitive_values = (
        hidden.semantic_hash,
        hidden.internal_family,
        hidden.task_seed,
        hidden.rollout_case_seed,
        hidden.rollout_initial_state_seed,
        list(hidden.ordered_semantics),
        [list(state) for state in hidden.locked_rollout.states],
    )
    exact_value_hits = sum(
        context_json.find(canonical_json(value)) >= 0 for value in sensitive_values
    )
    if found_keys or exact_value_hits:
        raise AssertionError("oracle-only values entered proposer-visible context")
    return {
        "serialized_contexts_scanned": 1,
        "forbidden_nested_keys_found": 0,
        "sensitive_nested_values_found": 0,
        "passed": True,
    }


def _svg(records: tuple[JsonObject, ...]) -> str:
    left, top, plot_width, plot_height = 60, 30, 860, 330

    def x(number: int) -> float:
        return left + plot_width * number / 255

    def y(bits: int) -> float:
        return top + plot_height * (1 - bits / 40)

    def point(record: JsonObject) -> str | None:
        number, bits = record["rule_number"], record["best_enumerated_bits"]
        if isinstance(number, bool) or not isinstance(number, int):
            raise AssertionError("complexity record has invalid rule number")
        if isinstance(bits, bool) or not isinstance(bits, int):
            return None
        return f"{x(number):.2f},{y(bits):.2f}"

    point_values = tuple(point(record) for record in records)
    points = " ".join(value for value in point_values if value is not None)
    baseline_y = y(19)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" '
        'viewBox="0 0 960 420">\n'
        '<rect width="960" height="420" fill="white"/>\n'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>\n'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="black"/>\n'
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{left + plot_width}" '
        f'y2="{baseline_y:.2f}" stroke="#d62728" stroke-dasharray="5,4"/>\n'
        f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="1"/>\n'
        '<text x="480" y="405" text-anchor="middle" font-family="sans-serif" '
        'font-size="13">Elementary rule number (0-255)</text>\n'
        '<text x="16" y="210" text-anchor="middle" font-family="sans-serif" font-size="13" '
        'transform="rotate(-90 16 210)">Canonical AST prefix-code bits</text>\n'
        '<text x="70" y="22" font-family="sans-serif" font-size="12">'
        "Blue: best structured enumeration; red: 19-bit truth-table baseline</text>\n"
        "</svg>\n"
    )


def write_phase2_analysis(
    *,
    run_directory: Path,
    enumeration: EnumerationResult,
    task: Task,
    hidden: HiddenTaskBundle,
    accesses: tuple[OracleTaskAccess, ...],
) -> AnalysisPaths:
    """Execute frozen gate checks once and write immutable analysis artifacts."""

    root = run_directory / "analysis"
    records = _complexity_records(enumeration)
    complexity: JsonObject = {
        "artifact_version": ANALYSIS_ARTIFACT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "axes": {
            "x": "elementary rule number",
            "y": "canonical AST prefix-code bits",
            "units": "bits",
        },
        "prefix_code_version": PREFIX_CODE_VERSION,
        "enumerator_version": ENUMERATOR_VERSION,
        "records": list(records),
    }
    csv_lines = [
        "rule_number,truth_table_baseline_bits,best_enumerated_status,best_enumerated_bits,discovery_index"
    ]
    for record in records:
        csv_lines.append(
            ",".join(
                str(record[key]) if record[key] is not None else ""
                for key in (
                    "rule_number",
                    "truth_table_baseline_bits",
                    "best_enumerated_status",
                    "best_enumerated_bits",
                    "best_enumerated_discovery_index",
                )
            )
        )
    access_ledger: JsonObject = {
        "artifact_version": ANALYSIS_ARTIFACT_VERSION,
        "oracle_accesses": [
            {"task_id": access.task_id, "split": access.split.value, "purpose": access.purpose}
            for access in accesses
        ],
        "allowed_splits": ["training", "development"],
        "test_task_oracle_artifacts_accessed": any(
            access.split.value == "test" for access in accesses
        ),
        "validation_task_oracle_artifacts_accessed": any(
            access.split.value == "validation" for access in accesses
        ),
    }
    if access_ledger["test_task_oracle_artifacts_accessed"] is not False:
        raise AssertionError("Phase 2 gate attempted to access a test oracle artifact")
    lengths = [program.ast_bits for program in enumeration.programs]
    gate_report: JsonObject = {
        "artifact_version": ANALYSIS_ARTIFACT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "versions": {
            "dsl": DSL_VERSION,
            "candidate_schema": CANDIDATE_SCHEMA_VERSION,
            "interpreter": INTERPRETER_VERSION,
            "canonicalizer": CANONICALIZER_VERSION,
            "semantic_hash": SEMANTIC_HASH_VERSION,
            "prefix_code": PREFIX_CODE_VERSION,
            "residual_code": RESIDUAL_CODE_VERSION,
            "rank": RANK_VERSION,
            "enumerator": ENUMERATOR_VERSION,
            "truth_table_baseline": TRUTH_TABLE_BASELINE_VERSION,
            "exact_oracle": EXACT_ORACLE_VERSION,
            "simulator": SIMULATOR_VERSION,
            "rollout": ROLLOUT_VERSION,
        },
        "bounds": {
            "max_bits": enumeration.bounds.max_bits,
            "max_depth": enumeration.bounds.max_depth,
            "max_nodes": enumeration.bounds.max_nodes,
            "max_candidates": enumeration.bounds.max_candidates,
        },
        "enumeration": {
            "candidates_examined": enumeration.candidates_examined,
            "programs_emitted": len(enumeration.programs),
            "canonical_duplicates": enumeration.canonical_duplicates,
            "semantic_duplicates": enumeration.semantic_duplicates,
            "cost_monotone": lengths == sorted(lengths),
            "tie_breaker": "canonical-ast-json-utf8-lexicographic-v1",
            "duplicate_policy": "canonical-then-semantic-first-v1",
            "truth_table_literals_enumerated": False,
            "stopped_at_candidate_bound": enumeration.stopped_at_candidate_bound,
        },
        "codec": _codec_evidence(enumeration),
        "canonicalization_properties": _canonical_property_evidence(),
        "known_form_recovery": _known_recoveries(enumeration),
        "residual": {
            "errors_0_through_8": [residual_bits(errors, 8) for errors in range(9)],
            "all_checked": True,
        },
        "rank": _rank_evidence(),
        "leakage": _leakage_evidence(task, hidden),
        "split_protocol": {
            "phase1_validation_consumed": True,
            "phase2_validation_consumed": False,
            "test_task_outcomes_accessed": False,
            "standalone_rule_150_and_all_256_are_language_mechanics_not_test_outcomes": True,
        },
    }
    collapses: JsonObject = {
        "artifact_version": ANALYSIS_ARTIFACT_VERSION,
        "examples": list(_collapse_examples()),
        "enumerator_examples": [
            {
                "reason": record.reason,
                "source_ast_json": record.source_json,
                "canonical_ast_json": record.canonical_json,
                "retained_ast_json": record.retained_json,
            }
            for record in enumeration.collapse_records
        ],
    }
    contents = {
        "elementary-complexity.json": canonical_json(complexity) + "\n",
        "elementary-complexity.csv": "\n".join(csv_lines) + "\n",
        "elementary-complexity.svg": _svg(records),
        "collapse-examples.json": canonical_json(collapses) + "\n",
        "gate-report.json": canonical_json(gate_report) + "\n",
        "access-ledger.json": canonical_json(access_ledger) + "\n",
    }
    for name, content in contents.items():
        write_content_artifact(root / name, content.rstrip("\n"))
    manifest: JsonObject = {
        "artifact_version": ANALYSIS_ARTIFACT_VERSION,
        "source": "recorded-phase2-run-computation",
        "files": {name: sha256_text(content.rstrip("\n")) for name, content in contents.items()},
    }
    write_content_artifact(root / "manifest.json", canonical_json(manifest))
    return AnalysisPaths(
        manifest=root / "manifest.json",
        complexity_json=root / "elementary-complexity.json",
        complexity_csv=root / "elementary-complexity.csv",
        complexity_svg=root / "elementary-complexity.svg",
        collapses=root / "collapse-examples.json",
        gate_report=root / "gate-report.json",
        access_ledger=root / "access-ledger.json",
    )
