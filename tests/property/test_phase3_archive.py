from __future__ import annotations

import random

from world_model_search.domain.types import (
    Archive,
    Candidate,
    OracleFeedback,
    OracleResponseMode,
    OracleResult,
)
from world_model_search.dsl.ast import (
    And,
    At,
    BitExpr,
    Const,
    Count,
    Eq,
    If,
    IntConst,
    Majority,
    Not,
    Or,
    Parity,
    Xor,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.interpreter import semantic_hash
from world_model_search.dsl.json_schema import DslCandidateDocument, ast_canonical_json
from world_model_search.evaluation.rank import rank_result
from world_model_search.search.archive import (
    ArchiveLayer,
    InsertionOutcome,
    MapElitesArchive,
    RepresentationFamily,
    descriptor,
    joint_size_bin,
)
from world_model_search.serialization import canonical_json, sha256_text
from world_model_search.tasks import load_public_task


def _candidate(task_id: str, index: int, ast: BitExpr, parents: tuple[str, ...] = ()) -> Candidate:
    payload = DslCandidateDocument(ast=ast).to_json()
    return Candidate(
        candidate_id=f"c{index:04d}",
        task_id=task_id,
        ast=ast,
        parent_ids=parents,
        proposer_id="mutation",
        operator_id="property",
        context_hash="0" * 64,
        payload_hash=sha256_text(payload),
        semantic_hash=semantic_hash(ast),
    )


def _result(errors: int, bits: int, exact: bool) -> OracleResult:
    return OracleResult(
        type_valid=True,
        total=True,
        local_errors=errors,
        local_cases=8,
        rollout_pass=exact,
        exact=exact,
        ast_bits=bits,
        residual_bits=1,
        runtime_ns=0,
        response=OracleFeedback(OracleResponseMode.SCORE_ONLY),
    )


def test_randomized_archive_matches_reference_elites_and_invariants(phase2_repository) -> None:
    task = load_public_task(
        phase2_repository / "artifacts/phase2-benchmark", "d737b0ee219de6a676c139d1"
    ).public_view()
    expressions = (
        Const(0),
        Const(1),
        At(-1),
        At(0),
        At(1),
        Not(At(0)),
        Xor(At(-1), At(1)),
        Parity((-1, 0, 1)),
        Majority((-1, 0, 1)),
    )
    records = []
    for index in range(300):
        expression_index = index % len(expressions)
        ast = expressions[expression_index]
        errors = (expression_index * 7) % 9
        exact = errors == 0
        records.append(
            (
                _candidate(task.task_id, index, ast, (f"p{index % 5}",)),
                _result(errors, encoded_length(canonicalize(ast)), exact),
            )
        )
    for order_seed in range(8):
        shuffled = list(records)
        random.Random(order_seed).shuffle(shuffled)
        archive = MapElitesArchive(task, reserve_size=2)
        assert isinstance(archive, Archive)
        reference: dict[object, tuple[object, str]] = {}
        reference_canonical: dict[object, set[str]] = {}
        monotone: dict[object, object] = {}
        outcomes: set[InsertionOutcome] = set()
        for candidate, result in shuffled:
            coordinate = descriptor(candidate.ast, result, task)
            decision = archive.insert(candidate, result)
            outcomes.add(decision.outcome)
            assert decision.coordinate == coordinate
            current = archive.cells[coordinate].elite
            old_rank = monotone.get(coordinate)
            assert old_rank is None or current.rank >= old_rank
            monotone[coordinate] = current.rank
            assert len(archive.cells[coordinate].reserve) <= 2
            assert len(
                {item.lineage_signature for item in archive.cells[coordinate].reserve}
            ) == len(archive.cells[coordinate].reserve)
            existing = reference.get(coordinate)
            candidate_rank = rank_result(result)
            canonical_key = ast_canonical_json(canonicalize(candidate.ast))
            already_present = canonical_key in reference_canonical.setdefault(coordinate, set())
            if not already_present and (
                existing is None
                or candidate_rank > existing[0]
                or (candidate_rank == existing[0] and candidate.candidate_id < existing[1])
            ):
                reference[coordinate] = (candidate_rank, candidate.candidate_id)
            if not already_present and decision.outcome is not InsertionOutcome.REJECTED:
                reference_canonical[coordinate].add(canonical_key)
        assert {
            coordinate: (cell.elite.rank, cell.elite.candidate.candidate_id)
            for coordinate, cell in archive.cells.items()
        } == reference
        assert {cell.elite.coordinate.layer for cell in archive.cells.values()} == {
            ArchiveLayer.PARTIAL,
            ArchiveLayer.EXACT,
        }
        assert InsertionOutcome.INSERTED in outcomes
        assert InsertionOutcome.DUPLICATE in outcomes


def test_archive_fails_closed_on_cross_task_insertion(phase2_repository) -> None:
    task = load_public_task(
        phase2_repository / "artifacts/phase2-benchmark", "d737b0ee219de6a676c139d1"
    ).public_view()
    archive = MapElitesArchive(task)
    candidate = _candidate("0" * 24, 0, At(0))
    try:
        archive.insert(candidate, _result(1, 5, False))
    except ValueError as exc:
        assert "cross-task" in str(exc)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("cross-task insertion was accepted")


def test_descriptor_families_layers_and_public_signatures_are_reachable(
    phase2_repository,
) -> None:
    task = load_public_task(
        phase2_repository / "artifacts/phase2-benchmark", "d737b0ee219de6a676c139d1"
    ).public_view()
    expressions = (
        At(0),
        If(Eq(Count((-1, 0, 1)), IntConst(1)), Const(1), Const(0)),
        Parity((-1, 0, 1)),
        Majority((-1, 0, 1)),
        If(Eq(Count((-1, 0, 1)), IntConst(1)), At(0), At(1)),
        Xor(Parity((-1, 0, 1)), Majority((-1, 0, 1))),
    )
    families = {
        descriptor(
            expression, _result(1, encoded_length(canonicalize(expression)), False), task
        ).representation_family
        for expression in expressions
    }
    assert families == set(RepresentationFamily)
    partial = descriptor(At(0), _result(1, 5, False), task)
    exact = descriptor(At(0), _result(0, 5, True), task)
    assert partial.layer is ArchiveLayer.PARTIAL
    assert exact.layer is ArchiveLayer.EXACT
    assert partial.error_signature_cluster.startswith("p2-")


def test_archive_records_every_transition_and_bounded_reserve_eviction(
    phase2_repository,
) -> None:
    task = load_public_task(
        phase2_repository / "artifacts/phase2-benchmark", "d737b0ee219de6a676c139d1"
    ).public_view()
    archive = MapElitesArchive(task, reserve_size=2)
    expressions = (
        And(At(-1), At(0)),
        And(At(-1), At(1)),
        And(At(0), At(1)),
        Or(At(-1), At(0)),
        Or(At(-1), At(1)),
    )
    errors = (4, 2, 5, 3, 7)
    decisions = [
        archive.insert(
            _candidate(task.task_id, 400 + index, expression, (f"lineage-{index}",)),
            _result(error, encoded_length(expression), False),
        )
        for index, (expression, error) in enumerate(zip(expressions, errors, strict=True))
    ]
    duplicate = archive.insert(
        _candidate(task.task_id, 499, expressions[1], ("different-lineage",)),
        _result(2, encoded_length(expressions[1]), False),
    )
    assert {decision.outcome for decision in (*decisions, duplicate)} == set(InsertionOutcome)
    coordinate = descriptor(expressions[0], _result(4, encoded_length(expressions[0]), False), task)
    cell = archive.cells[coordinate]
    assert cell.elite.candidate.candidate_id == "c0401"
    assert len(cell.reserve) == 2
    assert any(decision.evicted_candidate_id is not None for decision in decisions)


def test_joint_size_bin_includes_every_exact_edge_and_overflow_bin() -> None:
    node_boundaries = (
        (3, 0),
        (4, 1),
        (7, 1),
        (8, 2),
        (15, 2),
        (16, 3),
        (31, 3),
        (32, 4),
        (63, 4),
        (64, 5),
    )
    bit_boundaries = (
        (12, 0),
        (13, 1),
        (24, 1),
        (25, 2),
        (48, 2),
        (49, 3),
        (96, 3),
        (97, 4),
        (192, 4),
        (193, 5),
    )
    for nodes, node_index in node_boundaries:
        for bits, bit_index in bit_boundaries:
            assert joint_size_bin(nodes, bits) == f"b{max(node_index, bit_index)}"


def test_archive_ties_choose_lexicographically_smallest_candidate_id(
    phase2_repository,
) -> None:
    task = load_public_task(
        phase2_repository / "artifacts/phase2-benchmark", "d737b0ee219de6a676c139d1"
    ).public_view()
    first_ast = And(At(-1), At(0))
    second_ast = And(At(-1), At(1))
    bits = encoded_length(first_ast)
    assert encoded_length(second_ast) == bits
    result = _result(3, bits, False)
    assert descriptor(first_ast, result, task) == descriptor(second_ast, result, task)
    archive = MapElitesArchive(task, reserve_size=2)
    archive.insert(_candidate(task.task_id, 900, first_ast, ("lineage-a",)), result)
    decision = archive.insert(_candidate(task.task_id, 100, second_ast, ("lineage-b",)), result)
    assert decision.outcome is InsertionOutcome.REPLACED
    coordinate = descriptor(first_ast, result, task)
    assert archive.cells[coordinate].elite.candidate.candidate_id == "c0100"


def test_reserve_and_elite_transitions_match_an_independent_reference(
    phase2_repository,
) -> None:
    task = load_public_task(
        phase2_repository / "artifacts/phase2-benchmark", "d737b0ee219de6a676c139d1"
    ).public_view()
    expressions = (
        And(At(-1), At(0)),
        And(At(-1), At(1)),
        And(At(0), At(1)),
        Or(At(-1), At(0)),
        Or(At(-1), At(1)),
    )
    errors = (4, 2, 5, 3, 7)
    records = [
        (
            _candidate(task.task_id, 1000 + index, expression, (f"lineage-{index}",)),
            _result(error, encoded_length(expression), False),
        )
        for index, (expression, error) in enumerate(zip(expressions, errors, strict=True))
    ]
    records.append(
        (
            _candidate(task.task_id, 1099, expressions[1], ("duplicate-lineage",)),
            _result(errors[1], encoded_length(expressions[1]), False),
        )
    )
    coordinate = descriptor(records[0][0].ast, records[0][1], task)
    assert all(
        descriptor(candidate.ast, result, task) == coordinate for candidate, result in records
    )

    archive = MapElitesArchive(task, reserve_size=2)
    reference_elite = None
    reference_reserve: list[tuple[Candidate, OracleResult, str, str]] = []

    def better(
        left: tuple[Candidate, OracleResult, str, str],
        right: tuple[Candidate, OracleResult, str, str],
    ) -> bool:
        left_rank, right_rank = rank_result(left[1]), rank_result(right[1])
        return left_rank > right_rank or (
            left_rank == right_rank and left[0].candidate_id < right[0].candidate_id
        )

    def reserve_key(
        item: tuple[Candidate, OracleResult, str, str],
    ) -> tuple[int, int, int, int, int, int, str]:
        rank = rank_result(item[1])
        return (
            -rank.type_valid,
            -rank.total,
            -rank.negative_local_errors,
            -rank.exact,
            -rank.negative_ast_bits,
            -rank.negative_runtime_ns,
            item[0].candidate_id,
        )

    for candidate, result in records:
        canonical = ast_canonical_json(canonicalize(candidate.ast))
        lineage = sha256_text(
            canonical_json(
                {"ordered_parent_ids": candidate.parent_ids, "operator": candidate.operator_id}
            )
        )
        member = (candidate, result, canonical, lineage)
        existing_members = (() if reference_elite is None else (reference_elite,)) + tuple(
            reference_reserve
        )
        if any(item[2] == canonical for item in existing_members):
            expected_outcome = InsertionOutcome.DUPLICATE
        elif reference_elite is None:
            reference_elite = member
            expected_outcome = InsertionOutcome.INSERTED
        elif better(member, reference_elite):
            demoted = reference_elite
            reference_elite = member
            if not any(item[3] == demoted[3] for item in reference_reserve):
                reference_reserve = sorted([*reference_reserve, demoted], key=reserve_key)[:2]
            expected_outcome = InsertionOutcome.REPLACED
        elif any(item[3] == member[3] for item in reference_reserve):
            expected_outcome = InsertionOutcome.REJECTED
        else:
            old_reserve = list(reference_reserve)
            reference_reserve = sorted([*reference_reserve, member], key=reserve_key)[:2]
            expected_outcome = (
                InsertionOutcome.RESERVED
                if reference_reserve != old_reserve
                else InsertionOutcome.REJECTED
            )

        decision = archive.insert(candidate, result)
        assert decision.outcome is expected_outcome
        assert reference_elite is not None
        cell = archive.cells[coordinate]
        assert cell.elite.candidate.candidate_id == reference_elite[0].candidate_id
        assert [item.candidate.candidate_id for item in cell.reserve] == [
            item[0].candidate_id for item in reference_reserve
        ]
