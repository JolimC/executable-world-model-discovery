"""Versioned public descriptors and Phase 3 MAP-Elites archive policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from world_model_search.domain.types import Candidate, CandidateSummary, OracleResult, PublicTask
from world_model_search.dsl.ast import BitExpr, Expr, ast_size, children
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import encoded_length
from world_model_search.dsl.interpreter import evaluate
from world_model_search.dsl.json_schema import ast_canonical_json
from world_model_search.dsl.versions import (
    PHASE3_ARCHIVE_VERSION,
    PHASE3_DESCRIPTOR_VERSION,
    PHASE3_INCUMBENT_VERSION,
)
from world_model_search.evaluation.rank import CandidateRank, rank_result
from world_model_search.serialization import JsonObject, sha256_text

SIZE_NODE_EDGES = (3, 7, 15, 31, 63)
SIZE_BIT_EDGES = (12, 24, 48, 96, 192)
DEFAULT_RESERVE_SIZE = 2


class RepresentationFamily(StrEnum):
    POSITION_SPECIFIC = "position-specific"
    COUNT_BASED = "count-based"
    PARITY = "parity"
    THRESHOLD = "threshold"
    CONDITIONAL = "conditional"
    MIXED = "mixed"


class ArchiveLayer(StrEnum):
    PARTIAL = "partial"
    EXACT = "exact"


class InsertionOutcome(StrEnum):
    INSERTED = "inserted"
    REPLACED = "replaced"
    RESERVED = "reserved"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


@dataclass(frozen=True, order=True, slots=True)
class ArchiveCoordinate:
    size_bin: str
    representation_family: RepresentationFamily
    error_signature_cluster: str
    layer: ArchiveLayer

    def to_value(self) -> JsonObject:
        return {
            "descriptor_version": PHASE3_DESCRIPTOR_VERSION,
            "size_bin": self.size_bin,
            "representation_family": self.representation_family.value,
            "error_signature_cluster": self.error_signature_cluster,
            "layer": self.layer.value,
        }


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    candidate: Candidate
    result: OracleResult
    rank: CandidateRank
    coordinate: ArchiveCoordinate
    lineage_signature: str


@dataclass(frozen=True, slots=True)
class ArchiveCell:
    elite: ArchiveMember
    reserve: tuple[ArchiveMember, ...] = ()


@dataclass(frozen=True, slots=True)
class ArchiveDecision:
    archive_version: str
    task_id: str
    coordinate: ArchiveCoordinate
    outcome: InsertionOutcome
    candidate_id: str
    inserted_candidate_id: str | None
    replaced_candidate_id: str | None
    evicted_candidate_id: str | None
    role: str | None
    rank_payload: JsonObject
    decision_hash: str

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        coordinate: ArchiveCoordinate,
        outcome: InsertionOutcome,
        candidate_id: str,
        inserted_candidate_id: str | None,
        replaced_candidate_id: str | None,
        evicted_candidate_id: str | None,
        role: str | None,
        rank: CandidateRank,
    ) -> ArchiveDecision:
        rank_payload = rank_to_value(rank)
        payload: JsonObject = {
            "archive_version": PHASE3_ARCHIVE_VERSION,
            "task_id": task_id,
            "coordinate": coordinate.to_value(),
            "outcome": outcome.value,
            "candidate_id": candidate_id,
            "inserted_candidate_id": inserted_candidate_id,
            "replaced_candidate_id": replaced_candidate_id,
            "evicted_candidate_id": evicted_candidate_id,
            "role": role,
            "rank": rank_payload,
        }
        return cls(
            archive_version=PHASE3_ARCHIVE_VERSION,
            task_id=task_id,
            coordinate=coordinate,
            outcome=outcome,
            candidate_id=candidate_id,
            inserted_candidate_id=inserted_candidate_id,
            replaced_candidate_id=replaced_candidate_id,
            evicted_candidate_id=evicted_candidate_id,
            role=role,
            rank_payload=rank_payload,
            decision_hash=sha256_text(_json(payload)),
        )

    def to_value(self) -> JsonObject:
        return {
            "archive_version": self.archive_version,
            "task_id": self.task_id,
            "coordinate": self.coordinate.to_value(),
            "outcome": self.outcome.value,
            "candidate_id": self.candidate_id,
            "inserted_candidate_id": self.inserted_candidate_id,
            "replaced_candidate_id": self.replaced_candidate_id,
            "evicted_candidate_id": self.evicted_candidate_id,
            "role": self.role,
            "rank": self.rank_payload,
            "decision_hash": self.decision_hash,
        }


def _json(value: object) -> str:
    from world_model_search.serialization import canonical_json

    return canonical_json(value)


def rank_to_value(rank: CandidateRank) -> JsonObject:
    return {
        "type_valid": rank.type_valid,
        "total": rank.total,
        "negative_local_errors": rank.negative_local_errors,
        "exact": rank.exact,
        "negative_ast_bits": rank.negative_ast_bits,
        "negative_runtime_ns": rank.negative_runtime_ns,
    }


def _edge_index(value: int, edges: tuple[int, ...]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def size_bin(expr: BitExpr) -> str:
    """Joint conservative node/code bin with frozen edges on both quantities."""

    nodes, _ = ast_size(expr)
    return joint_size_bin(nodes, encoded_length(expr))


def joint_size_bin(node_count: int, code_bits: int) -> str:
    """Map measured size to the frozen joint bin, including exact edge behavior."""

    if (
        isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count < 1
        or isinstance(code_bits, bool)
        or not isinstance(code_bits, int)
        or code_bits < 1
    ):
        raise ValueError("joint size measurements must be positive integers")
    index = max(
        _edge_index(node_count, SIZE_NODE_EDGES),
        _edge_index(code_bits, SIZE_BIT_EDGES),
    )
    return f"b{index}"


def representation_family(expr: BitExpr) -> RepresentationFamily:
    """Classify syntax only, using the frozen precedence conditional/parity/threshold/count."""

    names = {type(node).__name__ for node in _walk(expr)}
    if "TruthTable" in names:
        return RepresentationFamily.MIXED
    if "If" in names:
        boolean_mechanisms = names & {"At", "Not", "And", "Or", "Xor", "Parity", "Majority"}
        if not boolean_mechanisms:
            if names & {"Le", "Ge", "Between"}:
                return RepresentationFamily.THRESHOLD
            return RepresentationFamily.COUNT_BASED
        if (names & {"Parity", "Xor"}) and (names & {"Majority", "Le", "Ge", "Between"}):
            return RepresentationFamily.MIXED
        return RepresentationFamily.CONDITIONAL
    position_ops = bool(names & {"At", "Not", "And", "Or"})
    parity_ops = bool(names & {"Parity", "Xor"})
    threshold_ops = bool(names & {"Majority", "Le", "Ge", "Between"})
    count_ops = bool(names & {"Count", "AddConst", "Eq", "IntConst"})
    active = sum((position_ops, parity_ops, threshold_ops, count_ops))
    if active > 1:
        # Threshold syntax necessarily contains integer syntax; that pair is one family.
        if threshold_ops and count_ops and not position_ops and not parity_ops:
            return RepresentationFamily.THRESHOLD
        return RepresentationFamily.MIXED
    if parity_ops:
        return RepresentationFamily.PARITY
    if threshold_ops:
        return RepresentationFamily.THRESHOLD
    if count_ops:
        return RepresentationFamily.COUNT_BASED
    return RepresentationFamily.POSITION_SPECIFIC


def _walk(expr: BitExpr) -> tuple[Expr, ...]:
    result: list[Expr] = []

    def visit(node: Expr) -> None:
        result.append(node)
        for child in children(node):
            visit(child)

    visit(expr)
    return tuple(result)


def public_probe_contract(task: PublicTask) -> tuple[tuple[tuple[int, int, int], int], ...]:
    """First-observed unique local cases from public traces, capped at sixteen."""

    probes: list[tuple[tuple[int, int, int], int]] = []
    seen: set[tuple[tuple[int, int, int], int]] = set()
    for demonstration in task.demonstrations:
        before = tuple(int(bit) for bit in demonstration.observation)
        after = tuple(int(bit) for bit in demonstration.successor)
        for index, expected in enumerate(after):
            size = len(before)
            probe = (
                (before[(index - 1) % size], before[index], before[(index + 1) % size]),
                expected,
            )
            if probe not in seen:
                probes.append(probe)
                seen.add(probe)
            if len(probes) == 16:
                return tuple(probes)
    if not probes:
        raise ValueError("public task has no fixed probes")
    return tuple(probes)


def error_signature_cluster(expr: BitExpr, task: PublicTask) -> str:
    probes = public_probe_contract(task)
    signature = "".join(str(int(evaluate(expr, case) != expected)) for case, expected in probes)
    return f"p{len(probes)}-{signature}"


def descriptor(expr: BitExpr, result: OracleResult, task: PublicTask) -> ArchiveCoordinate:
    canonical = canonicalize(expr)
    return ArchiveCoordinate(
        size_bin=size_bin(canonical),
        representation_family=representation_family(canonical),
        error_signature_cluster=error_signature_cluster(canonical, task),
        layer=ArchiveLayer.EXACT if result.exact else ArchiveLayer.PARTIAL,
    )


def _lineage_signature(candidate: Candidate) -> str:
    return sha256_text(
        _json({"ordered_parent_ids": candidate.parent_ids, "operator": candidate.operator_id})
    )


def _better(left: ArchiveMember, right: ArchiveMember) -> bool:
    return left.rank > right.rank or (
        left.rank == right.rank and left.candidate.candidate_id < right.candidate.candidate_id
    )


class MapElitesArchive:
    """Per-task archive with monotone elites and a bounded lineage-diverse reserve."""

    archive_version = PHASE3_ARCHIVE_VERSION

    def __init__(self, task: PublicTask, *, reserve_size: int = DEFAULT_RESERVE_SIZE) -> None:
        if reserve_size < 0:
            raise ValueError("reserve size must be nonnegative")
        self.task = task
        self.reserve_size = reserve_size
        self._cells: dict[ArchiveCoordinate, ArchiveCell] = {}

    @property
    def cells(self) -> dict[ArchiveCoordinate, ArchiveCell]:
        return dict(self._cells)

    def insert(self, candidate: Candidate, result: OracleResult) -> ArchiveDecision:
        if candidate.task_id != self.task.task_id:
            raise ValueError("cross-task archive insertion is forbidden")
        if not isinstance(candidate.ast, BitExpr):
            raise TypeError("Phase 3 archive requires typed DSL candidates")
        coordinate = descriptor(candidate.ast, result, self.task)
        member = ArchiveMember(
            candidate=candidate,
            result=result,
            rank=rank_result(result),
            coordinate=coordinate,
            lineage_signature=_lineage_signature(candidate),
        )
        cell = self._cells.get(coordinate)
        if cell is None:
            self._cells[coordinate] = ArchiveCell(elite=member)
            return self._decision(member, InsertionOutcome.INSERTED, role="elite")
        existing = (cell.elite, *cell.reserve)
        canonical_json = ast_canonical_json(canonicalize(candidate.ast))
        if any(
            ast_canonical_json(canonicalize(_typed_ast(item.candidate))) == canonical_json
            for item in existing
        ):
            return self._decision(member, InsertionOutcome.DUPLICATE)
        if _better(member, cell.elite):
            reserve = self._admit_reserve(cell.reserve, cell.elite)
            evicted = _evicted(cell.reserve, reserve, cell.elite)
            self._cells[coordinate] = ArchiveCell(elite=member, reserve=reserve)
            return self._decision(
                member,
                InsertionOutcome.REPLACED,
                replaced=cell.elite.candidate.candidate_id,
                evicted=evicted,
                role="elite",
            )
        reserve = self._admit_reserve(cell.reserve, member)
        if reserve == cell.reserve:
            return self._decision(member, InsertionOutcome.REJECTED)
        evicted = _evicted(cell.reserve, reserve, member)
        self._cells[coordinate] = ArchiveCell(elite=cell.elite, reserve=reserve)
        return self._decision(member, InsertionOutcome.RESERVED, evicted=evicted, role="reserve")

    def _admit_reserve(
        self, reserve: tuple[ArchiveMember, ...], member: ArchiveMember
    ) -> tuple[ArchiveMember, ...]:
        if self.reserve_size == 0 or any(
            item.lineage_signature == member.lineage_signature for item in reserve
        ):
            return reserve
        candidates = (*reserve, member)
        ordered = tuple(sorted(candidates, key=_reserve_sort_key))
        return ordered[: self.reserve_size]

    def _decision(
        self,
        member: ArchiveMember,
        outcome: InsertionOutcome,
        *,
        replaced: str | None = None,
        evicted: str | None = None,
        role: str | None = None,
    ) -> ArchiveDecision:
        inserted = (
            member.candidate.candidate_id
            if outcome
            in {
                InsertionOutcome.INSERTED,
                InsertionOutcome.REPLACED,
                InsertionOutcome.RESERVED,
            }
            else None
        )
        return ArchiveDecision.create(
            task_id=self.task.task_id,
            coordinate=member.coordinate,
            outcome=outcome,
            candidate_id=member.candidate.candidate_id,
            inserted_candidate_id=inserted,
            replaced_candidate_id=replaced,
            evicted_candidate_id=evicted,
            role=role,
            rank=member.rank,
        )

    def candidate_summaries(
        self, *, coordinate: ArchiveCoordinate | None = None
    ) -> tuple[CandidateSummary, ...]:
        cells = (
            ((coordinate, self._cells[coordinate]),)
            if coordinate is not None and coordinate in self._cells
            else tuple(sorted(self._cells.items()))
        )
        return tuple(
            CandidateSummary(member.candidate.candidate_id, member.candidate.ast)
            for _, cell in cells
            for member in (cell.elite, *cell.reserve)
        )

    def branch_ids(self) -> tuple[str, ...]:
        return tuple(
            _coordinate_id(self.task.task_id, coordinate) for coordinate in sorted(self._cells)
        )

    def coordinate_for_branch(self, branch_id: str) -> ArchiveCoordinate:
        matches = tuple(
            coordinate
            for coordinate in self._cells
            if _coordinate_id(self.task.task_id, coordinate) == branch_id
        )
        if len(matches) != 1:
            raise ValueError("archive branch is unavailable")
        return matches[0]


def _typed_ast(candidate: Candidate) -> BitExpr:
    if not isinstance(candidate.ast, BitExpr):
        raise TypeError("archive candidate is not a typed AST")
    return candidate.ast


def _reserve_sort_key(member: ArchiveMember) -> tuple[int, int, int, int, int, int, str]:
    rank = member.rank
    return (
        -rank.type_valid,
        -rank.total,
        -rank.negative_local_errors,
        -rank.exact,
        -rank.negative_ast_bits,
        -rank.negative_runtime_ns,
        member.candidate.candidate_id,
    )


def _evicted(
    old: tuple[ArchiveMember, ...], new: tuple[ArchiveMember, ...], added: ArchiveMember
) -> str | None:
    if added not in new:
        return None
    new_ids = {member.candidate.candidate_id for member in new}
    removed = tuple(
        member.candidate.candidate_id
        for member in old
        if member.candidate.candidate_id not in new_ids
    )
    return removed[0] if removed else None


def _coordinate_id(task_id: str, coordinate: ArchiveCoordinate) -> str:
    return f"{task_id}:{sha256_text(_json(coordinate.to_value()))[:16]}"


class SingleIncumbent:
    """Cost-matched one-best-candidate mechanism under the same rank/tie contract."""

    archive_version = PHASE3_INCUMBENT_VERSION

    def __init__(self, task: PublicTask) -> None:
        self.task = task
        self.member: ArchiveMember | None = None

    def insert(self, candidate: Candidate, result: OracleResult) -> ArchiveDecision:
        if candidate.task_id != self.task.task_id:
            raise ValueError("cross-task incumbent insertion is forbidden")
        if not isinstance(candidate.ast, BitExpr):
            raise TypeError("incumbent requires a typed AST")
        coordinate = descriptor(candidate.ast, result, self.task)
        member = ArchiveMember(
            candidate=candidate,
            result=result,
            rank=rank_result(result),
            coordinate=coordinate,
            lineage_signature=_lineage_signature(candidate),
        )
        previous = self.member
        if previous is None:
            self.member = member
            outcome = InsertionOutcome.INSERTED
        elif ast_canonical_json(canonicalize(candidate.ast)) == ast_canonical_json(
            canonicalize(_typed_ast(previous.candidate))
        ):
            outcome = InsertionOutcome.DUPLICATE
        elif _better(member, previous):
            self.member = member
            outcome = InsertionOutcome.REPLACED
        else:
            outcome = InsertionOutcome.REJECTED
        return ArchiveDecision.create(
            task_id=self.task.task_id,
            coordinate=coordinate,
            outcome=outcome,
            candidate_id=candidate.candidate_id,
            inserted_candidate_id=(
                candidate.candidate_id
                if outcome in {InsertionOutcome.INSERTED, InsertionOutcome.REPLACED}
                else None
            ),
            replaced_candidate_id=(
                previous.candidate.candidate_id
                if previous is not None and outcome is InsertionOutcome.REPLACED
                else None
            ),
            evicted_candidate_id=None,
            role=(
                "incumbent"
                if outcome in {InsertionOutcome.INSERTED, InsertionOutcome.REPLACED}
                else None
            ),
            rank=member.rank,
        )

    def candidate_summaries(self) -> tuple[CandidateSummary, ...]:
        if self.member is None:
            return ()
        return (CandidateSummary(self.member.candidate.candidate_id, self.member.candidate.ast),)

    def branch_ids(self) -> tuple[str, ...]:
        return (f"{self.task.task_id}:incumbent",) if self.member is not None else ()
