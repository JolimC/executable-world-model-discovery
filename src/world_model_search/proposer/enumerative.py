"""Deterministic cost-ordered enumeration of well-typed canonical programs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from world_model_search.dsl.ast import (
    AddConst,
    And,
    AstLimits,
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
    Xor,
    ast_size,
)
from world_model_search.dsl.canonicalize import (
    canonicalize,
    canonicalize_int,
    canonicalize_pred,
)
from world_model_search.dsl.codec import OPCODES, encode, value_encoded_length
from world_model_search.dsl.interpreter import (
    int_truth_table,
    predicate_truth_table,
    semantic_hash,
    truth_table,
)
from world_model_search.dsl.json_schema import ast_canonical_json
from world_model_search.dsl.versions import ENUMERATOR_VERSION


@dataclass(frozen=True, slots=True)
class EnumerationBounds:
    max_bits: int = 36
    max_depth: int = 8
    max_nodes: int = 15
    max_candidates: int = 50_000

    def __post_init__(self) -> None:
        for name, value in (
            ("max_bits", self.max_bits),
            ("max_depth", self.max_depth),
            ("max_nodes", self.max_nodes),
            ("max_candidates", self.max_candidates),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_ENUMERATION_BOUNDS = EnumerationBounds()


@dataclass(frozen=True, slots=True)
class EnumeratedProgram:
    discovery_index: int
    ast: BitExpr
    ast_bits: int
    ordered_semantics: tuple[int, ...]
    semantic_hash: str


@dataclass(frozen=True, slots=True)
class CollapseRecord:
    reason: str
    source_json: str
    canonical_json: str
    retained_json: str


@dataclass(frozen=True, slots=True)
class EnumerationResult:
    version: str
    bounds: EnumerationBounds
    programs: tuple[EnumeratedProgram, ...]
    candidates_examined: int
    canonical_duplicates: int
    semantic_duplicates: int
    collapse_records: tuple[CollapseRecord, ...]
    stopped_at_candidate_bound: bool


type Signature = tuple[int, ...] | tuple[bool, ...]


def _all_masks() -> tuple[tuple[int, ...], ...]:
    offsets = (-1, 0, 1)
    return tuple(mask for size in range(1, 4) for mask in combinations(offsets, size))


class _Enumerator:
    def __init__(self, bounds: EnumerationBounds) -> None:
        self.bounds = bounds
        self.limits = AstLimits(
            max_depth=bounds.max_depth,
            max_nodes=bounds.max_nodes,
            max_cases=8,
        )
        self.bit_by_cost: dict[int, list[BitExpr]] = {}
        self.int_by_cost: dict[int, list[IntExpr]] = {}
        self.pred_by_cost: dict[int, list[PredExpr]] = {}
        self.seen_json: set[tuple[str, str]] = set()
        self.seen_semantics: dict[str, dict[Signature, BitExpr | IntExpr | PredExpr]] = {
            "bit": {},
            "int": {},
            "pred": {},
        }
        self.programs: list[EnumeratedProgram] = []
        self.collapses: list[CollapseRecord] = []
        self.examined = 0
        self.canonical_duplicates = 0
        self.semantic_duplicates = 0
        self.stopped = False

    def _bounded(self, expr: BitExpr | IntExpr | PredExpr) -> bool:
        nodes, depth = ast_size(expr)
        return nodes <= self.bounds.max_nodes and depth <= self.bounds.max_depth

    def _process(
        self,
        kind: str,
        cost: int,
        raw_values: Sequence[BitExpr | IntExpr | PredExpr],
    ) -> None:
        for raw in sorted(raw_values, key=ast_canonical_json):
            if self.examined >= self.bounds.max_candidates:
                self.stopped = True
                return
            self.examined += 1
            if kind == "bit":
                if not isinstance(raw, BitExpr):
                    raise AssertionError("bit pool type mismatch")
                canonical: BitExpr | IntExpr | PredExpr = canonicalize(raw)
            elif kind == "int":
                if not isinstance(raw, IntExpr):
                    raise AssertionError("int pool type mismatch")
                canonical = canonicalize_int(raw)
            else:
                if not isinstance(raw, PredExpr):
                    raise AssertionError("predicate pool type mismatch")
                canonical = canonicalize_pred(raw)
            if not self._bounded(canonical) or value_encoded_length(canonical) != cost:
                self.canonical_duplicates += 1
                if len(self.collapses) < 24:
                    self.collapses.append(
                        CollapseRecord(
                            reason="canonical-rewrite",
                            source_json=ast_canonical_json(raw),
                            canonical_json=ast_canonical_json(canonical),
                            retained_json=ast_canonical_json(canonical),
                        )
                    )
                continue
            encoded = ast_canonical_json(canonical)
            identity = (kind, encoded)
            if identity in self.seen_json:
                self.canonical_duplicates += 1
                continue
            self.seen_json.add(identity)
            if isinstance(canonical, BitExpr):
                signature: Signature = truth_table(canonical, limits=self.limits)
            elif isinstance(canonical, IntExpr):
                signature = int_truth_table(canonical)
            else:
                signature = predicate_truth_table(canonical)
            retained = self.seen_semantics[kind].get(signature)
            if retained is not None:
                self.semantic_duplicates += 1
                if len(self.collapses) < 24:
                    self.collapses.append(
                        CollapseRecord(
                            reason="semantic-duplicate",
                            source_json=encoded,
                            canonical_json=encoded,
                            retained_json=ast_canonical_json(retained),
                        )
                    )
                continue
            self.seen_semantics[kind][signature] = canonical
            if isinstance(canonical, BitExpr):
                self.bit_by_cost.setdefault(cost, []).append(canonical)
                table = truth_table(canonical, limits=self.limits)
                self.programs.append(
                    EnumeratedProgram(
                        discovery_index=len(self.programs),
                        ast=canonical,
                        ast_bits=len(encode(canonical)),
                        ordered_semantics=table,
                        semantic_hash=semantic_hash(canonical, limits=self.limits),
                    )
                )
            elif isinstance(canonical, IntExpr):
                self.int_by_cost.setdefault(cost, []).append(canonical)
            else:
                self.pred_by_cost.setdefault(cost, []).append(canonical)

    def _int_candidates(self, cost: int) -> list[IntExpr]:
        result: list[IntExpr] = []
        for value in range(-3, 4):
            int_node = IntConst(value)
            if value_encoded_length(int_node) == cost:
                result.append(int_node)
        for mask in _all_masks():
            count_node = Count(mask)
            if value_encoded_length(count_node) == cost:
                result.append(count_node)
        overhead = len(OPCODES["AddConst"]) + 3
        for child in self.int_by_cost.get(cost - overhead, ()):
            for amount in range(-3, 4):
                result.append(AddConst(child, amount))
        return result

    def _pred_candidates(self, cost: int) -> list[PredExpr]:
        result: list[PredExpr] = []
        for name, constructor in (("Eq", Eq), ("Le", Le), ("Ge", Ge)):
            overhead = len(OPCODES[name])
            for left_cost in range(1, cost):
                right_cost = cost - overhead - left_cost
                for left in self.int_by_cost.get(left_cost, ()):
                    for right in self.int_by_cost.get(right_cost, ()):
                        result.append(constructor(left, right))
        overhead = len(OPCODES["Between"])
        for first_cost in range(1, cost):
            for second_cost in range(1, cost - first_cost):
                third_cost = cost - overhead - first_cost - second_cost
                for value in self.int_by_cost.get(first_cost, ()):
                    for lower in self.int_by_cost.get(second_cost, ()):
                        for upper in self.int_by_cost.get(third_cost, ()):
                            result.append(Between(value, lower, upper))
        return result

    def _bit_candidates(self, cost: int) -> list[BitExpr]:
        result: list[BitExpr] = []
        leaves: list[BitExpr] = [Const(0), Const(1), At(-1), At(0), At(1)]
        leaves.extend(Parity(mask) for mask in _all_masks())
        leaves.extend(Majority(mask) for mask in _all_masks())
        result.extend(node for node in leaves if value_encoded_length(node) == cost)
        not_overhead = len(OPCODES["Not"])
        result.extend(Not(child) for child in self.bit_by_cost.get(cost - not_overhead, ()))
        for name, constructor in (("And", And), ("Or", Or), ("Xor", Xor)):
            overhead = len(OPCODES[name])
            for left_cost in range(1, cost):
                right_cost = cost - overhead - left_cost
                for left in self.bit_by_cost.get(left_cost, ()):
                    for right in self.bit_by_cost.get(right_cost, ()):
                        result.append(constructor(left, right))
        overhead = len(OPCODES["If"])
        for pred_cost in range(1, cost):
            for then_cost in range(1, cost - pred_cost):
                else_cost = cost - overhead - pred_cost - then_cost
                for condition in self.pred_by_cost.get(pred_cost, ()):
                    for then_branch in self.bit_by_cost.get(then_cost, ()):
                        for else_branch in self.bit_by_cost.get(else_cost, ()):
                            result.append(If(condition, then_branch, else_branch))
        return result

    def run(self) -> EnumerationResult:
        for cost in range(1, self.bounds.max_bits + 1):
            self._process("int", cost, self._int_candidates(cost))
            if self.stopped:
                break
            self._process("pred", cost, self._pred_candidates(cost))
            if self.stopped:
                break
            self._process("bit", cost, self._bit_candidates(cost))
            if self.stopped:
                break
        return EnumerationResult(
            version=ENUMERATOR_VERSION,
            bounds=self.bounds,
            programs=tuple(self.programs),
            candidates_examined=self.examined,
            canonical_duplicates=self.canonical_duplicates,
            semantic_duplicates=self.semantic_duplicates,
            collapse_records=tuple(self.collapses),
            stopped_at_candidate_bound=self.stopped,
        )


def enumerate_programs(
    bounds: EnumerationBounds = DEFAULT_ENUMERATION_BOUNDS,
) -> EnumerationResult:
    """Enumerate without targets, randomness, truth-table literals, or hidden data."""

    return _Enumerator(bounds).run()
