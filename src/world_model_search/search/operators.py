"""Deterministic, type-compatible Phase 3 mutation and crossover operators."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from world_model_search.domain.types import ProposalBudget, ProposalContext
from world_model_search.dsl.ast import (
    MIN_SMALL_INT,
    AddConst,
    And,
    AstLimits,
    At,
    Between,
    BitExpr,
    Const,
    Count,
    Eq,
    Expr,
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
    ast_size,
    children,
    validate_ast,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.codec import encode
from world_model_search.dsl.interpreter import truth_table
from world_model_search.dsl.json_schema import DslCandidateDocument, ast_canonical_json
from world_model_search.dsl.versions import PHASE3_OPERATOR_VERSION, PHASE3_RNG_VERSION
from world_model_search.serialization import JsonObject, sha256_text

type AstPath = tuple[int, ...]


class OperatorId(StrEnum):
    LOCAL_MUTATION = "local-mutation"
    SUBTREE_REPLACEMENT = "subtree-replacement"
    SIMPLIFICATION = "simplification"
    CROSSOVER = "typed-crossover"


class AttemptOutcome(StrEnum):
    EMITTED = "emitted"
    NO_OP = "no-op"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    operator_id: OperatorId
    weight: int
    arity: int
    precondition: str


DEFAULT_OPERATOR_INVENTORY = (
    OperatorSpec(OperatorId.LOCAL_MUTATION, 4, 1, "one typed parent"),
    OperatorSpec(OperatorId.SUBTREE_REPLACEMENT, 3, 1, "one typed parent"),
    OperatorSpec(OperatorId.SIMPLIFICATION, 2, 1, "one typed parent"),
    OperatorSpec(OperatorId.CROSSOVER, 3, 2, "two parents; self-crossover fallback"),
)


@dataclass(frozen=True, slots=True)
class OperatorAttempt:
    operator_id: OperatorId
    operator_version: str
    outcome: AttemptOutcome
    source_ast: BitExpr | None
    canonical_ast: BitExpr | None
    selected_paths: tuple[AstPath, ...]
    choices: JsonObject
    rejection_reason: str | None
    crossover_arity: int


class CounterRng:
    """Stateless SHA-256 decisions domain-separated by stream and counter."""

    def __init__(self, master_seed: int, stream: str, counter: int) -> None:
        self.master_seed = master_seed
        self.stream = stream
        self.counter = counter

    def integer(self, domain: str, upper: int, *, subcounter: int = 0) -> int:
        if upper < 1:
            raise ValueError("upper must be positive")
        material = (
            f"{PHASE3_RNG_VERSION}\0{self.master_seed}\0{self.stream}\0"
            f"{self.counter}\0{domain}\0{subcounter}"
        ).encode()
        return int.from_bytes(hashlib.sha256(material).digest(), "big") % upper

    def weighted_index(self, domain: str, weights: tuple[int, ...]) -> int:
        if not weights or any(weight < 1 for weight in weights):
            raise ValueError("weights must be positive")
        draw = self.integer(domain, sum(weights))
        running = 0
        for index, weight in enumerate(weights):
            running += weight
            if draw < running:
                return index
        raise AssertionError("weighted decision did not select an index")


class MutationProposer:
    """Typed Phase 3 proposer behind the same public proposal capability boundary."""

    proposer_id = "mutation"

    def propose(
        self, context: ProposalContext, budget: ProposalBudget
    ) -> tuple[OperatorAttempt, ...]:
        if budget.max_candidates != 1:
            raise ValueError("the deterministic mutation proposer requires a batch of one")
        if budget.operator_id is None:
            raise ValueError("the mutation proposer requires a declared operator")
        try:
            operator_id = OperatorId(budget.operator_id)
        except ValueError as exc:
            raise ValueError("the mutation proposer received an unknown operator") from exc
        parents = tuple(parent.ast for parent in context.parents)
        if not all(isinstance(parent, BitExpr) for parent in parents):
            raise TypeError("the mutation proposer accepts only typed DSL parents")
        required = 2 if operator_id is OperatorId.CROSSOVER else 1
        if len(parents) != required:
            raise ValueError(f"{operator_id.value} requires {required} ordered parents")
        world = context.task.public_world_spec
        max_depth = getattr(world, "max_depth", None)
        max_nodes = getattr(world, "max_nodes", None)
        if not isinstance(max_depth, int) or not isinstance(max_nodes, int):
            raise TypeError("the mutation proposer requires the bounded elementary world contract")
        limits = AstLimits(max_depth=max_depth, max_nodes=max_nodes, max_cases=8)
        return (
            apply_operator(
                operator_id,
                tuple(parent for parent in parents if isinstance(parent, BitExpr)),
                master_seed=budget.proposer_seed,
                attempt_index=budget.start_index,
                limits=limits,
            ),
        )


def typed_paths(expr: Expr) -> tuple[tuple[AstPath, Expr], ...]:
    """Preorder paths with child indices in the frozen ``children`` ordering."""

    result: list[tuple[AstPath, Expr]] = []

    def visit(node: Expr, path: AstPath) -> None:
        result.append((path, node))
        for index, child in enumerate(children(node)):
            visit(child, (*path, index))

    visit(expr, ())
    return tuple(result)


def _rebuild(expr: Expr, new_children: tuple[Expr, ...]) -> Expr:
    if isinstance(expr, Not):
        return Not(_bit(new_children[0]))
    if isinstance(expr, And | Or | Xor):
        left, right = _bit(new_children[0]), _bit(new_children[1])
        if isinstance(expr, And):
            return And(left, right)
        if isinstance(expr, Or):
            return Or(left, right)
        return Xor(left, right)
    if isinstance(expr, If):
        return If(_pred(new_children[0]), _bit(new_children[1]), _bit(new_children[2]))
    if isinstance(expr, AddConst):
        return AddConst(_int(new_children[0]), expr.amount)
    if isinstance(expr, Eq | Le | Ge):
        int_left, int_right = _int(new_children[0]), _int(new_children[1])
        if isinstance(expr, Eq):
            return Eq(int_left, int_right)
        if isinstance(expr, Le):
            return Le(int_left, int_right)
        return Ge(int_left, int_right)
    if isinstance(expr, Between):
        return Between(_int(new_children[0]), _int(new_children[1]), _int(new_children[2]))
    if new_children:
        raise TypeError("leaf cannot receive children")
    return expr


def replace_at(expr: Expr, path: AstPath, replacement: Expr) -> Expr:
    if not path:
        if not _same_type(expr, replacement):
            raise TypeError("replacement type differs from selected subtree")
        return replacement
    child_index = path[0]
    old_children = children(expr)
    if child_index >= len(old_children):
        raise ValueError("AST path is outside the tree")
    updated = list(old_children)
    updated[child_index] = replace_at(updated[child_index], path[1:], replacement)
    return _rebuild(expr, tuple(updated))


def _same_type(left: Expr, right: Expr) -> bool:
    return (
        (isinstance(left, BitExpr) and isinstance(right, BitExpr))
        or (isinstance(left, IntExpr) and isinstance(right, IntExpr))
        or (isinstance(left, PredExpr) and isinstance(right, PredExpr))
    )


def _bit(expr: Expr) -> BitExpr:
    if not isinstance(expr, BitExpr):
        raise TypeError("expected BitExpr")
    return expr


def _int(expr: Expr) -> IntExpr:
    if not isinstance(expr, IntExpr):
        raise TypeError("expected IntExpr")
    return expr


def _pred(expr: Expr) -> PredExpr:
    if not isinstance(expr, PredExpr):
        raise TypeError("expected PredExpr")
    return expr


_MASKS: tuple[tuple[int, ...], ...] = (
    (-1,),
    (0,),
    (1,),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (-1, 0, 1),
)


def _generated_expr(
    kind: type[BitExpr] | type[IntExpr] | type[PredExpr],
    rng: CounterRng,
    domain: str,
    depth_budget: int,
) -> Expr:
    """Generate bounded syntax; every constructor is reachable by a counter choice."""

    depth_budget = max(1, depth_budget)
    if kind is BitExpr:
        leaf = (
            Const(rng.integer(domain + ":const", 2)),
            At((-1, 0, 1)[rng.integer(domain + ":at", 3)]),
        )
        if depth_budget == 1:
            return leaf[rng.integer(domain + ":leaf", len(leaf))]
        choice = rng.integer(domain + ":bit-op", 10)
        if choice < 2:
            return leaf[choice]
        if choice == 2:
            return Not(_bit(_generated_expr(BitExpr, rng, domain + ":not", depth_budget - 1)))
        if choice in (3, 4, 5):
            left = _bit(_generated_expr(BitExpr, rng, domain + ":left", depth_budget - 1))
            right = _bit(_generated_expr(BitExpr, rng, domain + ":right", depth_budget - 1))
            return (And, Or, Xor)[choice - 3](left, right)
        if choice == 6:
            return If(
                _pred(_generated_expr(PredExpr, rng, domain + ":condition", depth_budget - 1)),
                _bit(_generated_expr(BitExpr, rng, domain + ":then", depth_budget - 1)),
                _bit(_generated_expr(BitExpr, rng, domain + ":else", depth_budget - 1)),
            )
        mask = _MASKS[rng.integer(domain + ":mask", len(_MASKS))]
        return Parity(mask) if choice in (7, 9) else Majority(mask)
    if kind is IntExpr:
        choice = rng.integer(domain + ":int-op", 3 if depth_budget > 1 else 2)
        if choice == 0:
            return IntConst(MIN_SMALL_INT + rng.integer(domain + ":int", 7))
        if choice == 1:
            return Count(_MASKS[rng.integer(domain + ":mask", len(_MASKS))])
        return AddConst(
            _int(_generated_expr(IntExpr, rng, domain + ":child", depth_budget - 1)),
            MIN_SMALL_INT + rng.integer(domain + ":amount", 7),
        )
    choice = rng.integer(domain + ":pred-op", 4)
    int_left = _int(_generated_expr(IntExpr, rng, domain + ":left", max(1, depth_budget - 1)))
    int_right = _int(_generated_expr(IntExpr, rng, domain + ":right", max(1, depth_budget - 1)))
    if choice < 3:
        if choice == 0:
            return Eq(int_left, int_right)
        if choice == 1:
            return Le(int_left, int_right)
        return Ge(int_left, int_right)
    return Between(
        int_left,
        int_right,
        _int(_generated_expr(IntExpr, rng, domain + ":upper", max(1, depth_budget - 1))),
    )


def _local_edit(node: Expr, rng: CounterRng) -> Expr:
    if isinstance(node, Const):
        return Const(1 - node.value)
    if isinstance(node, At):
        choices = tuple(offset for offset in (-1, 0, 1) if offset != node.offset)
        return At(choices[rng.integer("mutate-at", len(choices))])
    if isinstance(node, Not):
        return node.expr
    if isinstance(node, And | Or | Xor):
        choice = rng.integer("mutate-binary", 2)
        if isinstance(node, And):
            return (Or(node.left, node.right), Xor(node.left, node.right))[choice]
        if isinstance(node, Or):
            return (And(node.left, node.right), Xor(node.left, node.right))[choice]
        return (And(node.left, node.right), Or(node.left, node.right))[choice]
    if isinstance(node, If):
        return If(node.condition, node.else_branch, node.then_branch)
    if isinstance(node, Parity | Majority):
        if isinstance(node, Parity):
            return Majority(node.mask)
        return Parity(node.mask)
    if isinstance(node, TruthTable):
        selected = rng.integer("mutate-truth-table-output", len(node.outputs))
        outputs = list(node.outputs)
        outputs[selected] = 1 - outputs[selected]
        return TruthTable(tuple(outputs))
    if isinstance(node, IntConst):
        return IntConst(MIN_SMALL_INT + ((node.value - MIN_SMALL_INT + 1) % 7))
    if isinstance(node, Count):
        mask_choices = tuple(mask for mask in _MASKS if mask != node.mask)
        return Count(mask_choices[rng.integer("mutate-count", len(mask_choices))])
    if isinstance(node, AddConst):
        amount = MIN_SMALL_INT + ((node.amount - MIN_SMALL_INT + 1) % 7)
        return AddConst(node.expr, amount)
    if isinstance(node, Eq | Le | Ge):
        choice = rng.integer("mutate-predicate", 2)
        if isinstance(node, Eq):
            return (Le(node.left, node.right), Ge(node.left, node.right))[choice]
        if isinstance(node, Le):
            return (Eq(node.left, node.right), Ge(node.left, node.right))[choice]
        return (Eq(node.left, node.right), Le(node.left, node.right))[choice]
    if isinstance(node, Between):
        return Between(node.value, node.upper, node.lower)
    raise TypeError(f"unsupported local edit: {type(node).__name__}")


def _simplification(node: Expr, rng: CounterRng) -> Expr:
    child_nodes = children(node)
    compatible = tuple(child for child in child_nodes if _same_type(node, child))
    if compatible:
        return compatible[rng.integer("simplify-child", len(compatible))]
    if isinstance(node, BitExpr):
        return Const(rng.integer("simplify-bit", 2))
    if isinstance(node, IntExpr):
        return IntConst(0)
    return Eq(IntConst(0), IntConst(0))


def _finalize(
    operator_id: OperatorId,
    source: BitExpr,
    paths: tuple[AstPath, ...],
    choices: JsonObject,
    limits: AstLimits,
) -> OperatorAttempt:
    try:
        canonical = canonicalize(source)
        validate_ast(canonical, limits)
        document = DslCandidateDocument(ast=canonical)
        parsed = DslCandidateDocument.from_json(document.to_json(), limits=limits)
        if parsed.ast != canonical or truth_table(canonical, limits=limits) != truth_table(
            parsed.ast, limits=limits
        ):
            raise ValueError("parser/interpreter round trip diverged")
        encode(canonical)
    except (TypeError, ValueError) as exc:
        return OperatorAttempt(
            operator_id=operator_id,
            operator_version=PHASE3_OPERATOR_VERSION,
            outcome=AttemptOutcome.REJECTED,
            source_ast=source,
            canonical_ast=None,
            selected_paths=paths,
            choices=choices,
            rejection_reason=str(exc),
            crossover_arity=2 if operator_id is OperatorId.CROSSOVER else 1,
        )
    outcome = (
        AttemptOutcome.NO_OP
        if choices.get("parent_canonical_hash") == sha256_text(ast_canonical_json(canonical))
        else AttemptOutcome.EMITTED
    )
    return OperatorAttempt(
        operator_id=operator_id,
        operator_version=PHASE3_OPERATOR_VERSION,
        outcome=outcome,
        source_ast=source,
        canonical_ast=canonical,
        selected_paths=paths,
        choices=choices,
        rejection_reason=None,
        crossover_arity=2 if operator_id is OperatorId.CROSSOVER else 1,
    )


def apply_operator(
    operator_id: OperatorId,
    parents: tuple[BitExpr, ...],
    *,
    master_seed: int,
    attempt_index: int,
    limits: AstLimits,
) -> OperatorAttempt:
    """Apply one charged operator attempt using only parents and declared RNG inputs."""

    required = 2 if operator_id is OperatorId.CROSSOVER else 1
    if len(parents) != required:
        raise ValueError(f"{operator_id.value} requires {required} ordered parents")
    first = canonicalize(parents[0])
    parent_hash = sha256_text(ast_canonical_json(first))
    path_rng = CounterRng(master_seed, "subtree-path", attempt_index)
    syntax_rng = CounterRng(master_seed, "replacement-syntax", attempt_index)
    first_paths = typed_paths(first)
    if operator_id is OperatorId.CROSSOVER:
        second = canonicalize(parents[1])
        compatible_pairs = tuple(
            (left_path, right_path, right_node)
            for left_path, left_node in first_paths
            for right_path, right_node in typed_paths(second)
            if _same_type(left_node, right_node)
        )
        selected = compatible_pairs[path_rng.integer("crossover-pair", len(compatible_pairs))]
        left_path, right_path, replacement = selected
        source = _bit(replace_at(first, left_path, replacement))
        return _finalize(
            operator_id,
            source,
            (left_path, right_path),
            {
                "parent_canonical_hash": parent_hash,
                "path_ordering": "preorder-child-index-v1",
                "pair_count": len(compatible_pairs),
            },
            limits,
        )
    selected_path, selected_node = first_paths[path_rng.integer("primary-path", len(first_paths))]
    if operator_id is OperatorId.LOCAL_MUTATION:
        replacement = _local_edit(selected_node, syntax_rng)
    elif operator_id is OperatorId.SUBTREE_REPLACEMENT:
        remaining_depth = max(1, limits.max_depth - len(selected_path))
        kind: type[BitExpr] | type[IntExpr] | type[PredExpr]
        if isinstance(selected_node, BitExpr):
            kind = BitExpr
        elif isinstance(selected_node, IntExpr):
            kind = IntExpr
        else:
            kind = PredExpr
        replacement = _generated_expr(kind, syntax_rng, "replacement", min(3, remaining_depth))
    elif operator_id is OperatorId.SIMPLIFICATION:
        replacement = _simplification(selected_node, syntax_rng)
    else:  # pragma: no cover - exhaustive enum
        raise AssertionError(operator_id)
    source = _bit(replace_at(first, selected_path, replacement))
    choices: JsonObject = {
        "parent_canonical_hash": parent_hash,
        "path_ordering": "preorder-child-index-v1",
        "available_path_count": len(first_paths),
        "source_nodes": ast_size(source)[0],
    }
    return _finalize(operator_id, source, (selected_path,), choices, limits)
