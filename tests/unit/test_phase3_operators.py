from __future__ import annotations

from world_model_search.dsl.ast import (
    MAX_SMALL_INT,
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
    If,
    IntConst,
    IntExpr,
    Not,
    Or,
    Parity,
    PredExpr,
    TruthTable,
    Xor,
    ast_size,
    validate_ast,
)
from world_model_search.dsl.canonicalize import canonicalize
from world_model_search.dsl.interpreter import truth_table
from world_model_search.search.operators import (
    AttemptOutcome,
    OperatorId,
    apply_operator,
    replace_at,
    typed_paths,
)


def _all_constructor_parent() -> BitExpr:
    count = Count((-1, 0, 1))
    condition = Between(
        AddConst(count, MIN_SMALL_INT),
        IntConst(MIN_SMALL_INT),
        IntConst(MAX_SMALL_INT),
    )
    return If(
        condition,
        Xor(Parity((-1, 1)), Not(At(-1))),
        Or(And(At(0), At(1)), Const(0)),
    )


def test_all_operators_are_counter_deterministic_typed_bounded_and_total() -> None:
    parent = _all_constructor_parent()
    second = If(Eq(Count((-1,)), IntConst(1)), At(1), Const(0))
    limits = AstLimits(max_depth=8, max_nodes=63, max_cases=8)
    observed_types: set[type[object]] = set()
    observed_paths: set[tuple[int, ...]] = set()
    for operator in OperatorId:
        for counter in range(128):
            parents = (parent, second) if operator is OperatorId.CROSSOVER else (parent,)
            first = apply_operator(
                operator,
                parents,
                master_seed=99173,
                attempt_index=counter,
                limits=limits,
            )
            second_attempt = apply_operator(
                operator,
                parents,
                master_seed=99173,
                attempt_index=counter,
                limits=limits,
            )
            assert first == second_attempt
            assert first.outcome in set(AttemptOutcome)
            if first.canonical_ast is not None:
                validate_ast(first.canonical_ast, limits)
                assert len(truth_table(first.canonical_ast, limits=limits)) == 8
            path_map = dict(typed_paths(canonicalize(parent)))
            if first.selected_paths:
                selected = path_map[first.selected_paths[0]]
                observed_types.add(
                    BitExpr
                    if isinstance(selected, BitExpr)
                    else IntExpr
                    if isinstance(selected, IntExpr)
                    else PredExpr
                )
                observed_paths.add(first.selected_paths[0])
    assert observed_types == {BitExpr, IntExpr, PredExpr}
    assert () in observed_paths
    assert any(len(path) >= 3 for path in observed_paths)


def test_type_compatible_replacement_covers_bit_int_pred_and_boundaries() -> None:
    parent = _all_constructor_parent()
    paths = dict(typed_paths(parent))
    bit_path = next(path for path, node in paths.items() if path and isinstance(node, BitExpr))
    int_path = next(path for path, node in paths.items() if isinstance(node, IntExpr))
    pred_path = next(path for path, node in paths.items() if isinstance(node, PredExpr))
    bit_replaced = replace_at(parent, bit_path, At(1))
    int_replaced = replace_at(parent, int_path, AddConst(IntConst(MAX_SMALL_INT), MIN_SMALL_INT))
    pred_replaced = replace_at(
        parent,
        pred_path,
        Between(IntConst(MIN_SMALL_INT), IntConst(MIN_SMALL_INT), IntConst(MAX_SMALL_INT)),
    )
    for expression in (bit_replaced, int_replaced, pred_replaced):
        assert isinstance(expression, BitExpr)
        validate_ast(expression)
        assert len(truth_table(expression)) == 8
    assert ast_size(parent)[0] >= 15


def test_operators_handle_ast_at_depth_limit_and_near_node_limit() -> None:
    def dense(level: int, salt: int) -> BitExpr:
        if level == 0:
            return At((-1, 0, 1)[salt % 3])
        left = dense(level - 1, salt * 2 + 1)
        right = dense(level - 1, salt * 2 + 2)
        return (And, Or, Xor)[salt % 3](left, right)

    def deep_int(mask: tuple[int, ...]) -> IntExpr:
        result: IntExpr = Count(mask)
        for _ in range(5):
            result = AddConst(result, MAX_SMALL_INT)
        return result

    parent = If(
        Between(deep_int((-1,)), deep_int((0,)), deep_int((1,))),
        dense(4, 1),
        At(0),
    )
    limits = AstLimits(max_depth=8, max_nodes=63, max_cases=8)
    validate_ast(parent, limits)
    nodes, depth = ast_size(parent)
    assert nodes >= 48
    assert depth == limits.max_depth
    for operator in OperatorId:
        parents = (parent, parent) if operator is OperatorId.CROSSOVER else (parent,)
        for counter in range(32):
            attempt = apply_operator(
                operator,
                parents,
                master_seed=555,
                attempt_index=counter,
                limits=limits,
            )
            if attempt.canonical_ast is not None:
                validate_ast(attempt.canonical_ast, limits)


def test_truth_table_local_mutation_is_total_and_flips_exactly_one_output() -> None:
    parent = TruthTable((0, 1, 0, 1, 1, 0, 1, 0))
    limits = AstLimits(max_depth=8, max_nodes=63, max_cases=8)
    selected_outputs: set[tuple[int, ...]] = set()
    for counter in range(64):
        attempt = apply_operator(
            OperatorId.LOCAL_MUTATION,
            (parent,),
            master_seed=8841,
            attempt_index=counter,
            limits=limits,
        )
        assert attempt.outcome is AttemptOutcome.EMITTED
        assert isinstance(attempt.canonical_ast, TruthTable)
        changed = sum(
            left != right
            for left, right in zip(parent.outputs, attempt.canonical_ast.outputs, strict=True)
        )
        assert changed == 1
        selected_outputs.add(attempt.canonical_ast.outputs)
    assert len(selected_outputs) == 8
