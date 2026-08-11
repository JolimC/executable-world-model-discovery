"""Fixed self-contained public prompt templates for Phase 4."""

from __future__ import annotations

import json
from dataclasses import dataclass

from world_model_search.domain.types import CandidateSummary, ProposalRole, PublicTask
from world_model_search.dsl.ast import BitExpr
from world_model_search.dsl.json_schema import ast_to_value
from world_model_search.serialization import JsonObject, canonical_json, sha256_text

DIRECT_PROMPT_VERSION = "phase4-direct-public-task-v1"
ITERATIVE_PROMPT_VERSION = "phase4-iterative-parent-score-v1"
FEEDBACK_SCHEMA_VERSION = "phase4-parent-score-feedback-v1"


@dataclass(frozen=True, slots=True)
class ParentScoreFeedback:
    """Deterministic score-only feedback safe for an iterative proposer."""

    candidate_id: str
    type_valid: bool
    total: bool
    local_errors: int
    local_cases: int
    exact: bool
    ast_bits: int
    residual_bits: int
    two_part_bits: int

    def __post_init__(self) -> None:
        values = (
            self.local_errors,
            self.local_cases,
            self.ast_bits,
            self.residual_bits,
            self.two_part_bits,
        )
        if (
            len(self.candidate_id) != 64
            or set(self.candidate_id) - set("0123456789abcdef")
            or not all(
                isinstance(value, bool) for value in (self.type_valid, self.total, self.exact)
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
        ):
            raise ValueError("parent feedback fields are malformed")
        if self.local_cases != 8 or self.local_errors > self.local_cases:
            raise ValueError("parent score must contain the bounded eight-case error count")
        if max(self.ast_bits, self.residual_bits, self.two_part_bits) > 1_000_000:
            raise ValueError("parent score coding fields exceed their public bound")
        if self.two_part_bits != self.ast_bits + self.residual_bits:
            raise ValueError("parent two-part bits do not reconcile")

    def to_value(self) -> JsonObject:
        return {
            "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "type_valid": self.type_valid,
            "total": self.total,
            "local_errors": self.local_errors,
            "local_cases": self.local_cases,
            "exact": self.exact,
            "ast_bits": self.ast_bits,
            "residual_bits": self.residual_bits,
            "two_part_bits": self.two_part_bits,
        }


def _public_task(task: PublicTask) -> JsonObject:
    value: object = json.loads(canonical_json(task))
    if not isinstance(value, dict):
        raise TypeError("public task must serialize as an object")
    return value


def _grammar() -> JsonObject:
    return {
        "dsl_version": "binary-ca-radius1-dsl-v1",
        "candidate_schema_version": 1,
        "root_type": "BitExpr",
        "bit_expr": {
            "Const": {"value": "integer 0 or 1"},
            "At": {"offset": "-1, 0, or 1"},
            "Not": {"expr": "BitExpr"},
            "And|Or|Xor": {"left": "BitExpr", "right": "BitExpr"},
            "If": {"condition": "PredExpr", "then": "BitExpr", "else": "BitExpr"},
            "Parity|Majority": {"mask": "nonempty sorted unique subset of [-1,0,1]"},
        },
        "int_expr": {
            "IntConst": {"value": "integer -3 through 3"},
            "Count": {"mask": "nonempty sorted unique subset of [-1,0,1]"},
            "AddConst": {"expr": "IntExpr", "amount": "integer -3 through 3"},
        },
        "pred_expr": {
            "Eq|Le|Ge": {"left": "IntExpr", "right": "IntExpr"},
            "Between": {"value": "IntExpr", "lower": "IntExpr", "upper": "IntExpr"},
        },
        "bounds": {"maximum_depth": 8, "maximum_nodes": 63, "exhaustive_cases": 8},
        "forbidden": [
            "TruthTable",
            "Python or source code",
            "unknown fields or opcodes",
            "prose or markdown",
        ],
    }


def render_prompt(
    *,
    task: PublicTask,
    role: ProposalRole,
    requested_batch_size: int,
    parent: CandidateSummary | None = None,
    feedback: ParentScoreFeedback | None = None,
) -> tuple[str, str, str]:
    """Return template ID, version, and exact canonical public input bytes."""

    if requested_batch_size < 1:
        raise ValueError("requested batch size must be positive")
    iterative = parent is not None or feedback is not None
    if (parent is None) != (feedback is None):
        raise ValueError("iterative prompt requires one parent and its feedback")
    if parent is not None and feedback is not None and parent.candidate_id != feedback.candidate_id:
        raise ValueError("parent feedback is associated with a different candidate")
    if parent is not None and not isinstance(parent.ast, BitExpr):
        raise TypeError("Phase 4 prompts accept only typed DSL parents")
    payload: JsonObject = {
        "prompt_contract": {
            "stateless": True,
            "complete_candidate_documents_only": True,
            "response_order_is_evaluation_order": True,
            "role": role.value,
            "requested_batch_size": requested_batch_size,
        },
        "public_task": _public_task(task),
        "dsl_grammar": _grammar(),
        "instruction": (
            "Return exactly the requested JSON candidate batch. Each candidate is a complete AST "
            "document. Do not include explanations, markdown, source code, hidden-data guesses, "
            "or a mutation operation."
        ),
    }
    if iterative:
        assert parent is not None and feedback is not None
        if not isinstance(parent.ast, BitExpr):
            raise TypeError("iterative Phase 4 parent must use the typed DSL")
        payload["selected_parent"] = {
            "candidate_id": parent.candidate_id,
            "ast": ast_to_value(parent.ast),
        }
        payload["selected_parent_score"] = feedback.to_value()
        template = "iterative"
        version = ITERATIVE_PROMPT_VERSION
    else:
        payload["independence_contract"] = {
            "parents_exposed": 0,
            "prior_candidates_exposed": 0,
            "oracle_scores_exposed": 0,
            "previous_model_outputs_exposed": 0,
        }
        template = "direct"
        version = DIRECT_PROMPT_VERSION
    rendered = canonical_json(payload)
    return template, version, rendered


def prompt_context_hash(rendered_input: str) -> str:
    return sha256_text(rendered_input)
