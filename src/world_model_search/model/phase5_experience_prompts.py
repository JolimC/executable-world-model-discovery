"""Strict LLM prompts for lineage lesson induction and cell-conditioned search."""

from __future__ import annotations

import json
from typing import cast

from world_model_search.memory.experience import (
    EXPERIENCE_LESSON_VERSION,
    ExperienceLessonProposal,
    ExperienceRetrievalRecord,
    SuccessfulLineageEvidence,
)
from world_model_search.search.archive import RepresentationFamily
from world_model_search.serialization import JsonObject, canonical_json

LESSON_INDUCTION_PROMPT_VERSION = "phase5-lineage-lesson-induction-v2"
EXPERIENCE_SEARCH_PROMPT_VERSION = "phase5-cell-conditioned-search-v2"


def lesson_induction_json_schema(
    *, requested_lessons: int, representation_family: RepresentationFamily
) -> JsonObject:
    if requested_lessons < 1:
        raise ValueError("requested lesson count must be positive")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["lesson_batch_version", "lessons"],
        "properties": {
            "lesson_batch_version": {"type": "integer", "const": 1},
            "lessons": {
                "type": "array",
                "minItems": requested_lessons,
                "maxItems": requested_lessons,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "lesson_text",
                        "archive_representation_family",
                        "source_evidence_ids",
                    ],
                    "properties": {
                        "lesson_text": {"type": "string"},
                        "archive_representation_family": {
                            "type": "string",
                            "const": representation_family.value,
                        },
                        "source_evidence_ids": {
                            "type": "array",
                            "minItems": 2,
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


def render_lesson_induction_prompt(
    *,
    evidence: tuple[SuccessfulLineageEvidence, ...],
    requested_lessons: int,
    representation_family: RepresentationFamily,
) -> str:
    if requested_lessons < 1 or not evidence:
        raise ValueError("lesson induction needs evidence and a positive lesson count")
    if any(item.role.value != "training" for item in evidence):
        raise ValueError("lesson induction accepts only training lineages")
    if any(item.archive_representation_family is not representation_family for item in evidence):
        raise ValueError("one lesson-induction request may contain only one representation family")
    if len({item.task_id for item in evidence}) < 2:
        raise ValueError("lesson induction requires evidence from at least two tasks")
    payload: JsonObject = cast(
        JsonObject,
        {
            "prompt_version": LESSON_INDUCTION_PROMPT_VERSION,
            "task": (
                "Infer concise reusable search lessons from contrasts between recorded successful "
                "lineages and their non-exact siblings. Every contrast came from one model request "
                "and the same selected parent. Do not infer hidden rules, reference programs, or "
                "task-family labels. Scope every lesson to the selected parent cell's declared "
                "public archive representation family and cite every supporting evidence ID."
            ),
            "archive_representation_family": representation_family.value,
            "requested_lessons": requested_lessons,
            "matched_lineage_contrasts": [item.proposer_value() for item in evidence],
            "response_contract": {
                "strict_json_schema": True,
                "lesson_version": EXPERIENCE_LESSON_VERSION,
                "one_request_one_representation_family": True,
                "compare_success_with_matched_failures": True,
                "generalize_from_consequential_revision_not_terminal_ast_alone": True,
            },
        },
    )
    rendered = canonical_json(payload)
    for forbidden in ("generator_family_id", "reference_ast", "semantic_hash", "oracle_handle"):
        if forbidden in rendered:
            raise ValueError("lesson-induction prompt contains evaluator-only data")
    return rendered


def parse_lesson_induction_response(
    data: str,
    *,
    evidence_catalog: dict[str, SuccessfulLineageEvidence],
    requested_lessons: int,
    representation_family: RepresentationFamily,
) -> tuple[ExperienceLessonProposal, ...]:
    try:
        value: object = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError("lesson induction response is not JSON") from exc
    if not isinstance(value, dict) or set(value) != {"lesson_batch_version", "lessons"}:
        raise ValueError("lesson induction response has missing or unknown fields")
    lessons = value.get("lessons")
    if value.get("lesson_batch_version") != 1 or not isinstance(lessons, list):
        raise ValueError("lesson induction response version or lessons are invalid")
    if len(lessons) != requested_lessons:
        raise ValueError("lesson induction response count differs from the request")
    proposals: list[ExperienceLessonProposal] = []
    for raw in lessons:
        if not isinstance(raw, dict) or set(raw) != {
            "lesson_text",
            "archive_representation_family",
            "source_evidence_ids",
        }:
            raise ValueError("lesson induction item has missing or unknown fields")
        evidence_ids = raw["source_evidence_ids"]
        if (
            not isinstance(evidence_ids, list)
            or len(evidence_ids) < 2
            or len(set(evidence_ids)) != len(evidence_ids)
            or any(not isinstance(item, str) for item in evidence_ids)
        ):
            raise ValueError("lesson induction evidence IDs are malformed")
        proposal = ExperienceLessonProposal(
            lesson_text=str(raw["lesson_text"]),
            archive_representation_family=RepresentationFamily(
                str(raw["archive_representation_family"])
            ),
            source_evidence_ids=tuple(sorted(evidence_ids)),
        )
        if proposal.archive_representation_family is not representation_family:
            raise ValueError(
                "lesson induction response changed the request's representation family"
            )
        sources = [evidence_catalog.get(item) for item in proposal.source_evidence_ids]
        if any(source is None for source in sources):
            raise ValueError("lesson induction cites unavailable evidence")
        if any(
            source is not None
            and source.archive_representation_family is not proposal.archive_representation_family
            for source in sources
        ):
            raise ValueError("lesson induction crosses archive representation families")
        proposals.append(proposal)
    if len({proposal.proposal_id for proposal in proposals}) != len(proposals):
        raise ValueError("lesson induction returned duplicate lessons")
    return tuple(proposals)


def inject_experience_memory(*, base_prompt: str, retrieval: ExperienceRetrievalRecord) -> str:
    """Insert one recorded block into an existing public iterative-search prompt."""

    try:
        payload: object = json.loads(base_prompt)
        memory: object = json.loads(retrieval.rendered_memory)
    except json.JSONDecodeError as exc:
        raise ValueError("experience prompt inputs must be JSON") from exc
    if not isinstance(payload, dict) or not isinstance(memory, dict):
        raise ValueError("experience prompt inputs must be JSON objects")
    if "experience_memory_block" in payload or "experience_contract" in payload:
        raise ValueError("base prompt already contains experience memory")
    payload["experience_contract"] = {
        "version": EXPERIENCE_SEARCH_PROMPT_VERSION,
        "selected_by_public_archive_representation_family": True,
        "lesson_is_advice_not_hidden_task_truth": True,
    }
    payload["experience_memory_block"] = memory
    return canonical_json(payload)


def assert_experience_prompt_isolation(memory_off: str, memory_on: str) -> None:
    try:
        off: object = json.loads(memory_off)
        on: object = json.loads(memory_on)
    except json.JSONDecodeError as exc:
        raise ValueError("experience prompts must be JSON") from exc
    if not isinstance(off, dict) or not isinstance(on, dict):
        raise ValueError("experience prompts must be objects")
    if set(off) != set(on):
        raise ValueError("experience prompts have different top-level contracts")
    allowed = {"experience_memory_block"}
    if any(off[key] != on[key] for key in off.keys() - allowed):
        raise ValueError("experience prompts differ outside the memory block")
