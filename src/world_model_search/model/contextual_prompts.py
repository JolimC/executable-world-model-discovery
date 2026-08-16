"""Canonical memory-v3 prompt injection and A/B isolation checks."""

from __future__ import annotations

from world_model_search.memory.contextual import MEMORY_BLOCK_SCHEMA
from world_model_search.memory.contextual_retrieval import RenderedMemoryBlock
from world_model_search.serialization import canonical_json, parse_json_object

CONTEXTUAL_SEARCH_PROMPT_VERSION = "phase5-contextual-experience-search-v1"
CONTEXTUAL_PROMPT_FIELD = "cross_task_memory"


def inject_contextual_memory(
    *,
    base_prompt: str,
    memory_block: RenderedMemoryBlock,
) -> str:
    payload = parse_json_object(base_prompt)
    if CONTEXTUAL_PROMPT_FIELD in payload:
        raise ValueError("base prompt already contains contextual memory")
    block = memory_block.value
    if block.get("schema_version") != MEMORY_BLOCK_SCHEMA:
        raise ValueError("contextual memory block schema is invalid")
    payload[CONTEXTUAL_PROMPT_FIELD] = block
    return canonical_json(payload)


def assert_contextual_prompt_isolation(control_prompt: str, treatment_prompt: str) -> None:
    """Prove that paired prompts differ only inside the canonical evidence block."""

    control = parse_json_object(control_prompt)
    treatment = parse_json_object(treatment_prompt)
    control_memory = control.pop(CONTEXTUAL_PROMPT_FIELD, None)
    treatment_memory = treatment.pop(CONTEXTUAL_PROMPT_FIELD, None)
    if not isinstance(control_memory, dict) or not isinstance(treatment_memory, dict):
        raise ValueError("paired prompts must both contain contextual memory blocks")
    if (
        control_memory.get("schema_version") != MEMORY_BLOCK_SCHEMA
        or treatment_memory.get("schema_version") != MEMORY_BLOCK_SCHEMA
    ):
        raise ValueError("paired prompt memory schema differs")
    if control != treatment:
        raise ValueError("control/treatment prompts differ outside contextual memory")
    control_items = control_memory.get("cross_task_experience")
    if not isinstance(control_items, list) or control_items:
        raise ValueError("control prompt does not contain the canonical empty memory block")
    forbidden = {"task_id", "run_id", "parent_candidate_id", "child_candidate_id"}
    if any(key in canonical_json(treatment_memory) for key in forbidden):
        raise ValueError("treatment memory exposes provenance identifiers")
