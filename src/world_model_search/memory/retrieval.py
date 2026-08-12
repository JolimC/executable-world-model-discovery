"""Deterministic capability-safe Phase 5 memory retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from world_model_search.domain.types import ElementaryPublicWorldSpec, PublicTask
from world_model_search.memory.types import MemorySnapshot, SafeMemoryItem
from world_model_search.phase5_versions import PHASE5_RETRIEVAL_VERSION
from world_model_search.serialization import JsonObject, canonical_json, sha256_json

_WORDS = re.compile(r"[a-z0-9_-]+")


def _terms(value: str) -> frozenset[str]:
    return frozenset(_WORDS.findall(value.casefold()))


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    query_id: str
    snapshot_hash: str
    eligible_record_ids: tuple[str, ...]
    rankings: tuple[tuple[str, int], ...]
    selected_record_ids: tuple[str, ...]
    exclusions: tuple[tuple[str, str], ...]
    rendered_memory: str
    rendered_bytes: int
    conservative_token_bound: int

    def to_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "retrieval_version": PHASE5_RETRIEVAL_VERSION,
                "query_id": self.query_id,
                "snapshot_hash": self.snapshot_hash,
                "eligible_record_ids": list(self.eligible_record_ids),
                "rankings": [
                    {"record_id": record_id, "score": score} for record_id, score in self.rankings
                ],
                "selected_record_ids": list(self.selected_record_ids),
                "exclusions": [
                    {"record_id": record_id, "reason": reason}
                    for record_id, reason in self.exclusions
                ],
                "rendered_memory_utf8": self.rendered_memory,
                "rendered_bytes": self.rendered_bytes,
                "conservative_token_bound": self.conservative_token_bound,
            },
        )


def _eligible(item: SafeMemoryItem, task: PublicTask) -> tuple[bool, str]:
    world = task.public_world_spec
    if not isinstance(world, ElementaryPublicWorldSpec):
        return False, "world-type-mismatch"
    if item.applicability.world_specification_version != world.specification_version:
        return False, "world-version-mismatch"
    if item.applicability.dsl_version != world.dsl_version:
        return False, "dsl-version-mismatch"
    if item.scope not in {"global-f0", f"world:{world.specification_version}"}:
        return False, "scope-not-publicly-applicable"
    return True, "eligible"


def retrieve_memory(
    *,
    task: PublicTask,
    snapshot: MemorySnapshot,
    public_search_state: JsonObject,
    max_items: int,
    max_bytes: int,
    max_tokens: int,
) -> RetrievalRecord:
    """Rank using public bytes only and apply stable item/byte/token bounds."""

    if min(max_items, max_bytes, max_tokens) < 0:
        raise ValueError("retrieval bounds must be nonnegative")
    query_payload: JsonObject = {
        "retrieval_version": PHASE5_RETRIEVAL_VERSION,
        "public_task": cast(JsonObject, __import__("json").loads(canonical_json(task))),
        "public_search_state": public_search_state,
        "snapshot_hash": snapshot.snapshot_hash,
        "bounds": {"items": max_items, "bytes": max_bytes, "tokens": max_tokens},
    }
    query_id = sha256_json(query_payload)
    query_terms = _terms(canonical_json(query_payload["public_task"])) | _terms(
        canonical_json(public_search_state)
    )
    eligible: list[SafeMemoryItem] = []
    exclusions: list[tuple[str, str]] = []
    for item in snapshot.items:
        allowed, reason = _eligible(item, task)
        if allowed:
            eligible.append(item)
        else:
            exclusions.append((item.record_id, reason))
    ranked = sorted(
        ((item, len(query_terms & _terms(item.proposer_text))) for item in eligible),
        key=lambda pair: (-pair[1], pair[0].record_id),
    )
    selected: list[SafeMemoryItem] = []
    for item, _score in ranked:
        if len(selected) >= max_items:
            exclusions.append((item.record_id, "item-bound"))
            continue
        candidate = canonical_json(
            {
                "memory_block_version": "phase5-memory-block-v1",
                "items": [entry.to_value() for entry in (*selected, item)],
            }
        )
        byte_count = len(candidate.encode("utf-8"))
        if byte_count > max_bytes:
            exclusions.append((item.record_id, "byte-bound"))
            continue
        if byte_count > max_tokens:
            exclusions.append((item.record_id, "conservative-token-bound"))
            continue
        selected.append(item)
    rendered = canonical_json(
        {
            "memory_block_version": "phase5-memory-block-v1",
            "items": [item.to_value() for item in selected],
        }
    )
    rendered_bytes = len(rendered.encode("utf-8"))
    if rendered_bytes > max_bytes or rendered_bytes > max_tokens:
        raise ValueError("retrieval bounds cannot contain even the explicit empty memory block")
    return RetrievalRecord(
        query_id=query_id,
        snapshot_hash=snapshot.snapshot_hash,
        eligible_record_ids=tuple(item.record_id for item in eligible),
        rankings=tuple((item.record_id, score) for item, score in ranked),
        selected_record_ids=tuple(item.record_id for item in selected),
        exclusions=tuple(sorted(exclusions)),
        rendered_memory=rendered,
        rendered_bytes=rendered_bytes,
        conservative_token_bound=rendered_bytes,
    )
