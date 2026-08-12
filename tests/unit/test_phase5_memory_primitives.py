from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from world_model_search.domain.types import SplitLabel
from world_model_search.dsl.ast import And, At, Majority
from world_model_search.dsl.codec import CodecError, encoded_length
from world_model_search.dsl.primitives import (
    PrimitiveCall,
    PrimitiveDefinition,
    PrimitiveRegistry,
    decode_library,
    decode_program,
    encode_library,
    encode_program,
    expand_primitives,
    library_definition_cost,
)
from world_model_search.errors import PersistenceError
from world_model_search.memory.store import Phase5MemoryStore
from world_model_search.memory.types import (
    EvidenceFact,
    EvidencePolarity,
    MemoryApplicability,
    MemoryKind,
    ValidationState,
)
from world_model_search.serialization import sha256_json


def _digest(label: str) -> str:
    return sha256_json({"label": label})


def _fact(task: str, family: str, role: SplitLabel, suffix: str = "one") -> EvidenceFact:
    return EvidenceFact(
        task_id=task,
        family_id=family,
        role=role,
        semantic_hash=_digest(f"semantic:{task}"),
        run_hash=_digest(f"run:{task}:{suffix}"),
        candidate_hash=_digest(f"candidate:{task}:{suffix}"),
        evaluation_hash=_digest(f"evaluation:{task}:{suffix}"),
        artifact_hash=_digest(f"artifact:{task}"),
    )


def _applicability() -> MemoryApplicability:
    return MemoryApplicability("elementary-public-world-v1", "binary-ca-radius1-dsl-v1")


def test_memory_counts_independent_tasks_and_families_not_retries(tmp_path: Path) -> None:
    first = _fact("task-a", "family-a", SplitLabel.TRAINING, "first")
    retry = _fact("task-a", "family-a", SplitLabel.TRAINING, "retry")
    second = _fact("task-b", "family-b", SplitLabel.TRAINING)
    catalog = {fact.evidence_id: fact for fact in (first, retry, second)}
    registry_hash = _digest("split")
    with Phase5MemoryStore(
        tmp_path / "memory.sqlite3",
        split_registry_hash=registry_hash,
        evidence_catalog=catalog,
    ) as store:
        for fact in catalog.values():
            store.admit_evidence(fact)
        record = store.propose_record(
            kind=MemoryKind.SEARCH_LESSON,
            proposer_text="Prefer compact typed compositions that preserve exact correctness.",
            scope="global-f0",
            applicability=_applicability(),
            support_evidence_ids=(first.evidence_id, retry.evidence_id),
            provenance_hashes=(_digest("provenance"),),
        )
        assert store.independent_support(record, EvidencePolarity.SUPPORT) == (1, 1)
        with pytest.raises(PersistenceError, match="two independent training"):
            store.transition(record, ValidationState.PROMOTED, reason="premature")
        store.link_evidence(record, second.evidence_id, EvidencePolarity.SUPPORT)
        assert store.independent_support(record, EvidencePolarity.SUPPORT) == (2, 2)
        store.transition(record, ValidationState.PROMOTED, reason="independent transfer support")
        snapshot = store.freeze_snapshot(tmp_path / "snapshot.json")
        assert tuple(item.record_id for item in snapshot.items) == (record,)
        assert store.deterministic_export() == store.deterministic_export()


def test_memory_rejects_missing_provenance_counterevidence_and_corruption(
    tmp_path: Path,
) -> None:
    facts = (
        _fact("task-a", "family-a", SplitLabel.TRAINING),
        _fact("task-b", "family-b", SplitLabel.TRAINING),
        _fact("task-c", "family-c", SplitLabel.DEVELOPMENT),
    )
    catalog = {fact.evidence_id: fact for fact in facts}
    path = tmp_path / "memory.sqlite3"
    with Phase5MemoryStore(
        path, split_registry_hash=_digest("split"), evidence_catalog=catalog
    ) as store:
        with pytest.raises(PersistenceError, match="eligible catalog"):
            store.admit_evidence(_fact("unknown", "unknown-family", SplitLabel.TRAINING))
        for fact in facts:
            store.admit_evidence(fact)
        record = store.propose_record(
            kind=MemoryKind.HYPOTHESIS,
            proposer_text="This bounded structural lesson may transfer within F0.",
            scope="global-f0",
            applicability=_applicability(),
            support_evidence_ids=(facts[0].evidence_id, facts[1].evidence_id),
            provenance_hashes=(_digest("provenance"),),
        )
        store.link_evidence(record, facts[2].evidence_id, EvidencePolarity.COUNTER)
        with pytest.raises(PersistenceError, match="no counterevidence"):
            store.transition(record, ValidationState.PROMOTED, reason="contradicted")
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("UPDATE evidence_link SET link_hash=? WHERE sequence=1", ("0" * 64,))
    connection.close()
    with (
        Phase5MemoryStore(path, split_registry_hash=_digest("split")) as store,
        pytest.raises(PersistenceError, match="evidence link hash mismatch"),
    ):
        store.audit()


def test_memory_schema_and_split_mismatch_refuse_migration(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite3"
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES ('database_schema_version','0')")
    connection.close()
    with pytest.raises(PersistenceError, match="schema version mismatch"):
        Phase5MemoryStore(path, split_registry_hash=_digest("split"))

    valid = tmp_path / "valid.sqlite3"
    with Phase5MemoryStore(valid, split_registry_hash=_digest("split")):
        pass
    with pytest.raises(PersistenceError, match="split registry identity mismatch"):
        Phase5MemoryStore(valid, split_registry_hash=_digest("another-split"))


def test_primitive_language_is_typed_prefix_decodable_and_exactly_expandable() -> None:
    definition = PrimitiveDefinition(And(At(-1), At(1)))
    registry = PrimitiveRegistry(
        _digest("split"), _digest("analysis"), (_digest("evidence"),), (definition,)
    )
    library = encode_library(registry.definitions)
    assert decode_library(library) == registry.definitions
    assert library_definition_cost(registry.definitions) == len(library)
    call = PrimitiveCall(definition.primitive_id)
    program = encode_program(call, registry)
    decoded = decode_program(program, registry)
    assert decoded == call
    assert expand_primitives(decoded, registry) == definition.ast
    assert len(program) < encoded_length(definition.ast)
    with pytest.raises(CodecError, match="trailing"):
        decode_program(program + "0", registry)
    with pytest.raises(CodecError):
        decode_library(library + "0")


def test_primitive_rejects_builtin_macro_equivalence_and_unsafe_definition() -> None:
    with pytest.raises(ValueError, match="built-in macro"):
        PrimitiveDefinition(Majority((-1, 0, 1)))
    with pytest.raises(ValueError, match="canonical"):
        PrimitiveDefinition(And(At(1), At(-1)))


@pytest.mark.parametrize("net_positive", [False, True])
def test_primitive_promotion_requires_exact_positive_net_gate(
    tmp_path: Path, net_positive: bool
) -> None:
    training = (
        _fact("train-a", "train-family-a", SplitLabel.TRAINING),
        _fact("train-b", "train-family-b", SplitLabel.TRAINING),
    )
    development = (
        _fact("dev-a", "dev-family-a", SplitLabel.DEVELOPMENT),
        _fact("dev-b", "dev-family-b", SplitLabel.DEVELOPMENT),
    )
    catalog = {fact.evidence_id: fact for fact in (*training, *development)}
    with Phase5MemoryStore(
        tmp_path / "primitive.sqlite3",
        split_registry_hash=_digest("split"),
        evidence_catalog=catalog,
    ) as store:
        for fact in catalog.values():
            store.admit_evidence(fact)
        record = store.propose_record(
            kind=MemoryKind.PRIMITIVE_PROPOSAL,
            proposer_text="Use the frozen typed primitive only where its applicability matches.",
            scope="global-f0",
            applicability=_applicability(),
            support_evidence_ids=tuple(fact.evidence_id for fact in training),
            provenance_hashes=(_digest("primitive-provenance"),),
            definition_cost_bits=12,
        )
        for fact in development:
            store.link_evidence(record, fact.evidence_id, EvidencePolarity.VALIDATION)
        gate = {
            "strictly_positive_net_gain": net_positive,
            "all_correct": True,
            "definition_cost_bits": 12,
        }
        if net_positive:
            store.transition(
                record, ValidationState.PROMOTED, reason="gate passed", gate_payload=gate
            )
            assert store.freeze_snapshot(tmp_path / "promoted.json").items
        else:
            with pytest.raises(PersistenceError, match="positive net-MDL"):
                store.transition(
                    record, ValidationState.PROMOTED, reason="gate failed", gate_payload=gate
                )
            store.transition(record, ValidationState.REJECTED, reason="zero-or-negative net gain")
            assert not store.freeze_snapshot(tmp_path / "rejected.json").items
