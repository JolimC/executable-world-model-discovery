from __future__ import annotations

from pathlib import Path

from world_model_search.dsl.primitives import (
    PrimitiveDefinition,
    PrimitiveRegistry,
    decode_program,
    encode_program,
    expand_primitives,
    replace_subtree,
)
from world_model_search.evaluation.phase5_transfer import (
    load_transfer_registry,
    reference_program,
    selector_core,
)
from world_model_search.serialization import sha256_json


def test_phase5_primitive_codec_round_trips_every_predeclared_reference_stratum() -> None:
    split = load_transfer_registry(Path("configs/phase5-transfer-split-v1.yaml"))
    definition = PrimitiveDefinition(selector_core())
    registry = PrimitiveRegistry(
        split.content_hash,
        sha256_json({"analysis": "property-test"}),
        (sha256_json({"evidence": "property-test"}),),
        (definition,),
    )
    observed = 0
    for family in split.families:
        for variant in family.variants:
            reference = reference_program(family.generator, variant)
            learned = replace_subtree(reference, definition.ast, definition.primitive_id)
            bits = encode_program(learned, registry)
            decoded = decode_program(bits, registry)
            assert encode_program(decoded, registry) == bits
            assert expand_primitives(decoded, registry) == reference
            observed += 1
    assert observed == 12
