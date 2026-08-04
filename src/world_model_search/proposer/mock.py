"""Deterministic proposer used only by the Phase 0 end-to-end fixture."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from world_model_search.domain.types import (
    CandidatePayload,
    ProposalBudget,
    ProposalContext,
    ProposalRole,
    RuleExpr,
)


class MockProposer:
    """Generate typed fixture candidates from an explicit proposer seed."""

    proposer_id = "mock"

    def propose(
        self, context: ProposalContext, budget: ProposalBudget
    ) -> Sequence[CandidatePayload]:
        del context  # Its safe type is still exercised by the end-to-end loop.
        proposals: list[CandidatePayload] = []
        for index in range(budget.start_index, budget.start_index + budget.max_candidates):
            material = f"mock-proposer-v1:{budget.proposer_seed}:{index}".encode()
            fingerprint = hashlib.sha256(material).hexdigest()[:16]
            proposals.append(
                CandidatePayload(
                    ast=RuleExpr(
                        node="Phase0Fixture",
                        arguments=(("index", index), ("fingerprint", fingerprint)),
                    ),
                    role=ProposalRole.EXPLOIT,
                )
            )
        return tuple(proposals)
