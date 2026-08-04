from __future__ import annotations

from world_model_search.config import AppConfig
from world_model_search.domain.types import ProposalBudget, ProposalContext
from world_model_search.proposer.mock import MockProposer
from world_model_search.search.fixture import make_fixture_task


def test_mock_proposer_is_seeded_and_deterministic(app_config: AppConfig) -> None:
    context = ProposalContext(task=make_fixture_task(app_config).public_view())
    budget = ProposalBudget(max_candidates=2, start_index=1, proposer_seed=42)
    proposer = MockProposer()
    assert proposer.propose(context, budget) == proposer.propose(context, budget)
    assert proposer.propose(context, budget) != proposer.propose(
        context, ProposalBudget(max_candidates=2, start_index=1, proposer_seed=43)
    )
