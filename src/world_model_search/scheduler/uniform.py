"""Counter-derived uniform scheduler for Phase 3 sorted branch sets."""

from __future__ import annotations

from dataclasses import dataclass

from world_model_search.dsl.versions import PHASE3_SCHEDULER_VERSION
from world_model_search.search.operators import CounterRng
from world_model_search.serialization import JsonObject, sha256_json


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    scheduler_version: str
    eligible_branch_ids: tuple[str, ...]
    selected_branch_id: str
    selected_index: int
    probability_numerator: int
    probability_denominator: int
    remaining_proposal_attempts: int
    remaining_oracle_calls: int
    decision_hash: str

    def to_value(self) -> JsonObject:
        return {
            "scheduler_version": self.scheduler_version,
            "eligible_branch_ids": list(self.eligible_branch_ids),
            "selected_branch_id": self.selected_branch_id,
            "selected_index": self.selected_index,
            "selection_probability": {
                "numerator": self.probability_numerator,
                "denominator": self.probability_denominator,
            },
            "remaining_budget": {
                "proposal_attempts": self.remaining_proposal_attempts,
                "oracle_calls": self.remaining_oracle_calls,
            },
            "decision_hash": self.decision_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> SchedulerDecision:
        if not isinstance(value, dict):
            raise ValueError("scheduler decision must be an object")
        eligible = value.get("eligible_branch_ids")
        probability = value.get("selection_probability")
        remaining = value.get("remaining_budget")
        if (
            value.get("scheduler_version") != PHASE3_SCHEDULER_VERSION
            or not isinstance(eligible, list)
            or not all(isinstance(item, str) for item in eligible)
            or eligible != sorted(eligible)
            or len(eligible) != len(set(eligible))
            or not isinstance(probability, dict)
            or not isinstance(remaining, dict)
        ):
            raise ValueError("scheduler decision structure is invalid")
        selected = value.get("selected_branch_id")
        index = value.get("selected_index")
        numerator = probability.get("numerator")
        denominator = probability.get("denominator")
        proposals = remaining.get("proposal_attempts")
        calls = remaining.get("oracle_calls")
        if (
            not isinstance(selected, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(eligible)
            or eligible[index] != selected
            or numerator != 1
            or denominator != len(eligible)
            or isinstance(proposals, bool)
            or not isinstance(proposals, int)
            or isinstance(calls, bool)
            or not isinstance(calls, int)
        ):
            raise ValueError("scheduler selection/probability is invalid")
        payload = dict(value)
        recorded_hash = payload.pop("decision_hash", None)
        if not isinstance(recorded_hash, str) or sha256_json(payload) != recorded_hash:
            raise ValueError("scheduler decision hash mismatch")
        return cls(
            scheduler_version=PHASE3_SCHEDULER_VERSION,
            eligible_branch_ids=tuple(eligible),
            selected_branch_id=selected,
            selected_index=index,
            probability_numerator=numerator,
            probability_denominator=denominator,
            remaining_proposal_attempts=proposals,
            remaining_oracle_calls=calls,
            decision_hash=recorded_hash,
        )


class UniformScheduler:
    scheduler_version = PHASE3_SCHEDULER_VERSION

    def select(
        self,
        branch_ids: tuple[str, ...],
        *,
        master_seed: int,
        selection_counter: int,
        remaining_proposal_attempts: int,
        remaining_oracle_calls: int,
    ) -> SchedulerDecision:
        eligible = tuple(sorted(branch_ids))
        if not eligible or len(eligible) != len(set(eligible)):
            raise ValueError("eligible branches must be nonempty and unique")
        index = CounterRng(master_seed, "scheduler-choice", selection_counter).integer(
            "uniform-index", len(eligible)
        )
        payload = {
            "scheduler_version": self.scheduler_version,
            "eligible_branch_ids": eligible,
            "selected_branch_id": eligible[index],
            "selected_index": index,
            "selection_probability": {"numerator": 1, "denominator": len(eligible)},
            "remaining_budget": {
                "proposal_attempts": remaining_proposal_attempts,
                "oracle_calls": remaining_oracle_calls,
            },
        }
        return SchedulerDecision(
            scheduler_version=self.scheduler_version,
            eligible_branch_ids=eligible,
            selected_branch_id=eligible[index],
            selected_index=index,
            probability_numerator=1,
            probability_denominator=len(eligible),
            remaining_proposal_attempts=remaining_proposal_attempts,
            remaining_oracle_calls=remaining_oracle_calls,
            decision_hash=sha256_json(payload),
        )
