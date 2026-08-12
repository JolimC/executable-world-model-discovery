"""Strict Decimal-free Phase 4 price and hierarchical ceiling policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from world_model_search.errors import ConfigurationError
from world_model_search.model.types import ModelUsage
from world_model_search.serialization import JsonObject, sha256_json

PRICE_POLICY_VERSION = "phase4-price-and-ceilings-v1"
DUAL_BUDGET_POLICY_VERSION = "published-rate-and-reconciled-cash-v2"
CASH_BUDGET_VERSION = "reconciled-cash-plus-unreconciled-published-v1"
SUPPORTED_PRICE_POLICY_VERSIONS = frozenset({PRICE_POLICY_VERSION, DUAL_BUDGET_POLICY_VERSION})
CASH_VERIFICATION_LEVELS = frozenset(
    {"user-reported-unverified", "provider-export-verified", "invoice-verified"}
)


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _keys(value: dict[str, object], expected: set[str], location: str) -> None:
    if set(value) != expected:
        raise ConfigurationError(f"{location} has missing or unknown fields")


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"{location} must be a nonnegative integer")
    return value


def _date_or_offset_datetime(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{location} must be an ISO date or offset datetime")
    try:
        if "T" not in value:
            date.fromisoformat(value)
        else:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError
    except ValueError as exc:
        raise ConfigurationError(f"{location} must be an ISO date or offset datetime") from exc
    return value


@dataclass(frozen=True, slots=True)
class CashBudgetPolicy:
    budget_version: str
    personal_lifetime_cap_nano_usd: int
    safety_buffer_nano_usd: int
    opening_reconciled_cash_nano_usd: int
    opening_covered_published_nano_usd: int
    opening_observed_at: str
    opening_scope: str
    opening_source: str
    opening_verification: str

    def opening_reconciliation_value(self) -> JsonObject:
        return {
            "billed_nano_usd": self.opening_reconciled_cash_nano_usd,
            "covered_published_nano_usd": self.opening_covered_published_nano_usd,
            "observed_at": self.opening_observed_at,
            "scope": self.opening_scope,
            "source": self.opening_source,
            "verification": self.opening_verification,
        }


@dataclass(frozen=True, slots=True)
class PriceEntry:
    provider: str
    model: str
    endpoint: str
    service_tier: str
    uncached_input_nano_usd_per_token: int
    cached_input_nano_usd_per_token: int
    output_nano_usd_per_token: int
    provider_call_fee_nano_usd: int
    source_url: str
    verified_at: str

    def cost(self, usage: ModelUsage) -> int:
        uncached = usage.input_tokens - usage.cached_input_tokens
        return (
            uncached * self.uncached_input_nano_usd_per_token
            + usage.cached_input_tokens * self.cached_input_nano_usd_per_token
            + usage.output_tokens * self.output_nano_usd_per_token
            + self.provider_call_fee_nano_usd
        )

    def maximum_cost(self, *, input_token_bound: int, max_output_tokens: int) -> int:
        if input_token_bound < 0 or max_output_tokens < 1:
            raise ValueError("preflight token bounds are invalid")
        return (
            input_token_bound * self.uncached_input_nano_usd_per_token
            + max_output_tokens * self.output_nano_usd_per_token
            + self.provider_call_fee_nano_usd
        )


def _price_entry(value: object) -> PriceEntry:
    price = _mapping(value, "price")
    _keys(
        price,
        {
            "provider",
            "model",
            "endpoint",
            "service_tier",
            "uncached_input_nano_usd_per_token",
            "cached_input_nano_usd_per_token",
            "output_nano_usd_per_token",
            "provider_call_fee_nano_usd",
            "source_url",
            "verified_at",
        },
        "price",
    )
    strings = ("provider", "model", "endpoint", "service_tier", "source_url", "verified_at")
    if any(not isinstance(price[name], str) or not price[name] for name in strings):
        raise ConfigurationError("price string fields must be nonempty")
    entry = PriceEntry(
        provider=str(price["provider"]),
        model=str(price["model"]),
        endpoint=str(price["endpoint"]),
        service_tier=str(price["service_tier"]),
        uncached_input_nano_usd_per_token=_integer(
            price["uncached_input_nano_usd_per_token"], "price.uncached_input"
        ),
        cached_input_nano_usd_per_token=_integer(
            price["cached_input_nano_usd_per_token"], "price.cached_input"
        ),
        output_nano_usd_per_token=_integer(price["output_nano_usd_per_token"], "price.output"),
        provider_call_fee_nano_usd=_integer(
            price["provider_call_fee_nano_usd"], "price.provider_call_fee"
        ),
        source_url=str(price["source_url"]),
        verified_at=str(price["verified_at"]),
    )
    if (
        entry.provider != "openai"
        or entry.model != "gpt-5-mini-2025-08-07"
        or entry.endpoint != "v1/responses"
        or entry.service_tier != "default"
    ):
        raise ConfigurationError("price entry does not match the frozen provider contract")
    return entry


@dataclass(frozen=True, slots=True)
class PricePolicy:
    policy_version: str
    currency_unit: str
    opening_balance_nano_usd: int
    project_lifetime_cap_nano_usd: int
    prior_phase_0_3_spend_nano_usd: int
    phase4_cap_nano_usd: int
    canary_cap_nano_usd: int
    development_cap_nano_usd: int
    locked_test_cap_nano_usd: int
    request_cap_nano_usd: int
    pilot_child_cap_nano_usd: int
    locked_child_cap_nano_usd: int
    price: PriceEntry
    cash_budget: CashBudgetPolicy | None = None

    @property
    def uses_reconciled_cash_budget(self) -> bool:
        return self.cash_budget is not None

    @property
    def content_hash(self) -> str:
        return sha256_json(self.to_value())

    def to_value(self) -> JsonObject:
        price: JsonObject = {
            "provider": self.price.provider,
            "model": self.price.model,
            "endpoint": self.price.endpoint,
            "service_tier": self.price.service_tier,
            "uncached_input_nano_usd_per_token": self.price.uncached_input_nano_usd_per_token,
            "cached_input_nano_usd_per_token": self.price.cached_input_nano_usd_per_token,
            "output_nano_usd_per_token": self.price.output_nano_usd_per_token,
            "provider_call_fee_nano_usd": self.price.provider_call_fee_nano_usd,
            "source_url": self.price.source_url,
            "verified_at": self.price.verified_at,
            "reasoning_billing": "included-in-output-tokens-v1",
        }
        if self.cash_budget is not None:
            return {
                "policy_version": self.policy_version,
                "currency_unit": self.currency_unit,
                "opening_published_rate_nano_usd": self.opening_balance_nano_usd,
                "published_exposure_ceilings_nano_usd": {
                    "phase4": self.phase4_cap_nano_usd,
                    "canary": self.canary_cap_nano_usd,
                    "development_pilot": self.development_cap_nano_usd,
                    "locked_test": self.locked_test_cap_nano_usd,
                    "one_request": self.request_cap_nano_usd,
                    "pilot_child": self.pilot_child_cap_nano_usd,
                    "locked_child": self.locked_child_cap_nano_usd,
                },
                "cash_budget": {
                    "budget_version": self.cash_budget.budget_version,
                    "personal_lifetime_nano_usd": (self.cash_budget.personal_lifetime_cap_nano_usd),
                    "safety_buffer_nano_usd": self.cash_budget.safety_buffer_nano_usd,
                    "opening_reconciliation": (self.cash_budget.opening_reconciliation_value()),
                },
                "price": price,
            }
        return {
            "policy_version": self.policy_version,
            "currency_unit": self.currency_unit,
            "opening_balance_nano_usd": self.opening_balance_nano_usd,
            "ceilings_nano_usd": {
                "project_lifetime": self.project_lifetime_cap_nano_usd,
                "prior_phase_0_3_spend": self.prior_phase_0_3_spend_nano_usd,
                "phase4": self.phase4_cap_nano_usd,
                "canary": self.canary_cap_nano_usd,
                "development_pilot": self.development_cap_nano_usd,
                "locked_test": self.locked_test_cap_nano_usd,
                "one_request": self.request_cap_nano_usd,
                "pilot_child": self.pilot_child_cap_nano_usd,
                "locked_child": self.locked_child_cap_nano_usd,
            },
            "price": price,
        }

    def stage_cap(self, stage: str) -> int:
        try:
            return {
                "canary": self.canary_cap_nano_usd,
                "development": self.development_cap_nano_usd,
                "pilot": self.development_cap_nano_usd,
                "locked-test": self.locked_test_cap_nano_usd,
                "fake": 0,
            }[stage]
        except KeyError as exc:
            raise ConfigurationError(f"unknown Phase 4 stage: {stage}") from exc

    def child_cap(self, stage: str) -> int:
        return (
            self.locked_child_cap_nano_usd
            if stage == "locked-test"
            else self.pilot_child_cap_nano_usd
        )


def _dual_policy_from_mapping(root: dict[str, object]) -> PricePolicy:
    _keys(
        root,
        {
            "policy_version",
            "currency_unit",
            "opening_published_rate_nano_usd",
            "published_exposure_ceilings_nano_usd",
            "cash_budget",
            "price",
        },
        "price policy",
    )
    if root["currency_unit"] != "nano-USD":
        raise ConfigurationError("unsupported dual-budget currency unit")
    ceilings = _mapping(
        root["published_exposure_ceilings_nano_usd"],
        "published_exposure_ceilings_nano_usd",
    )
    ceiling_names = {
        "phase4",
        "canary",
        "development_pilot",
        "locked_test",
        "one_request",
        "pilot_child",
        "locked_child",
    }
    _keys(ceilings, ceiling_names, "published_exposure_ceilings_nano_usd")
    values = {
        name: _integer(ceilings[name], f"published_exposure_ceilings_nano_usd.{name}")
        for name in ceiling_names
    }
    immutable_exposure = {
        "phase4": 30_000_000_000,
        "canary": 250_000_000,
        "development_pilot": 9_750_000_000,
        "locked_test": 20_000_000_000,
        "one_request": 10_000_000,
        "pilot_child": 150_000_000,
        "locked_child": 500_000_000,
    }
    if values != immutable_exposure:
        raise ConfigurationError(
            "dual-budget published exposure ceilings must preserve the frozen local caps"
        )
    opening_published = _integer(
        root["opening_published_rate_nano_usd"],
        "opening_published_rate_nano_usd",
    )
    cash_raw = _mapping(root["cash_budget"], "cash_budget")
    _keys(
        cash_raw,
        {
            "budget_version",
            "personal_lifetime_nano_usd",
            "safety_buffer_nano_usd",
            "opening_reconciliation",
        },
        "cash_budget",
    )
    if cash_raw["budget_version"] != CASH_BUDGET_VERSION:
        raise ConfigurationError("unsupported cash-budget enforcement version")
    personal_cap = _integer(
        cash_raw["personal_lifetime_nano_usd"], "cash_budget.personal_lifetime_nano_usd"
    )
    safety_buffer = _integer(
        cash_raw["safety_buffer_nano_usd"], "cash_budget.safety_buffer_nano_usd"
    )
    if personal_cap != 100_000_000_000 or safety_buffer > personal_cap:
        raise ConfigurationError("dual-budget personal cash ceiling must remain $100")
    opening_raw = _mapping(cash_raw["opening_reconciliation"], "cash_budget.opening_reconciliation")
    _keys(
        opening_raw,
        {
            "billed_nano_usd",
            "covered_published_nano_usd",
            "observed_at",
            "scope",
            "source",
            "verification",
        },
        "cash_budget.opening_reconciliation",
    )
    billed = _integer(
        opening_raw["billed_nano_usd"],
        "cash_budget.opening_reconciliation.billed_nano_usd",
    )
    covered = _integer(
        opening_raw["covered_published_nano_usd"],
        "cash_budget.opening_reconciliation.covered_published_nano_usd",
    )
    if covered != opening_published:
        raise ConfigurationError(
            "opening cash reconciliation must cover the complete opening published amount"
        )
    if billed + safety_buffer > personal_cap:
        raise ConfigurationError("opening reconciled cash plus safety buffer exceeds $100")
    scope = opening_raw["scope"]
    source = opening_raw["source"]
    verification = opening_raw["verification"]
    if not isinstance(scope, str) or not scope or not isinstance(source, str) or not source:
        raise ConfigurationError("opening cash reconciliation scope/source must be nonempty")
    if verification not in CASH_VERIFICATION_LEVELS:
        raise ConfigurationError("opening cash reconciliation verification is invalid")
    cash_budget = CashBudgetPolicy(
        budget_version=CASH_BUDGET_VERSION,
        personal_lifetime_cap_nano_usd=personal_cap,
        safety_buffer_nano_usd=safety_buffer,
        opening_reconciled_cash_nano_usd=billed,
        opening_covered_published_nano_usd=covered,
        opening_observed_at=_date_or_offset_datetime(
            opening_raw["observed_at"],
            "cash_budget.opening_reconciliation.observed_at",
        ),
        opening_scope=scope,
        opening_source=source,
        opening_verification=str(verification),
    )
    return PricePolicy(
        policy_version=DUAL_BUDGET_POLICY_VERSION,
        currency_unit="nano-USD",
        opening_balance_nano_usd=opening_published,
        project_lifetime_cap_nano_usd=personal_cap,
        prior_phase_0_3_spend_nano_usd=0,
        phase4_cap_nano_usd=values["phase4"],
        canary_cap_nano_usd=values["canary"],
        development_cap_nano_usd=values["development_pilot"],
        locked_test_cap_nano_usd=values["locked_test"],
        request_cap_nano_usd=values["one_request"],
        pilot_child_cap_nano_usd=values["pilot_child"],
        locked_child_cap_nano_usd=values["locked_child"],
        price=_price_entry(root["price"]),
        cash_budget=cash_budget,
    )


def policy_from_mapping(raw: object) -> PricePolicy:
    root = _mapping(raw, "price policy")
    if root.get("policy_version") == DUAL_BUDGET_POLICY_VERSION:
        return _dual_policy_from_mapping(root)
    _keys(
        root,
        {
            "policy_version",
            "currency_unit",
            "opening_balance_nano_usd",
            "ceilings_nano_usd",
            "price",
        },
        "price policy",
    )
    if root["policy_version"] != PRICE_POLICY_VERSION or root["currency_unit"] != "nano-USD":
        raise ConfigurationError("unsupported Phase 4 price policy version or unit")
    ceilings = _mapping(root["ceilings_nano_usd"], "ceilings_nano_usd")
    ceiling_names = {
        "project_lifetime",
        "prior_phase_0_3_spend",
        "phase4",
        "canary",
        "development_pilot",
        "locked_test",
        "one_request",
        "pilot_child",
        "locked_child",
    }
    _keys(ceilings, ceiling_names, "ceilings_nano_usd")
    values = {name: _integer(ceilings[name], f"ceilings_nano_usd.{name}") for name in ceiling_names}
    immutable = {
        "project_lifetime": 100_000_000_000,
        "prior_phase_0_3_spend": 0,
        "phase4": 30_000_000_000,
        "canary": 250_000_000,
        "development_pilot": 9_750_000_000,
        "locked_test": 20_000_000_000,
        "one_request": 10_000_000,
        "pilot_child": 150_000_000,
        "locked_child": 500_000_000,
    }
    if values != immutable:
        raise ConfigurationError("price policy ceilings must exactly match the frozen contract")
    entry = _price_entry(root["price"])
    return PricePolicy(
        policy_version=PRICE_POLICY_VERSION,
        currency_unit="nano-USD",
        opening_balance_nano_usd=_integer(
            root["opening_balance_nano_usd"], "opening_balance_nano_usd"
        ),
        project_lifetime_cap_nano_usd=values["project_lifetime"],
        prior_phase_0_3_spend_nano_usd=values["prior_phase_0_3_spend"],
        phase4_cap_nano_usd=values["phase4"],
        canary_cap_nano_usd=values["canary"],
        development_cap_nano_usd=values["development_pilot"],
        locked_test_cap_nano_usd=values["locked_test"],
        request_cap_nano_usd=values["one_request"],
        pilot_child_cap_nano_usd=values["pilot_child"],
        locked_child_cap_nano_usd=values["locked_child"],
        price=entry,
    )


def load_price_policy(path: Path) -> PricePolicy:
    if not path.is_file():
        raise ConfigurationError(f"price policy does not exist: {path}")
    try:
        value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("cannot read Phase 4 price policy") from exc
    return policy_from_mapping(value)
