# World Model Search

World Model Search is an oracle-grounded research testbed for cumulative synthesis of compact,
executable predictors. The repository now includes the **Phase 4 implementation**: a
vendor-neutral batched LLM proposer, strict structured AST output, fixed direct/iterative prompts,
exact caching, bounded retries, crash-safe request records, hierarchical token/dollar budgets, a
cumulative cost ledger, A/B/C conditions, a target-blind random baseline, paired analysis, offline
replay, enforced Holm decisions, and frozen reporting. Paid work that cannot reserve its next
worst-case request stops before dispatch as a replayable cost-cap result. Phase 0-3 records and meanings
remain backward-readable.

The frozen Phase 3 comparison completed with a small negative no-worse result, so the phase stops without
tuning; this repository treats reproducible negative evidence as a valid experimental outcome.
Post-result hardening adds total operator coverage, stricter archive/leakage/budget tests, development-only
480-child reproducibility, task-clustered uncertainty diagnostics, and separate CPU/elapsed accounting;
it does not rerun or reinterpret the consumed validation experiment.

The live Phase 4 compatibility gate **passed on the authorized revised training canary**.
The first canary's two attempts exhausted its 512-token output allowance; the versioned correction raised
the per-request allowance to 2,048 without changing the model or dollar ceilings. `PHASE4-LIVE-CANARY-V2`
returned and evaluated the exact two-item strict batch in one call, then replayed offline with zero
provider calls.

The authorized development pilot subsequently completed all 60 children: ten independent development
tasks, two seeds, three conditions, and 256 evaluations per child. Provider-disabled replay, artifact
integrity, request/token/oracle accounting, and ledger reconciliation all passed. H1 (B-A) was not
supported. H2 (C-B) was positive but inconclusive and is not established. Cumulative Phase 4 usage is
$6.5268 at the frozen published-rate policy, with zero uncertain usage and no active reservation. The
user separately reported $4.65 from the provider dashboard; that value is an unverified dashboard
reconciliation, not a replacement for the conservative local ledger.

**Phase 5 engineering and its no-cost smoke are complete.** The repository now adds a separate typed
SQLite memory, capability-safe deterministic retrieval, an exactly coded zero-arity learned-primitive
language, a predeclared structural-family F0 transfer benchmark, strict C/D prompt and manifest
isolation, provider-disabled replay, and a transfer-matrix report. One primitive passed the no-cost
development promotion gate (+46 net bits after charging its 50-bit definition once). The paired
two-seed recoding smoke is machinery evidence only because it reuses those development families; it is
not an H3 test. The subsequent live canary passed compatibility, and the complete 256-request C/D
development pilot replayed cleanly but found zero exact solutions in either arm. Condition D's mean
best score was 5.0 versus C's 5.5, so development does not support H3. The final memory, primitive
registry, and analysis plan are frozen without refitting. No sealed-test access occurred; H3 remains
unconfirmed. Phase 6 interestingness, active queries, hidden state, and self-modifying search remain
absent.

The canary and development authorities record the user's completed-run authorization. The separate
sealed-test declaration binds the final freeze hashes while still denying model and test-oracle access.
Freezing that declaration is not authorization to run the sealed test.

A separate Phase 5 v2 preparation now makes memory experience-derived. It reuses frozen Phase 4
condition-C search as explicitly retrospective single-source-family training, pairs each first-exact
lineage with its three same-request/same-parent failures, and prepares one unexecuted induction request
for each adequately supported parent-cell representation family. Prospective sole-lesson and bundle
validation remain fail-closed pending fresh tasks, a cost forecast, and explicit authorization; the v1
pilot is unchanged. See `docs/phase5_experience_memory_v2.md`.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then run:

```console
uv sync --locked --dev
uv run --locked wms tasks generate --config configs/smoke.yaml
uv run --locked wms oracle verify --task d737b0ee219de6a676c139d1 \
  --candidate examples/phase2-rule90-candidate.json
uv run --locked wms solve --config configs/phase2-smoke.yaml \
  --proposer enumerative --run-id phase2-smoke
uv run --locked wms replay --run phase2-smoke
uv run --locked wms report --run phase2-smoke --out artifacts/reports/phase2-smoke

# Phase 3 training smoke and locked paired experiment:
uv run --locked wms solve --config configs/phase3-smoke.yaml \
  --proposer mutation --run-id phase3-smoke
uv run --locked wms replay --run phase3-smoke
uv run --locked wms report --run phase3-smoke --out artifacts/reports/phase3-smoke
uv run --locked wms benchmark --experiment experiments/phase3-archive-smoke.yaml

# Phase 4 zero-cost engine, offline reproduction, and forecast:
uv run --locked wms solve --config configs/phase4-fake-smoke.yaml \
  --proposer llm --run-id PHASE4-FAKE-SMOKE
uv run --locked wms replay --run PHASE4-FAKE-SMOKE
uv run --locked wms report --run PHASE4-FAKE-SMOKE \
  --out artifacts/reports/PHASE4-FAKE-SMOKE
uv run --locked wms benchmark --experiment experiments/phase4-fake-smoke.yaml
uv run --locked wms benchmark \
  --experiment experiments/phase4-primary-pilot.yaml --dry-run
uv run --locked wms baseline random --config configs/phase4-fake-smoke.yaml --count 256

# Phase 5 family audit, no-cost C/D smoke, and provider-disabled replay:
uv run --locked wms phase5 dry-run
uv run --locked wms phase5 smoke
uv run --locked wms phase5 replay

# Provider-disabled Phase 4-C retrospective experience preparation:
uv run --locked wms phase5 prepare-experience

# Provider-disabled validation of the frozen next-stage designs:
uv run --locked wms phase5 live-dry-run \
  --experiment experiments/phase5-canary.yaml \
  --authority experiments/phase5-canary.pending.yaml
uv run --locked wms phase5 live-dry-run \
  --experiment experiments/phase5-development.yaml \
  --authority experiments/phase5-development.pending.yaml
uv run --locked wms phase5 replay-live \
  --experiment experiments/phase5-development.yaml
uv run --locked wms phase5 finalize-development \
  --experiment experiments/phase5-development.yaml

# The Phase 0 mock lifecycle remains available:
uv run wms solve --config configs/smoke.yaml --proposer mock --run-id smoke
uv run wms replay --run smoke
uv run wms report --run smoke --out artifacts/reports/smoke
```

To exercise interruption and resumption deliberately:

```console
uv run wms solve --config configs/smoke.yaml --proposer mock \
  --run-id resumable --interrupt-after 2
uv run wms solve --resume resumable
```

Live use is never enabled by an API key alone. The frozen canary requires both the CLI flag and the
independent environment opt-in; the key is read only inside the OpenAI transport adapter:

```console
WMS_ALLOW_LIVE_MODEL=1 uv run --locked wms llm canary \
  --config configs/phase4-openai-canary.yaml --allow-live-model
```

The original `PHASE4-LIVE-CANARY` remains immutable failed evidence. The authorized
`PHASE4-LIVE-CANARY-V2` correction uses a 2,048-token per-request maximum, a fresh cache namespace, and
unchanged model, endpoint, tier, retry, and dollar ceilings; it passed strict two-item batch validation.
Do not substitute another model, endpoint, service tier, or provider without explicit authorization. The
code-side dollar figure is a conservative published-rate estimate even when an external promotional
credit makes dashboard cost lower; the separately configured OpenAI dashboard budget remains the billing
backstop. See [`docs/phase4_implementation_report.md`](docs/phase4_implementation_report.md).

Run the complete local CI command with `./scripts/ci.sh`. See
[`docs/phase_status.md`](docs/phase_status.md) for gate evidence and
[`docs/design_decisions.md`](docs/design_decisions.md) for the frozen Phase 0-4 contracts. The
post-Phase-4 [dual-budget policy](docs/dual_budget_policy.md) preserves published-rate scientific
accounting while enforcing the user's personal `$100` ceiling against reconciled provider cash plus
worst-case unreconciled exposure. See the
[`Phase 5 implementation report`](docs/phase5_implementation_report.md) for gate evidence, no-cost
results, limitations, and the pending live-authorization boundary.
