# World Model Search

World Model Search is an oracle-grounded research testbed for cumulative synthesis of compact,
executable predictors. The repository now includes the **Phase 4 no-cost implementation**: a
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

The live scientific Phase 4 compatibility gate **passed on the authorized revised training canary**.
The first canary's two attempts exhausted its 512-token output allowance; the versioned correction raised
the per-request allowance to 2,048 without changing the model or dollar ceilings. `PHASE4-LIVE-CANARY-V2`
returned and evaluated the exact two-item strict batch in one call, then replayed offline with zero
provider calls. The cumulative published-rate estimate is $0.00357 across both canaries, with no uncertain
charge. No development pilot exists, and the one-time test registry is therefore not frozen. Fake A/B/C
evidence validates the machinery but is not evidence for H1 or H2.
Learned primitives, interestingness schedulers, active queries, memory, and meta-learning remain absent
as later-phase mechanisms.

The versioned development pilot now retains ten independent development tasks and 256 evaluations per
condition, but uses two search seeds rather than four so its 60 children can each retain the policy's
$0.15 safety ceiling while summed exposure remains $9.00 under the $9.75 stage cap. It requests batches
of four with 2,048 output tokens/request. Its no-cost dry run verifies the exact successful canary record
and reports 3,780 planned logical calls. Existing incomplete hash-matched children resume through the
crash-safe request state machine; the pilot has not been authorized or executed.

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
[`docs/design_decisions.md`](docs/design_decisions.md) for the frozen Phase 0-4 contracts.
