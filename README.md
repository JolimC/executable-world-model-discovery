# World Model Search

World Model Search is an oracle-grounded research testbed for cumulative synthesis of compact,
executable predictors. The repository is complete through **Phase 3**: in addition to the deterministic
shell, exact elementary-cellular-automaton oracle, and typed loop-free DSL, it has deterministic typed
mutation/crossover, a MAP-Elites archive with lineage reserve, a matched single-incumbent control,
uniform branch scheduling, exhaustive attempt/oracle budgets, recorded replay, and a frozen paired
validation experiment.

The frozen Phase 3 comparison completed with a small negative no-worse result, so the phase stops without
tuning; this repository treats reproducible negative evidence as a valid experimental outcome.
Post-result hardening adds total operator coverage, stricter archive/leakage/budget tests, development-only
480-child reproducibility, task-clustered uncertainty diagnostics, and separate CPU/elapsed accounting;
it does not rerun or reinterpret the consumed validation experiment.

Learned primitives, surrogate schedulers, active queries, memory/meta-learning, and language-model
integration remain deliberately absent; those belong to later phases.

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

Run the complete local CI command with `./scripts/ci.sh`. See
[`docs/phase_status.md`](docs/phase_status.md) for gate evidence and
[`docs/design_decisions.md`](docs/design_decisions.md) for the frozen Phase 3 contracts.
