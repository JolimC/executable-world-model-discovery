# World Model Search

World Model Search is an oracle-grounded research testbed for cumulative synthesis of compact,
executable predictors. The repository is complete through **Phase 2**: it has the deterministic Phase 0
shell, the Phase 1 elementary-cellular-automaton benchmark/oracle mechanics, and a typed loop-free DSL
with canonicalization, exact interpretation, prefix/residual coding, correctness-first ranking,
cost-ordered enumeration, replay, and frozen analysis/reporting.

Archive search, mutation/crossover, learned primitives, schedulers, active queries, and language-model
integration are deliberately not implemented; those belong to later phases.

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
[`docs/design_decisions.md`](docs/design_decisions.md) for the frozen Phase 2 contracts.
