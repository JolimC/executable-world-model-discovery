# World Model Search

World Model Search is an oracle-grounded research testbed for cumulative synthesis of compact,
executable predictors. The repository currently contains **Phase 0 only**: a deterministic project
shell with typed contracts, mock proposal/evaluation, resumable persistence, and replay.

No cellular-automaton simulator, real task split, typed program DSL, search archive, learned scheduler,
or language-model integration is implemented yet.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then run:

```console
uv sync --locked --dev
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
[`docs/phase_status.md`](docs/phase_status.md) for recorded Phase 0 gate evidence.

