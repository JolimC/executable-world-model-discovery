# Phase status

## Phase 0 — Project contract and deterministic shell

**Outcome: complete.** Phase 0 was implemented and stopped at its declared boundary on August 3,
2026. The package is rooted at `src/world_model_search/`; no nested repository was created. The run
shell is mock-only and contains no cellular-automaton logic or live language-model call.

### Build items

- `pyproject.toml` and `uv.lock` define the Python 3.12+ package, pinned runtime/development
  dependencies, the `wms` entry point, and formatting/lint/type/test configuration.
- Strict frozen configuration dataclasses, a fail-before-write YAML loader, structured JSON diagnostic
  logging, immutable canonical manifests/artifacts, a SQLite WAL ledger, and CI commands are present.
- Frozen dataclasses cover internal/public tasks, candidates and payloads, oracle feedback/results,
  proposal context/budget, events, branches, and split labels. Protocols cover proposers, archives, and
  schedulers.
- `MockProposer`, `MockOracle`, and the `phase0-no-ca-fixture` exercise start, transactional event
  recording, interruption, resumption, finalization, read-only reporting, and artifact-based replay.
- The CLI includes the Section 10 surface. Later-phase commands fail closed and nonzero; Phase 0 solve,
  resume, replay, and frozen reporting are executable.

### Gate evidence

| Gate | Executable evidence | Recorded evidence | Result |
|---|---|---|---|
| Start, interrupt, resume, replay | `tests/integration/test_run_lifecycle.py::test_run_can_interrupt_resume_and_replay_deterministically` and `tests/regression/test_deterministic_replay.py` | `docs/phase0_cli_transcript.txt`: `transcript-resume` stops after 2/4 events, resumes to 4/4, then replays all four hashes with `proposer_invocations: 0` | Pass |
| Same seeds produce identical event payload hashes | `tests/integration/test_run_lifecycle.py::test_identical_seeds_have_identical_event_payload_hashes` | Independent `transcript-seed-a` and `transcript-seed-b` runs record the identical four SHA-256 hashes; transcript records `same-seed event payload hashes equal: true` | Pass |
| Invalid configuration fails before run creation | `tests/integration/test_invalid_config_gate.py::test_invalid_configuration_creates_no_run_artifacts` | Transcript records exit 2 for `max_steps: 0` and `invalid run artifact exists: false` | Pass |

The deterministic replay regression additionally monkeypatches proposer generation to fail if called.
Replay succeeds, establishing that it consumes recorded proposal artifacts rather than silently
regenerating them. The report integration test similarly disables proposer and oracle calls and produces
a report from frozen data.

### Commands and results

Commands were run from the repository root. `uv` 0.8.4 used CPython 3.14.4 in this development
environment; Ruff and mypy target Python 3.12, and CI explicitly provisions Python 3.12.

```console
uv sync --locked --dev
# exit 0; 14 packages resolved, project and 13 dependencies installed

uv run ruff format --check .
# exit 0; 40 files already formatted

uv run ruff check .
# exit 0; All checks passed!

uv run mypy
# exit 0; Success: no issues found in 30 source files

uv run pytest -q
# exit 0; 20 passed

uv run wms solve --config configs/smoke.yaml --proposer mock --run-id phase0-smoke
# exit 0; status=completed, completed_steps=4

uv run python scripts/record_phase0_transcript.py
# exit 0; docs/phase0_cli_transcript.txt

./scripts/ci.sh
# exit 0; format, lint, strict type checking, and 20 tests passed
```

The smoke event payload hashes are, in logical order:

```text
4807c6fe0c206b9af72cd2fbcdc1d01140568f890ef90873620cf8aa46926268
dce176bdadf4c30b360ffbc71bf5a58c932c94bfb1c2985244ef508bc01e7210
4cc0a19a7de9fef796bd23e3cb600984c2df4cfc7865d2299100d5106bd49444
5f12926b6f3a26ed4565b5e67b0d6b0a46ba89d89038cbb9ad90c0f1ad1f4f6b
```

### Presentable artifacts

- One-page architecture summary: `docs/architecture_phase0.md`
- Generated CLI transcript: `docs/phase0_cli_transcript.txt`
- Deterministic replay test: `tests/regression/test_deterministic_replay.py`
- Frozen-data report command: `wms report --run RUN_ID --out OUTPUT`

### Deterministic hashing boundary

Canonical event payloads include logical step and task identity; deterministic candidate identity,
proposal content hash, parent/proposer/operator/context fields; and oracle correctness/feedback fields.
They exclude run/database IDs, timestamps, paths, runtime/memory/platform/Git diagnostics, process IDs,
and UUIDs. See `docs/design_decisions.md` DD-004 for the full boundary.

### Deviations and limitations

The documented deviations are in `docs/design_decisions.md` DD-006. In summary, later-mechanism tables
and semantic fields are deferred until their owning phases, the AST/oracle are explicit non-CA fixtures,
and later CLI operations are fail-closed skeletons rather than fake implementations.

No real task split was generated and no split-based experiment, semantic deduplication, leakage
experiment, exact oracle, rollout, DSL, MDL code, archive, scheduler, memory, benchmark, or model call
was performed. Split labels are immutable metadata only. SIGINT handling is implemented, while recorded
gate evidence uses `--interrupt-after` so interruption occurs reproducibly at a transaction boundary.
SQLite is tested within one local process lifecycle; cross-host filesystems and concurrent writers have
not been qualified. Phase 1 has not begun.
