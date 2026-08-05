# Phase status

## Phase 0 — Project contract and deterministic shell

**Outcome: complete.** Phase 0 was implemented and stopped at its declared boundary on August 3,
2026, then received a public-context hardening update on August 4, 2026. The package is rooted at
`src/world_model_search/`; no nested repository was created. The run shell is mock-only and contains
no cellular-automaton logic or live language-model call.

### Build items

- `pyproject.toml` and `uv.lock` define the Python 3.12+ package, pinned runtime/development
  dependencies, the `wms` entry point, and formatting/lint/type/test configuration.
- Strict frozen configuration dataclasses, a fail-before-write YAML loader, structured JSON diagnostic
  logging, immutable canonical manifests/artifacts, a SQLite WAL ledger, and CI commands are present.
- Frozen dataclasses cover internal/public tasks, a minimal public world specification, candidates and
  payloads, oracle feedback/results, proposal context/budget, events, branches, and split labels.
  `internal_family_id` is retained only in internal task/run metadata and is structurally absent from
  proposer context. Protocols cover proposers, archives, and schedulers.
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
# exit 0; 41 files already formatted

uv run ruff check .
# exit 0; All checks passed!

uv run mypy
# exit 0; Success: no issues found in 30 source files

uv run pytest -q
# exit 0; 22 passed

uv run wms solve --config configs/smoke.yaml --proposer mock --run-id phase0-smoke
# exit 0; status=completed, completed_steps=4

uv run python scripts/record_phase0_transcript.py
# exit 0; docs/phase0_cli_transcript.txt

./scripts/ci.sh
# exit 0; format, lint, strict type checking, and 22 tests passed
```

The smoke event payload hashes are, in logical order:

```text
468bb21c1f47ecbd8e45f9bca03895e685e041e792cc33c7f69693ce5957206a
ad0654b457830856f64d888441814462f6fea8360b324df9a74e6a4e34262256
1d19d934e05e891b5811a27ea51bad76867c4943be282ea6a1fec4077fd71ddc
9317d0f48adb97e6d7bcaa387af16032f4af6a6a541f30eb03a7ab367d2e11be
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
not been qualified.

## Phase 1 — World generator and exact oracle

Phase 1 adds all 256 elementary radius-1 binary CA semantics, independent scalar and bulk simulators,
an independent locked-trajectory verifier, deterministic semantic-disjoint split assignment, and
capability-separated public/oracle task bundles. `wms tasks generate --config configs/smoke.yaml`
creates `artifacts/phase1-benchmark`; its manifest and validation report are canonical frozen data.

Executable evidence is in `test_elementary_phase1.py` and `test_phase1_generation.py`: every reference
passes, all 2,048 one-bit mutations fail, and 1,024 explicitly seeded cases establish scalar/bulk and
local-equivalence/rollout agreement including size-one periodic lattices. Generation covers 64 tasks
per split with 256 distinct semantic hashes. Public bundles are scanned for oracle representations,
hashes, and seeds. Validation was consumed once by recorded analysis v1. Test assignment metadata was
audited, but `test_outcomes_accessed` is false and no test outcome was evaluated.

Phase 0 solve/resume/replay/report remains unchanged. `oracle verify` stays fail-closed because a
candidate-file convention would prematurely introduce the Phase 2 DSL. Remaining risks are the small,
enumerable semantic universe (semantic hashes therefore remain oracle-only), fixed seeded properties
rather than Hypothesis, and the existing single-process/local-filesystem qualification.
