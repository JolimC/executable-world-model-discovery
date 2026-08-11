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

At the Phase 1 boundary, Phase 0 solve/resume/replay/report remained unchanged and `oracle verify`
stayed fail-closed because a candidate-file convention would prematurely introduce the Phase 2 DSL.
Remaining risks were the small,
enumerable semantic universe (semantic hashes therefore remain oracle-only), fixed seeded properties
rather than Hypothesis, and the existing single-process/local-filesystem qualification.

## Phase 2 — Typed DSL, canonicalization, and MDL

**Outcome: complete.** Phase 2 implements the typed binary radius-1 language and strict JSON surface,
total interpreter, task-independent canonicalizer, Phase-1-compatible semantic identity, uniquely
decodable prefix code, residual/two-part MDL, correctness-first rank, cost-ordered enumerator, fully
charged truth-table baseline, typed exact oracle, and recorded run/replay/report evidence. It stops
before Phase 3: there is no mutation, crossover, archive behavior, scheduler policy, memory, active
query, or model proposer.

### Frozen contracts and bounds

- DSL `binary-ca-radius1-dsl-v1`; candidate schema 1; interpreter
  `binary-ca-radius1-interpreter-v1`; canonicalizer `binary-ca-canonicalizer-v1`.
- Semantic hash `elementary-local-semantics-v1`, identical to the Phase 1 ordered `000..111` payload
  and kept oracle/internal-only.
- Prefix code `binary-ca-prefix-v1`; residual `enumerative-residual-gamma-v1`; rank
  `correctness-first-rank-v1`; literal baseline `elementary-truth-table-v1` at 19 bits.
- DSL limits: depth 8, nodes 63, cases 8. Enumerator limits: 36 bits, depth 8, nodes 15, 50,000 raw
  examinations. Order is bit cost then canonical JSON; duplicates retain their first representative.
- Configuration schema 2; run manifest schema 3; database/candidate/event/results schema 2; analysis
  artifact `phase2-analysis-bundle-v1`. Phase 0 configuration schema 1 and run-manifest schema 2 retain
  explicit readers.

### Gate evidence

| Gate | Executable/recorded evidence | Result |
|---|---|---|
| Deterministic uniquely decodable coding | `test_phase2_codec.py`; 256 catalog and 512 seeded nested round trips in `analysis/gate-report.json`; opcode/complete-value prefix checks; truncation, extension, type, range, and noncanonical rejection | Pass |
| Canonicalization preservation/idempotence | `test_phase2_interpreter_canonical.py`; recorded seeds 0–511 span all 17 constructors and compare 4,096 exhaustive before/after cases; explicit fold/order/idempotence/double-negation/dead-branch examples | Pass |
| Rule 90, Rule 150, majority, parity | `test_phase2_mdl_enumerator.py`; target-blind indices 10, 6, 5, and 6 respectively, all 8 bits, plus standard XOR forms | Pass |
| Total interpreter and CA agreement | finite AST/case limits; all 256 literal tables/hashes and 1,024 Phase 2 rollout transitions; retained Phase 1 scalar/bulk/independent checks | Pass |
| Residual/MDL and rank | every `e=0..8`, invalid/nonbinary boundaries, 256 randomized correctness-dominance cases, and exact-length-before-runtime checks | Pass |
| Enumerator determinism/order/duplicates | two full enumerations and independent runs; 29,529 examined, 256 emitted, 12,858 canonical and 16,123 semantic collapses; monotone; cap not hit | Pass |
| Leakage/capability boundary | exact public types; parse-before-oracle; forbidden capability monkeypatches; all Phase 2 public bundles cover only `000,111`; nested training values scanned | Pass |
| Lifecycle/replay/frozen report | Phase 0 regressions plus `test_phase2_lifecycle.py` for interrupt/resume, zero-proposer replay, stable independent events/results, versioned persistence, and reporting with live oracle/enumerator disabled | Pass |

The enumerator finds a structured representative for every elementary semantic. Lengths range 4–33
bits: 64 are shorter than the 19-bit truth-table baseline and 192 are longer. The artifact retains both
lengths instead of implying that every structured program compresses its literal baseline.

### Reproduction commands and artifacts

```console
uv sync --locked --dev
uv run --locked wms tasks generate --config configs/smoke.yaml
uv run --locked wms oracle verify --task d737b0ee219de6a676c139d1 \
  --candidate examples/phase2-rule90-candidate.json
uv run --locked wms solve --config configs/phase2-smoke.yaml --proposer enumerative \
  --run-id PHASE2-LOCKED-SMOKE
uv run --locked wms replay --run PHASE2-LOCKED-SMOKE
uv run --locked wms report --run PHASE2-LOCKED-SMOKE \
  --out artifacts/reports/PHASE2-LOCKED-SMOKE
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy
uv run --locked pytest
./scripts/ci.sh
```

The recorded run is under `artifacts/runs/PHASE2-LOCKED-SMOKE`. Its frozen `analysis/` contains
`elementary-complexity.{json,csv,svg}`, `collapse-examples.json`, `gate-report.json`,
`access-ledger.json`, and a content-hash manifest. The report copies them to
`artifacts/reports/PHASE2-LOCKED-SMOKE` without live computation. `artifacts/` remains ignored; the
commands above regenerate the evidence on a clean checkout.

### Split use, validation state, and leakage correction

The exact-oracle smoke uses training task `d737b0ee219de6a676c139d1` (Rule 90). Development is allowed but
unused. Phase 2 did not consume validation, and its application access ledger records no validation or
test oracle artifact. Rule 150 is assigned to test, so its required recovery and the all-256 plot are
standalone public language mechanics, not a task-level oracle call or held-out performance claim.

Legacy Phase 1 random traces frequently cover all eight F0 neighborhoods, making semantics
reconstructable. Phase 2 does not rewrite them: it generates and exclusively loads the versioned
`artifacts/phase2-benchmark` bundle whose public coverage is exactly `000` and `111`. DD-015 records
this deliberate compatibility/leakage resolution.

### Limitations and claims

This phase establishes representation, exact verification, compression accounting, exhaustive baseline
enumeration, and reproducible lifecycle infrastructure. It does not test H1/H2 or show that a loop,
archive, language model, scheduler, memory, or learned primitive improves search. F0 has only 256
semantics and enumeration recovers all, so secrecy and search difficulty remain limited. Runtime is
diagnostic; no cross-host timing order is claimed. Fixed seeded properties are used instead of
Hypothesis. The database remains single-process/local-filesystem qualified.

## Phase 3 — Deterministic mutation, archive search, and paired validation

**Outcome: complete with a negative gate result; stopped without tuning.** The 240 paired cases give a
diverse-minus-incumbent normalized exact-solve AUC of `-0.000390625`, with deterministic paired-bootstrap
95% interval `[-0.002994791666666667, 0.0018229166666666667]`. The point estimate fails the frozen
zero-tolerance no-worse gate, and the interval does not establish superiority.

Phase 3 implements the smallest executable comparison between a diversity-preserving MAP-Elites search
and a matched single-incumbent search. Both use the same seven charged public DSL baselines, stateless
counter RNG, typed operators, exact oracle, score-only response, proposal/oracle caps, and
continue-after-exact behavior. The only intended experimental difference is branch retention/selection.

The archive uses syntax plus fixed public-probe descriptors, separate exact/partial layers, one elite per
cell, and a bounded lineage reserve. Uniform scheduling records its eligible set and exact selection
probability. Attempt and oracle accounting remains separate: invalid/no-op attempts consume proposal
budget, while all emitted evaluations—including duplicates—consume oracle budget. The locked experiment's
SQLite schema 3 and manifest schema 4 record proposal attempts, evaluations, transitions,
parent-ordered lineage, budget states, events, all versioned policies, and validation authority. New runs
use the backward-readable SQLite schema 4 diagnostic extension described below.

Executable evidence covers all operators and typed path classes, depth/node/integer boundaries,
descriptor-family reachability, randomized archive monotonicity against a reference implementation,
reserve behavior, cross-task rejection, strict configuration/registry loading, hidden-field scans,
interruption/resume, independent deterministic execution, zero-generation replay, and frozen-data
reporting. The locked registry contains 12 publicly selected opaque validation task IDs and 20 seeds
(240 exact task/seed pairs; 480 child runs), with 96 proposal attempts and 32 charged oracle calls per
child. Validation/test discipline and the final paired outcome are recorded in the generated experiment
artifacts described below.

All 480 children completed and reconciled exactly to 32 oracle calls each (15,360 total); every child
records zero language-model calls. The incumbent solved 3/240 runs and had mean normalized exact AUC
`0.00390625`; the diverse archive solved 4/240 and had mean AUC `0.003515625`. Diverse archive coverage
averaged `8.083333333333334` occupied coordinates at the final budget, versus the control's intentionally
absent archive. That coverage is a diversity diagnostic, not correctness evidence. All 17 aggregate
files passed their content-hash audit, and the application ledger reports zero test-oracle access.

### Reproduction commands and presentable artifacts

```console
uv sync --locked --dev
uv run --locked wms tasks generate --config configs/smoke.yaml
uv run --locked wms solve --config configs/phase3-smoke.yaml \
  --proposer mutation --run-id phase3-smoke
uv run --locked wms replay --run phase3-smoke
uv run --locked wms report --run phase3-smoke --out artifacts/reports/phase3-smoke
uv run --locked wms benchmark --experiment experiments/phase3-archive-smoke.yaml
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy
uv run --locked pytest
./scripts/ci.sh
```

The experiment writes its immutable freeze and summary beneath
`artifacts/experiments/phase3-archive-smoke-v1-locked/`; the content-hashed paired report is copied to
`artifacts/reports/phase3-archive-smoke-v1-locked/`. Each child run contains lineage JSON/DOT, exact and
coverage curves, operator diagnostics, budget reconciliation, an access ledger, and an analysis
manifest. Aggregate outputs retain raw JSON/CSV rows, paired differences, bootstrap interval, solve and
coverage SVG/CSV curves, operator outcomes, showcased lineages, failure analysis, child-contract hashes,
and an explicit zero-test-access ledger. Artifacts are ignored and regenerable from the frozen registry.

An initial pre-oracle freeze attempt used the unsuffixed output directory and then failed because child
run roots incorrectly affected benchmark discovery. No hidden artifact was opened and no validation
outcome was consumed. The path rule was corrected to bind schema-3 runs to the repository's frozen
Phase 2 benchmark; the preserved aborted freeze is superseded by the `-locked` registry roots.

After all locked children completed, aggregate CSV publication exposed a projection bug: raw JSON rows
also contain the JSON-only transition-outcome mapping. The fix ignores nonselected fields only for the
CSV projection. The final analysis was regenerated from immutable results without child reexecution or
oracle access; `analysis-amendment.json` freezes both source hashes, the narrow change, and unchanged
mechanism/statistical contracts. Its SHA-256 is
`8327d4f7e6b5e61dcd646e2b8ce95a646b53ce2e48345bc7b472ba6f25ba2f0f`; the final analysis-manifest hash
is `450228ee745aac4af4fbefe5784f15f88657602c06a30ed82cec23b488fc8780`.

The recorded-evidence supplement under `analysis/supplement/` explicitly marks archive-invariant and
20-seed reproducibility gates passed, the archive no-worse gate negative/failed, and all five additional
regression gates passed. It records 18,480 proposal attempts, all five archive transition outcomes, and
per-child manifest/result/event/proposal/database/analysis hashes. The evidence-amendment hash is
`2aa5ddc5e2f180bc1dad331fa214f100283861e230d0f956fd6613f3ac1d3cd1`; the supplement-manifest hash is
`530e403ff3fae5b5c3ba1c165f2539be52675447ea4c09ded8ffd4e756c2020d`.

### Post-result implementation hardening

Phase 3's mechanics and evidence have been strengthened without reopening the consumed validation set.
The runner now uses the shared proposer/archive/scheduler interfaces directly; `TruthTable` local
mutation is total and flips one chosen output bit; and new tests exercise invalid/no-op proposal-cap
exhaustion, exact descriptor-bin boundaries, lexical ties, elite/reserve replacement and eviction, and
the actual public context delivered to the proposer. The full 480-child aggregate is executed twice in
an integration test using development tasks, and its deterministic contracts match exactly.

New runs record per-attempt process CPU and elapsed time, the oracle contribution to each, and LLM
call/token counts. Reports copy and hash these values as `runtime-diagnostics.json`; they remain outside
deterministic replay hashes because timing depends on the host. Phase 3's LLM counts remain zero. Future
paired analyses also emit a whole-task-cluster bootstrap interval alongside the original task-seed-pair
interval so repeated seeds on the same task are not treated as fully independent in that sensitivity
view. The locked point estimate, original interval, gate decision, child runs, and artifacts above were
not recomputed or modified.

No Phase 4+ mechanism or scientific claim was implemented. Remaining limitations are the 256-semantics
F0 universe, built-in parity/majority macros, two public descriptor probes, a small locked smoke profile,
fixed seeded randomized tests rather than exhaustive syntax-space proofs, application-level rather than
OS-level access auditing, and single-process/local-filesystem persistence qualification.
