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

At the Phase 3 handoff, no Phase 4+ mechanism or scientific claim was implemented. Remaining limitations are the 256-semantics
F0 universe, built-in parity/majority macros, two public descriptor probes, a small locked smoke profile,
fixed seeded randomized tests rather than exhaustive syntax-space proofs, application-level rather than
OS-level access auditing, and single-process/local-filesystem persistence qualification.

## Phase 4 — Language-model proposer and primary experiment infrastructure

**Engineering complete; development pilot complete; confirmatory locked test waived; H1/H2 remain
unconfirmed.** Phase 4 adds the executable LLM/search/persistence/analysis machinery. The first
authorized canary's two charged Responses calls reached its 512-output-token limit and failed schema
completion. The authorized versioned correction raised only the per-request output allowance and
derived token caps: its single call returned two strict candidate documents, both validated and
evaluated, and replayed without provider access. The development pilot then completed all 60 children
and passed provider-disabled replay, artifact-integrity, accounting, and ledger audits. H1 was not
supported; H2 was positive but inconclusive. The user deliberately waived the planned locked
confirmatory test after reviewing the pilot. This is a disclosed protocol change from the original
plan, and neither hypothesis is treated as established.

### Frozen implementation contracts

- Package 0.4.0 pins the official `openai==2.53.0` SDK. SDK objects remain inside
  `OpenAIResponsesBackend`; search, persistence, replay, and analysis use immutable provider-neutral
  request, response, usage, error, and backend contracts.
- The model is exactly `gpt-5-mini-2025-08-07` through synchronous `v1/responses`, default service
  tier, low reasoning effort, disabled storage/truncation, and strict JSON-schema output. The current
  official model page documents the snapshot and rates but labels the snapshot deprecated. The canary
  established project access to the exact snapshot and request fields, but not a complete structured
  batch under the frozen output limit. No alias or replacement is used.
- Price policy `phase4-price-and-ceilings-v1` uses integer nano-USD: $0.25/MTok uncached input,
  $0.025/MTok cached input, and $2/MTok output. Reasoning is already a subset of output. Hard ceilings
  are exactly $100 project, $30 Phase 4, $0.25 canary, $9.75 development, $20 locked test,
  $0.01/request, $0.15 pilot child, and $0.50 locked child. Pilot v2 uses 60 children at the full
  $0.15 child ceiling, so all child ceilings still total $9.00. The policy hash reverified on August 11, 2026 is
  `120ca1d0cb66d23230ff8267d4c0eb492421e8de55dc4e1e97950e5cd7fc93fa`.
- Live dispatch needs `--allow-live-model`, `WMS_ALLOW_LIVE_MODEL=1`, a key at the adapter boundary,
  exact price-policy/ledger state, and successful hierarchical reservation. Fake, dry-run, replay,
  reporting, prompt snapshots, and tests do not make provider calls.
- Configuration schema 4, manifest/database schema 5, event/results schema 4, candidate identity v1,
  request-state v1, joint-budget v1, and recorded analysis/report v1 are additive. Schemas 1-3 and all
  Phase 0-3 meanings remain explicit readers.
- A is stateless direct sampling with no parent, score, prior output, or conversation. B and C use the
  same iterative prompt, role, and bounded parent-associated score; C selects one uniform archive branch
  per call. Batch items share the call context and commit in response order.
- Strict parsing rejects fenced/prose/trailing/duplicate JSON, wrong root/version/role/size, unknown or
  ill-typed fields, forbidden ranges/macros, and excessive ASTs. Items validate independently. Accepted
  complete ASTs are canonicalized and evaluated directly; no post-LLM mutation occurs.
- Exact cache identity freezes transport/model/endpoint/tier, exact prompt bytes, schema, role, batch,
  and settings while excluding secrets, paths, timestamps, and provider request IDs. Retries allow one
  identical request only for provider-established zero-usage rate limits or charged malformed
  envelopes. Ambiguous transport/usage failures are terminal. Immutable response/failure evidence and
  usage precede item commits; uncertain dispatch retains its reservation. Every attempt rechecks token
  caps, and an unaffordable paid reservation ends as replayable `cost-cap-exhausted` before dispatch.
- Seven shared initialization evaluations are charged in A/B/C. Duplicate items remain charged. The
  target-blind `random-dsl` baseline reports construction/proposal/oracle/CPU cost and zero model work;
  it is contextual, not token-matched.

### No-cost experiment and presentable artifact

`experiments/phase4-fake-smoke.yaml` executes A/B/C through the real engine on one training task and one
seed. The paired fake outcome is H1 `0.0`, H2 `0.0`, with degenerate `[0.0, 0.0]` machinery-test
intervals. Both clustered two-sided p-values and Holm-adjusted p-values are `1.0`, so neither primary
hypothesis is rejected. The aggregate records `blocked-fake-evidence-only`; this is not a scientific
conclusion.

The generated record-only artifact is
`artifacts/reports/phase4-fake-smoke-v1/phase4-artifact.json`. It links child manifest/results/analysis
hashes, exact prompt/request/response records, budget/cost/token reconciliation, raw paired rows,
H1/H2 intervals, curves, best-program/MDL and lineage sources, validity/duplicate/cache/retry states,
runtime diagnostics, access ledgers, lock/model/prompt/price contracts, and failure analysis. Its
SHA-256 in this verification run is
`73598cfd022c027f30feab54766d4e578f942c753e4350b4c9ce045574c6fbad`; the aggregate deterministic
summary hash is `56cc7c3236910da239f72722c553aa46e4ebe094fe33510420e6054864398320`.
Generated `artifacts/` remain ignored and reproducible.

### Live training-canary evidence

`PHASE4-LIVE-CANARY` made one logical call and the one permitted identical retry. Both physical requests
returned HTTP 200 with 1,400 input and 512 output tokens. The first used 320 reasoning tokens and ended
with truncated JSON; the retry used all 512 output tokens as reasoning and returned no candidate text.
Both are immutable charged `schema-failure` records, with zero valid proposal items and no post-model
repair. This is a compatibility-gate failure even though the run lifecycle reached terminal status
`completed`.

The two estimates are $0.001374 and $0.001086, totaling $0.00246. The ledger records $0.005676 reserved,
$0.003216 released, $0 uncertain, and no active reservation; `reserved = actual + uncertain + released`.
The canary-stage balance is $0.24754, the Phase 4 balance $29.99754, and the project balance $99.99754.
Dashboard-export reconciliation is unavailable; these are conservative published-rate estimates.

Replay ran with the API key and live opt-in removed and HTTP proxies pointed at a closed local endpoint.
It made zero proposer/provider calls and reproduced deterministic summary hash
`76a8591e515991a3ff52afb574790885d39bf7dd4d66efbcab0719f6cd2e855c`. The record-only report is
`artifacts/reports/PHASE4-LIVE-CANARY/summary.json`, canonical SHA-256
`b14778d30b1151f1a0a99c8f923fc7b6cccc1034bb57cf58849f8bfd929df4b3`.

The authorized correction changed `max_output_tokens` from 512 to 2,048, the retry-aware output cap from
2,048 to 4,096, the total-token cap from 52,048 to 54,096, and the cache namespace to
`phase4-canary-output-2048-v2`. It did not change the exact model, endpoint, service tier, prompt/schema,
role, retry count, dollar policy, or $100 project ceiling. `PHASE4-LIVE-CANARY-V2` made one physical
request, used 1,400 input tokens (1,280 cached), 524 output tokens including 384 reasoning tokens, and
returned exactly two valid proposal items. Both were evaluated; one was a canonical/semantic duplicate,
which remains charged behavior. The request completed without retry or uncertain usage.

The corrected request reserved $0.00591025, reconciled to $0.00111 at normal published rates, and
released $0.00480025. Across both canaries the ledger records $0.01158625 reserved = $0.00357 actual +
$0 uncertain + $0.00801625 released, with no active reservation. The remaining balances are $0.24643
canary, $29.99643 Phase 4, and $99.99643 project. Promotional credits are intentionally not deducted from
these conservative ledger amounts; dashboard-export reconciliation remains unavailable.

Provider-disabled replay made zero proposer/provider calls and reproduced deterministic summary hash
`23bba3d953eec518c129e5a226ad3aa82167ac40638b69023720eb90dbf6d6cc`. The corrected record-only report
is `artifacts/reports/PHASE4-LIVE-CANARY-V2/summary.json`, canonical SHA-256
`5be29b06b2434b771a0301c92fbc66c399b71655403848bff5ac9464a461850f`.

### Development pilot and scientific closeout

`phase4-primary-pilot-v2` completed 60/60 development children: ten opaque development tasks, two
search seeds, and conditions A/direct, B/single-incumbent, and C/uniform-diverse-archive. Each child
performed 256 charged evaluations. The run recorded 3,780 logical model calls, 3,797 physical attempts,
17 bounded schema-failure retries, 14,940 valid proposal items, zero invalid items, and 15,360 exact
oracle invocations. All 60 provider-disabled replays passed with zero proposer/provider calls. There
were zero test-oracle invocations.

The primary endpoint was normalized exact-solve AUC over 20 task/seed rows per condition. Mean AUC was
`0.2015625` for A, `0.1927734375` for B, and `0.2892578125` for C. H1 (B-A) was not supported: estimate
`-0.0087890625`, task-clustered 95% interval `[-0.208203125, 0.1470703125]`, Holm-adjusted
`p=0.9293070693`. H2 (C-B) was positive but inconclusive: estimate `0.096484375`, task-clustered 95%
interval `[-0.093359375, 0.326171875]`, Holm-adjusted `p=0.7347265273`. The interval crosses zero and
the multiplicity-adjusted test did not reject, so H2 is not established.

The conservative published-rate equivalent is **$6.5268** cumulatively for Phase 4 (exact ledger
amount `$6.52680755`, including both canaries and the pilot). Uncertain usage is `$0`, and no active
reservation exists. The user separately reported **$4.65** from the provider website. That number is
recorded only as an **unverified dashboard reconciliation**: no provider export or invoice was supplied,
and it does not replace or reduce the frozen published-rate ledger.

The user deliberately waived the locked confirmatory test after the pilot. The non-executable pending
declaration was closed as `skipped-by-user-after-development-pilot`; model calls and test-oracle access
remain prohibited. No locked registry was frozen, no locked child was run, and test outcomes were never
accessed. Skipping confirmation is scientifically acceptable as a disclosed protocol change, but it
leaves H1/H2 unconfirmed and cannot support a claim that H2 is established.

### Gate disposition

| Gate | Evidence | Disposition |
|---|---|---|
| Provider-neutral strict proposer | scripted/OpenAI boundary plus schema, prompt, cache and leakage tests | Pass, no-cost |
| Budget/cache/retry/cost reconciliation | integer price/usage, joint caps, concurrent/idempotent ledger, malformed retry, uncertain failure, pre-dispatch cost-cap tests | Pass, no-cost |
| Resume, replay, frozen report | every request boundary plus mid-batch; replay disables provider/cache; report disables oracle | Pass, no-cost |
| Matched A/B/C and H1/H2 analysis | strict registries, rotating blocks, clustered/sensitivity bootstrap, enforced Holm, negative/null/positive tests | Pass as machinery only |
| Live canary and replay | corrected 2,048-token request returned two valid items; exact provider-disabled replay and ledger audit pass | Pass, live training evidence |
| Development pilot/power review | 60/60 children complete; provider-disabled replay, artifact/hash, request/token/oracle, ledger, cost, runtime, variance, and power audits complete | Pass as development evidence; H1 unsupported, H2 positive but inconclusive |
| One-time locked test | closed pending declaration prohibits model/test access; access ledgers record zero test calls and unconsumed test authority | Deliberately waived by user; outcomes never accessed |

Final local verification covers 97 tests: the complete Phase 0-4 suite, the preserved 480-child Phase 3
aggregate, Phase 4 boundary/leakage/accounting tests, Ruff formatting/lint, and strict mypy over 64
source files. The stable fake run records 11 events and replays with zero proposer invocations; its
deterministic summary hash is
`0f8c1a6e48321bc2969437863a40b547d93c4bc5f65954a4a4f5f50019255a31`.

### Reproduction commands

```console
uv sync --locked --dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy
uv run --locked pytest

uv run --locked wms solve --config configs/phase4-fake-smoke.yaml \
  --proposer llm --run-id PHASE4-FAKE-SMOKE
uv run --locked wms replay --run PHASE4-FAKE-SMOKE
uv run --locked wms report --run PHASE4-FAKE-SMOKE \
  --out artifacts/reports/PHASE4-FAKE-SMOKE
uv run --locked wms benchmark --experiment experiments/phase4-fake-smoke.yaml
uv run --locked wms benchmark \
  --experiment experiments/phase4-primary-pilot.yaml --dry-run
uv run --locked wms baseline random --config configs/phase4-fake-smoke.yaml --count 256
```

Evidence remains F0-only: binary radius-one elementary cellular automata under the small frozen DSL.
F1/F2 were not backfilled, limiting external validity and any claim of broad world-model discovery.
Phase 4 adds no cross-task memory, learned primitive, interestingness/UCB scheduler, active query,
hidden state, scaffold mutation, or Phase 5-8 claim. No Phase 5 mechanism has been implemented yet.
The live pilot is development evidence only—not confirmation of H1/H2, broad discovery, transfer, or
general model superiority.
