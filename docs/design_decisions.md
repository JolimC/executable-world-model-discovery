# Design decisions

## DD-001 — Standard-library CLI and strict YAML loader

The CLI uses `argparse`; YAML is the only runtime dependency. Configuration is decoded manually into
frozen dataclasses with exact key sets. This keeps the shell small and makes the validation-before-write
boundary auditable. Repository-relative run roots are required to prevent configuration from directing
artifacts outside the repository.

## DD-002 — Public task capability type

`Task` and `PublicTask` are separate frozen types. `Task.public_view()` is the sole conversion used to
build `ProposalContext`. The internal type records `internal_family_id` for generation, split management,
and analysis. The public type has no family field; it carries a `PublicWorldSpec` limited to the mechanics
needed to interpret observations, successors, and candidate payloads. Fine-grained generator labels can
therefore not silently become proposer hints. The public type also cannot represent task seeds, hidden
artifact IDs, exact cases, or locked rollout suites. This is stronger than depending on a serializer
deny-list.

If later world specifications require dimension, alphabet, neighborhood, or boundary semantics, those
properties should be added explicitly to `PublicWorldSpec`. A narrow internal label such as a rule
grammar, symmetry class, or parity family must remain internal unless family visibility is declared as
an experimental factor.

This separation changes the serialized public context and therefore intentionally advances the run
manifest schema to version 2. Resume, replay, and reporting reject older schemas before writing or
mixing events across the context boundary; reproducing a schema-1 run requires its original code.

The configured master seed is deterministically domain-separated with SHA-256 into distinct internal
task and proposer RNG seeds. The proposer seed in `ProposalBudget` is explicit reproducibility input,
never equals the hidden task seed, and is not present in `ProposalContext`. No oracle-only value is
derived from the proposer seed or exposed in proposer context.

## DD-003 — Immutable manifest, mutable SQLite lifecycle

The resolved run manifest is created once and never updated. Interruption/completion status and
`next_step` live in SQLite so that resumability does not mutate the manifest. Proposal outputs and final
results are immutable canonical JSON artifacts. Diagnostic timestamps and evaluation timings remain in
SQLite/logs outside deterministic hashes.

## DD-004 — Deterministic event hashing boundary

The canonical event payload contains the schema version, logical step, task identifier, deterministic
candidate fields (content identity, proposal artifact hash, parents, proposer, operator, and public
context hash), and deterministic oracle correctness/feedback fields. SHA-256 is computed only over the
canonical JSON payload.

The following are expressly excluded: event database IDs, run IDs, audit timestamps, absolute and
artifact paths, oracle `runtime_ns`, runtime/memory diagnostics, wall/CPU time, host OS/hardware, Git
state, process IDs, and random UUIDs. Those fields may be recorded for audit diagnostics but cannot
change an event payload hash. Final deterministic metrics likewise exclude timing.

## DD-005 — Replay starts from recorded proposals

Replay does not instantiate or call a proposer. It loads immutable proposal JSON, verifies its content
hash, reconstructs candidate identity, invokes only the deterministic recorded-version mock evaluator,
compares event payload bytes/hashes, and rebuilds final metrics. There is no silent generation fallback.

## DD-006 — Phase-scoped deviations from the proposal

- The Section 9 SQLite schema is implemented only for Phase 0's active entities: run lifecycle, task,
  candidate, evaluation, and event. Archive entries, breakthroughs, scheduler predictions, memory items,
  and meta changes are deferred with their mechanisms; creating empty speculative tables now would
  prematurely freeze later schemas. Candidate semantic hashes and evaluation peak-memory fields are
  likewise deferred because Phase 0 has neither the semantic DSL nor a memory measurement contract;
  proposal content hashes and runtime diagnostics are recorded instead.
- `RuleExpr` is an explicitly opaque fixture node rather than the Phase 2 typed DSL. Its bit count is a
  mock diagnostic and makes no MDL claim.
- `tasks generate`, real `oracle verify`, and `benchmark` are present as fail-closed CLI skeletons. They
  return a clear nonzero “later phase” result. `solve` supports only `mock`; enumerative and LLM
  proposers are not stubbed as misleading implementations.
- A deliberate `--interrupt-after` option provides reproducible interruption evidence in addition to
  `KeyboardInterrupt` handling. It stops only after a fully committed event transaction.
- The proposal lists `runtime_ns` but not peak memory in the illustrative `OracleResult`; Phase 0 follows
  that exact shape. Memory measurement remains a future diagnostic.

## DD-007 — Split preparation only

The immutable split enum represents training, development, validation, and test labels. It is metadata
only. Phase 0 performs no split generation, semantic deduplication, leakage experiment, locked-outcome
access, or split-based evaluation.

## DD-008 — Phase 1 elementary semantics and independent execution

Rule bit `4*left + 2*center + right` supplies the output, so ordered semantics are `000` through `111`;
Rules 30, 90, 110, and 150 are constants over that representation. Lattice indices increase left to
right, both ends wrap periodically, and updates are synchronous. The scalar implementation performs
explicit modular indexing and calls the rule; the bulk implementation rotates immutable sequences and
indexes a lookup table. The rollout verifier independently decodes rule bits while comparing locked
successive states, avoiding shared simulator or local-equivalence machinery.

## DD-009 — Phase 1 identity, splits, and authority boundary

The semantic hash is canonical SHA-256 over domain `elementary-local-semantics-v1` and the eight outputs
in `000..111` order; names, rule numbers, paths, IDs, and seeds are excluded. A domain-separated seed
shuffles all 256 semantics once and assigns shuffled positions round-robin to training, development,
validation, and test. Duplicate semantics therefore cannot cross splits. Public IDs derive from opaque
shuffled slots rather than rule numbers.

Public v1 artifacts contain only an opaque ID, split, CA mechanics, and demonstrations. Oracle v1
artifacts separately contain reference semantics, internal family, domain-separated task/rollout/initial
seeds, semantic hash, and locked trajectory; only `load_hidden_task` is the typed oracle-authority load
surface. Leakage tests scan serialized public bundles as well as their schema. Benchmark manifest schema
3 is distinct from Phase 0 run-manifest schema 2, so existing resume/replay meaning is unchanged.

Generator, simulator, oracle, rollout, split, artifact, and analysis versions are frozen in the
benchmark manifest. Validation v1 is marked consumed. Test semantics participate only in structural
assignment/deduplication auditing; outcomes are neither invoked nor reported. `oracle verify` remains
fail-closed until the Phase 2 candidate language exists.

## DD-010 — Phase 2 typed grammar, JSON, and totality boundary

The Phase 2 language is `binary-ca-radius1-dsl-v1`, restricted to a one-dimensional binary lattice,
radius one, offsets `(-1, 0, 1)` in public `left, center, right` order, periodic boundaries, and
synchronous updates. Frozen immutable node classes are:

- `BitExpr`: `Const`, `At`, `Not`, `And`, `Or`, `Xor`, `If`, `Parity`, `Majority`, and the dedicated
  `TruthTable` baseline constructor;
- `IntExpr`: `IntConst`, `Count`, and `AddConst`;
- `PredExpr`: `Eq`, `Le`, `Ge`, and `Between`.

`If.condition` is exactly `PredExpr`; both branches are `BitExpr`. Masks are nonempty sorted unique
subsets of `(-1, 0, 1)`. `IntConst` and `AddConst.amount` are integers in `[-3, 3]`; booleans are never
integers. `Parity` is sum modulo two. `Majority` is one when the selected count is at least
`ceil(mask-size / 2)`. `Between(value, lower, upper)` is inclusive. Default language limits are depth
8, 63 nodes, and exactly eight exhaustive cases; the recorded enumerator uses a narrower 15-node bound.

Candidate JSON schema 1 is an exact-field object containing `candidate_schema_version`, `dsl_version`,
and `ast`. Nodes use an explicit `op` plus named children/fields. The decoder rejects duplicate or
trailing data, missing/unknown fields, unknown versions/opcodes, wrong child types, bool-as-int,
invalid masks/offsets/ranges/macros, and structural-limit excess. Candidates are data only. The
interpreter exposes no import, file, environment, network, clock, random, reflection, or subprocess
capability. Finite AST and case limits establish totality; wall-clock timing is diagnostic only.

`ElementaryPublicWorldSpec` is a separate Phase 2 public capability type. This leaves the serialized
Phase 0 `PublicWorldSpec` and schema-2 context hashes unchanged. The legacy opaque
`RuleExpr`/`CandidatePayload` remains the explicit Phase 0 reader and is never reinterpreted as DSL.

## DD-011 — Canonicalization and semantic identity

Canonicalizer `binary-ca-canonicalizer-v1` is a bottom-up, size-decreasing or order-orienting pass. Its
ordering key is `(node class name, canonical AST JSON)`. It recursively canonicalizes children; sorts
`And`, `Or`, `Xor`, and `Eq`; folds Boolean constants; applies `And`/`Or` idempotence and
`Xor(x,x)=0`; removes double negation; removes `AddConst(_,0)` and combines/folds small constants when
the result stays in range; replaces single-input parity/majority with `At`; and removes equal or
statically selected `If` branches. There are no reverse rewrites, target-dependent choices, or hidden
queries. Macros of size two or three are retained.

Candidate identity and primary code length use the canonical AST. A noncanonical source is retained
only as analysis data. Semantic identity hashes canonical JSON with domain
`elementary-local-semantics-v1` and outputs in `000..111` order. This intentionally uses the exact Phase
1 payload, so a DSL expression and `ElementaryRule` with the same semantics have the same SHA-256.
Semantic hashes are internal evaluation/run-analysis metadata; they are absent from public task types,
public bundles, proposal contexts, oracle feedback, and CLI verification output.

## DD-012 — Prefix code, residual code, and two-part MDL

Prefix code `binary-ca-prefix-v1` is an unpadded bit string. The prefix-free opcodes are:

| Node | Opcode | Paid fields after opcode |
|---|---:|---|
| `Const` | `000` | one value bit |
| `At` | `001` | offset: `00=-1`, `01=0`, `10=1`; `11` invalid |
| `Not` | `0100` | one encoded `BitExpr` |
| `And` | `0101` | two encoded `BitExpr` children |
| `Or` | `0110` | two encoded `BitExpr` children |
| `Xor` | `0111` | two encoded `BitExpr` children |
| `If` | `10000` | one `PredExpr`, then two `BitExpr` branches |
| `Parity` | `10001` | three mask-membership bits; `000` invalid |
| `Majority` | `10010` | three mask-membership bits; `000` invalid |
| `IntConst` | `10011` | three-bit biased value `value + 3`; `111` invalid |
| `Count` | `10100` | three mask-membership bits; `000` invalid |
| `AddConst` | `10101` | one `IntExpr`, then a three-bit biased amount |
| `Eq` | `10110` | two encoded `IntExpr` children |
| `Le` | `10111` | two encoded `IntExpr` children |
| `Ge` | `11000` | two encoded `IntExpr` children |
| `Between` | `11001` | value, lower, and upper `IntExpr` children |
| `TruthTable` | `11111111110` | all eight outputs in `000..111` order |

Fixed arity supplies tree boundaries; every scalar, mask, and child remains paid. The decoder expects a
root `BitExpr`, consumes exactly one value, re-canonicalizes/re-encodes, and rejects truncation,
extension, invalid codewords, type mismatches, out-of-range fields, and noncanonical streams. Thus
`ast_bits` is exact codec length. The truth-table baseline is 19 bits (11-bit opcode plus eight outputs),
is interpreted and verified like every candidate, and is excluded from structured enumeration.

Residual code `enumerative-residual-gamma-v1` defines `L_N(e)` as the Elias-gamma length of `e+1`:
`2*floor(log2(e+1))+1`. Binary residual length adds `ceil(log2(binomial(N,e)))`; a combination count of
one contributes zero location bits. Hence `e=0` costs one bit, while `e=N` costs only the gamma length.
Invalid `e,N` fail. Alphabet size `q>2` adds `e*ceil(log2(q-1))`; Phase 2 otherwise remains binary.
Two-part MDL is canonical `ast_bits + residual_bits`.

## DD-013 — Rank, enumeration, and all-256 baseline analysis

Rank `correctness-first-rank-v1` compares type validity, totality, negative local errors, full exactness,
negative canonical AST bits, and then an optional negative runtime diagnostic. Runtime is excluded from
deterministic rank/events/replay by default. If explicitly requested, it is only a same-host final tie
breaker; no cross-host deterministic claim is made.

Enumerator `cost-ordered-semantic-first-v1` builds typed subtrees bottom-up by exact prefix-code cost,
then breaks ties by canonical AST JSON UTF-8 lexical order. It canonicalizes before insertion, retains
the first representative of each complete eight-case semantic signature, and counts canonical and
semantic duplicates. Truth-table literals are excluded. There are no targets, random decisions,
mutation, crossover, archive, or scheduler. Locked bounds are 36 bits, depth 8, 15 nodes, and 50,000
raw candidates. The gate exhausts work at 29,529 examined candidates without hitting the cap and emits
all 256 semantics. Structured lengths range 4–33 bits: 64 beat the 19-bit literal baseline and 192 do
not. “Not found within bounds” remains distinct from “unrepresentable.”

The standalone public mechanics analysis recovers majority at index 5, parity/Rule 150 at index 6, and
Rule 90 at index 10, each as an 8-bit macro. Standard Rule 90 XOR (14 bits) and Rule 150 nested XOR
(23 bits) are independently checked. Rule 150 is not loaded through its generated test task and this is
not held-out agent performance.

## DD-014 — Exact oracle, configuration, and versioned run lifecycle

`HiddenTaskBundle` strictly validates the oracle artifact, reference semantics/hash, seeds, and
independently verified locked trajectory. `HiddenTaskStore` validates manifest hashes, enforces a
training/development allowlist before opening a hidden file, and records every authorized load.
`ExactDslOracle` canonicalizes/interprets all eight cases, checks the locked rollout through candidate
semantics, computes semantic/AST/residual metadata, and requires zero errors plus rollout agreement for
`exact`. The locked configuration uses score-only feedback.

Configuration schema 2 adds DSL/enumerator bounds and versions while schema 1 remains the strict mock
contract. `wms oracle verify` parses candidate data before hidden access, emits a deterministic result
plus separately named timing, returns 0 exact / 1 valid but failed / 2 invalid, and omits hidden data.

Phase 2 uses run-manifest schema 3, database schema 2, candidate identity/event/results schema 2, and
analysis bundle v1. Candidate rows add source/canonical AST, semantic hash, schema, and exact bit length.
Resume regenerates deterministic enumeration and writes only missing transactional evaluations. Replay
reads recorded candidates, never calls a proposer, checks frozen evaluations, and recomputes oracle,
event, and results data. Reports hash/read frozen candidate/evaluation/event/analysis data and copy the
recorded analysis without importing live enumeration/canonicalization/oracle code. Manifest schema 2
remains an explicit Phase 0 reader and is never upgraded in place.

## DD-015 — Phase 2 public bundles and split protocol

An audit found that 249 of 256 legacy Phase 1 public bundles incidentally observe all eight local
neighborhoods. Their F0 semantics are therefore reconstructable despite having no explicit truth table.
Calling that nonleaking would be false.

Phase 2 adds, rather than overwrites, `phase2-task-bundle-v1` under `artifacts/phase2-benchmark`. It
preserves opaque IDs and split assignment, but public traces use only uniform-zero and uniform-one
states and declare coverage `[000,111]`. Its internal manifest schema is 4. `wms tasks generate`
produces both the legacy Phase 1 evidence and the strengthened Phase 2 bundle. Phase 2 loaders reject
the legacy version. This preserves Phase 1 while making the new proposer context satisfy the stricter
Phase 2 leakage gate.

The recorded smoke uses one training task (the opaque Rule 90 assignment). Development is permitted but
unused. Phase 1 validation remains consumed; Phase 2 validation is unconsumed. Rule 150 and all-256
analyses use public mathematical semantics, not generated test-task oracle artifacts. The application
access ledger records no validation/test oracle load. Active queries remain disabled.

## DD-016 — Phase 3 operators and stateless RNG

Phase 3 freezes four typed local operators: local mutation, typed subtree replacement, simplification,
and typed crossover. Their declared weights are 4:3:2:3. Preorder paths include the root and every
`BitExpr`, `IntExpr`, and `PredExpr` position. Replacement and crossover preserve the selected static
type; generated syntax is bounded by the Phase 2 depth/node/case limits and is passed through strict
candidate JSON, canonicalization, the total interpreter, and prefix coding before it can be evaluated.
Rejected and canonical no-op attempts are explicit outcomes rather than silent retries.

Randomness is `sha256-counter-streams-v1`: the master search seed, attempt counter, and a domain label
produce independent scheduler, parent, operator, path, and replacement-syntax streams. This prevents a
change in one decision's draw count from shifting later decisions. No operator accepts an oracle bundle,
target semantics, hidden seed, semantic hash, error location, or rollout state.

## DD-017 — Public descriptor, archive, reserve, and incumbent

The descriptor is computed only from candidate syntax and a fixed public probe set. Its coordinates are
joint canonical node/code-length bins, one of six representation-family labels, and a public-probe output
cluster. The probe set is the first up to 16 distinct observed local neighborhoods from public traces;
the Phase 2 bundle supplies only `000` and `111`. Descriptor bins, family rules, probe ordering, and
serialization are versioned and target-independent.

`map-elites-lineage-reserve-v1` maintains separate exact and partial layers. Each cell has one elite
under correctness-first rank and lexical candidate-ID tie breaking, plus a two-entry lineage-signature
reserve. Updates are monotone and task scoped. The matched control is one incumbent with the same rank,
tie rule, seeds, initialization, operators, budgets, and continue-after-exact policy. It deliberately
retains shorter exact replacements rather than stopping at the first solution.

## DD-018 — Scheduling, initialization, and exhaustive budgets

The scheduler samples uniformly from sorted eligible branch IDs and records the entire eligible set,
selected index/branch, exact rational probability, remaining budget, and decision hash. Archive branches
are occupied elite/reserve branches; the control exposes only its incumbent branch. Cross-task selection
is rejected.

Both conditions start from the same seven target-independent charged DSL candidates: constants zero/one,
the three local observations, full-mask parity, and full-mask majority. There is no task-specific seed.
Every proposal attempt is charged, including invalid and no-op attempts. Every emitted candidate
evaluation is charged, including canonical/semantic duplicates; there is no oracle-result cache. The
ledger separately reconciles attempts, operator selections, parse/type/canonical stages, duplicates,
oracle calls, scheduler decisions, archive outcomes, and the permanently zero language-model-call count.

## DD-019 — Phase 3 identity, persistence, resume, and replay

Candidate identity hashes the canonical AST, ordered parents, proposer/operator IDs, public-context hash,
payload hash, codec version, and identity schema. The locked experiment used SQLite schema 3, which
stores proposal attempts independently of candidates/evaluations so repeated charged evaluations do not
collapse. New Phase 3 runs use the backward-readable schema 4 diagnostic extension described in DD-023.
Candidate parents must be earlier candidates in the same task/run; ordered lineage edges form an
auditable acyclic DAG. Attempt, candidate, evaluation, archive transition, budget state, event, and
next-step state commit atomically.

Manifest schema 4 freezes every mechanism version, bounds, budget rule, archive/descriptor policy,
authority declaration, and source/lock state. Resume reconstructs the mechanism from committed decisions.
Replay starts from recorded proposal documents and scheduler decisions: it does not call operator
generation or live scheduler selection. It rechecks candidate identity, oracle results, archive/incumbent
transitions, budget states, event hashes, final metrics, and the frozen individual-analysis manifest.
Reporting reads and hashes committed database/artifact records without invoking operators, scheduling,
search, or the oracle.

## DD-020 — Paired locked experiment and inference rule

`experiments/phase3-archive-smoke.yaml` is the complete, strict registry. It freezes 12 opaque validation
task IDs selected by a hash order over public files and split labels only, 20 unique search seeds, both
conditions, a 96-attempt/32-oracle budget, score-only feedback, all mechanism versions, and normalized
exact-solve AUC as the primary endpoint. Pairing is exact by task ID and seed. The zero-tolerance gate is
the paired mean `diverse - incumbent >= 0`; the deterministic paired-bootstrap interval is descriptive,
and superiority is claimed only when its lower 95% bound exceeds zero. A negative point estimate is
reported and stops Phase 3 without tuning.

The freeze is written before any validation oracle call and records repository state, dependency lock,
registry/config/analysis hashes, tasks, seeds, budgets, endpoint, and the prohibition on test access.
Validation is consumed once for this frozen comparison. Test oracle outcomes are never opened. Aggregate
artifacts include raw paired rows, differences and confidence interval, solve curves, archive coverage
over cost, operator outcomes, budget utilization, predeclared lineages, access ledgers, limitations, and
content hashes. CPU/runtime remain non-inferential diagnostics.

## DD-021 — Scope boundary after Phase 3

Phase 3 uses no language model or external API. It adds no learned primitives, surrogate ranking,
learned scheduler, memory retrieval, active queries, ensemble selection, meta-learning, or later-domain
task family. `language_model_calls` is required to remain zero. F0's 256 semantics and the two-public-
probe descriptor make this a mechanics and paired-smoke result, not evidence of broad world-model
discovery or scaling.

## DD-022 — Frozen Phase 3 outcome and post-freeze export amendment

The locked comparison produced a diverse-minus-incumbent normalized exact-AUC mean of `-0.000390625`
over 240 paired task/seed cases. Its deterministic paired-bootstrap 95% interval is
`[-0.002994791666666667, 0.0018229166666666667]`. The point estimate fails the predeclared zero-tolerance
no-worse gate; the interval does not support superiority. Phase 3 therefore stops with a negative result
and no operator, descriptor, archive, task, seed, budget, or analysis tuning.

All 480 children completed before the first aggregate export. That export then rejected the JSON-only
`transition_outcomes` field while projecting rows to CSV. The correction only tells `csv.DictWriter` to
ignore fields outside the declared CSV columns and adds an amendment hash to provenance. Aggregation was
rerun from the 480 immutable results, without reexecuting a child or opening any oracle artifact. The
original and corrected analysis-source hashes, repository state, unchanged-contract assertions, and
reason are recorded in `analysis-amendment.json`; its hash is included in the final analysis manifest and
summary. This is an artifact-serialization correction, not an outcome-driven experimental change.

A second evidence-only amendment adds requirements that were absent from the first aggregate bundle:
explicit results for all eight gates, paired solve-rate-difference intervals at every charged cost, and
per-child hashes for manifests, results, event lists, proposal artifacts, timing-free database records,
and individual-analysis manifests. It changes neither the primary calculation nor its result and reads
only already completed records. The supplement has its own content-hash manifest linked to the immutable
primary manifest and evidence-amendment hash.

## DD-023 — Post-result Phase 3 implementation and measurement hardening

After freezing the negative result, Phase 3 was hardened without reopening validation or changing its
scientific outcome. The concrete mutation proposer, MAP-Elites archive, single-incumbent control, and
uniform scheduler now implement the shared generic capability protocols used by the runner. Local edit
is total over every allowed AST constructor, including `TruthTable`, whose mutation flips exactly one
deterministically selected output bit. Invalid and canonical no-op proposals remain charged attempts and
cap exhaustion is tested directly.

SQLite schema 4 adds one diagnostic row per new proposal attempt: process CPU, monotonic elapsed time,
oracle CPU/elapsed contributions, and language-model call/token counts. These host-dependent values are
exported as `runtime-diagnostics.json`, copied and hashed by reports, and deliberately excluded from the
deterministic result and replay hashes. Schema-3 locked runs remain readable. Language-model counts are
zero because Phase 3 has no LLM proposer.

Future experiment analyses report both the frozen task-seed-pair bootstrap and a task-clustered bootstrap
that resamples whole task IDs as a dependence sensitivity check. The clustered interval is not used to
reinterpret the already consumed validation result. Regression evidence now includes exact joint-bin
edges, elite/reserve/tie transitions against an independent reference model, forced invalid/no-op cap
exhaustion, captured proposer-context leakage scans, and two independent executions of the complete
480-child aggregate on development tasks. No locked child or validation aggregate was rerun or tuned.

## DD-024 — Phase 4 provider boundary and pinned OpenAI contract

The domain-facing interface is provider-neutral and immutable. `LLMProposer` owns public prompts,
roles, batch validation, cache identity, canonical proposal conversion, and stateless semantics.
Transport adapters own authentication, dispatch, error normalization, usage, and diagnostics. SDK types
never enter search, persistence, replay, or analysis records.

The live adapter is deliberately narrow: official `openai==2.53.0`, `v1/responses`, exact snapshot
`gpt-5-mini-2025-08-07`, default service tier, low reasoning, strict JSON schema, `store=false`, and
disabled truncation. On August 11, 2026, the official model page continued to list the snapshot and
rates but labeled the snapshot deprecated. The two authorized training canaries established project
access to the exact snapshot and accepted request fields, with no silent alias/substitution. The first
512-token canary failed by output-cap exhaustion; the versioned 2,048-token correction returned a strict
complete batch. The key is read only after both live opt-ins and never enters config, CLI text, hashes,
logs, artifacts, reports, or sanitized exceptions.

## DD-025 — Phase 4 prompts, feedback, batches, and lineage

Direct prompt v1 contains only the public task/demonstrations, public DSL grammar/bounds, role, exact
batch count, schema contract, and independence declaration. Iterative prompt v1 adds exactly one typed
parent AST and its associated score-only record: ID, validity/totality, local errors/cases, exactness,
AST bits, residual bits, and two-part bits. Runtime, error locations, counterexamples, rollout state,
semantic hashes, internal family, paths, secrets, and history are structurally absent.

One synchronous response contains several complete candidate documents. Envelope/item order is frozen.
Root failures may consume one declared identical retry; item failures do not trigger repair. Valid items
become `llm-direct-v1` or `llm-revision-v1` lineage and are evaluated directly. Canonicalization is
allowed; Phase 3 mutation is not.

## DD-026 — Cache, paid request state, and cumulative ledger

Exact cache identity includes transport/model/endpoint/tier, exact rendered input, prompt/schema/role/
batch/settings, and excludes secrets, time, paths, run/request IDs, and provider IDs. Entries carry a
content hash and exact request key. Locked runs use a new experiment/run-scoped namespace.

Request preparation and worst-case reservation commit before external dispatch. Immutable response or
sanitized failure evidence is hashed before usage reconciliation. Items then commit in ordinal order
with candidate, evaluation, transition, event, and budget. A durable responded batch resumes offline;
a dispatched request without durable response becomes usage-uncertain instead of being duplicated.
Reconciliation is idempotent only when the complete recorded usage/failure hash matches. Pending,
post-dispatch, post-response, post-finalization, retry, and item boundaries are exercised explicitly.
An unaffordable next reservation produces a replayable `cost-cap-exhausted` child before dispatch.

The SQLite project ledger uses WAL and `BEGIN IMMEDIATE`, integer nano-USD, append-only usage records,
and request/child/stage/Phase-4/project checks. `local_state/` is outside ordinary artifact cleanup. The
OpenAI dashboard remains the external backstop if all local evidence is lost; local cost is a
published-rate estimate rather than an invoice. Promotional or data-sharing token credits may reduce the
provider invoice but do not reduce reservations, ledger estimates, or any scientific hard ceiling.

## DD-027 — Matched A/B/C mechanisms and budgets

A pays for seven shared initial candidates but never exposes them; each call is independent direct
sampling. B selects the current incumbent and exposes its AST/score. C samples one occupied archive
branch uniformly and uses that branch's primary parent; returned items enter independently computed
cells in response order. B/C share the iterative prompt and exploit-only role. Existing incumbent,
archive, rank, descriptor, reserve, and uniform scheduler behavior is reused without retuning Phase 3.

All conditions share snapshot/endpoint/tier/settings, role facts, schema, batch, requests/tokens/items/
oracle caps, initialization, score mode, continue-after-exact rule, and child ceiling. Intended
differences are parent/feedback exposure and incumbent versus archive retention/branch selection.
Invalids and duplicates are measured behavior, not grounds for unbounded refill.

## DD-028 — Price/retry policy and live sequence

`phase4-price-and-ceilings-v1` is the sole money authority. Rates are 250 nano-USD/token uncached input,
25 cached input, and 2,000 output; reasoning is already in output. Ceiling increases require explicit
user editing and may never exceed $30 Phase 4 or $100 project. Pilot v2 uses the policy's $0.15 maximum
per development child and 60 children, so summed child ceilings remain $9.00 under the $9.75 stage cap.

Retry v1 permits at most one identical retry for a rate-limit response with provider-established zero
usage or a malformed envelope whose usage was recorded. Authentication, permission, invalid-request,
timeout, connection, and server failures are not retried. Ambiguous usage is terminal and retains the
full reservation. Paid work order is fake, training canary plus offline replay, development pilot plus
power/cost review, complete freeze, then one locked test. The revised training canary passed; this
handoff still stops before the development pilot.

## DD-029 — Experiment inference, authority, and record-only artifact

The pilot fixes ten opaque development IDs, two seeds, A/B/C, rotating condition blocks, 256 charged
evaluations/child, all caps, H1=B-A, H2=C-B, descriptive C-A, normalized exact AUC, exact task/seed
pairing, task-clustered primary bootstrap, task-seed sensitivity, 95% intervals, and Holm multiplicity.
The reduction from four to two within-task seeds is a cost-feasibility correction: raising the output
allowance and child ceiling while retaining 120 children would make summed child exposure exceed the
$9.75 development cap. Ten independent task clusters are retained because tasks, not repeated seeds,
are the primary bootstrap unit. Dry-run reads no hidden artifact/key and performs no write/provider call;
pilot v2 also verifies the exact successful canary configuration and deterministic result hashes before
declaring itself ready. If the benchmark process is interrupted, an existing hash-matched child without
terminal results is resumed through the paid-request state machine rather than rejected or restarted.

The test file is intentionally pending and non-executable. Test IDs/sample/seeds/budgets/cache cannot be
frozen until pilot variance, power, validity, token, and cost evidence fits $20 locked and $30 Phase 4.
`Phase4Authority.locked_test` is narrow and hash-bound for that future one-time run. Consumed Phase 3
validation is not reused as confirmation; test outcomes remain unopened.

Individual and aggregate artifacts are built from committed records only. They preserve negative/null/
failure outcomes, cost/token curves, MDL/best-program sources, archive coverage separate from
correctness, proposal/cache/retry diagnostics, lineage, runtime, access, contract hashes, raw pairs,
and H1/H2 intervals. Fake evidence is labeled machinery-only.

The two primary clustered bootstrap p-values are computed from centered task-cluster resamples. Holm's
two-sided two-hypothesis procedure records adjusted p-values and sequential rejection decisions;
`superiority_established` additionally requires a positive point estimate. Sensitivity intervals and
the descriptive C-A comparison do not replace that primary decision rule.

## DD-030 — Phase 4 scope and F0-only deviation

The wider F1/F2 recommendation would require new task, oracle, DSL, and authority work. Phase 4 remains
explicitly F0-only rather than smuggling new families into LLM integration. A target-blind deterministic
random DSL proposer fills the missing contextual baseline while the enumerator and Phase 3 evidence stay
intact.

No cross-task memory/retrieval, language promotion, learned primitive, interestingness/UCB scheduler,
active query, hidden-state model, scaffold modification, or Phase 5-8 mechanism is implemented. The
256-semantics F0 universe and small grammar cannot establish broad discovery, transfer, or general LLM
superiority.
