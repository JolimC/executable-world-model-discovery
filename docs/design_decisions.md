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
