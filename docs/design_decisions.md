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
