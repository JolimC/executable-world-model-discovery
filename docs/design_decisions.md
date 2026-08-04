# Design decisions

## DD-001 — Standard-library CLI and strict YAML loader

The CLI uses `argparse`; YAML is the only runtime dependency. Configuration is decoded manually into
frozen dataclasses with exact key sets. This keeps the shell small and makes the validation-before-write
boundary auditable. Repository-relative run roots are required to prevent configuration from directing
artifacts outside the repository.

## DD-002 — Public task capability type

`Task` and `PublicTask` are separate frozen types. `Task.public_view()` is the sole conversion used to
build `ProposalContext`. The public type cannot represent task seeds, hidden artifact IDs, exact cases,
or locked rollout suites. This is stronger than depending on a serializer deny-list.

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
