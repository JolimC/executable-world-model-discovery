# Phase 0 architecture summary

Phase 0 is a deterministic, mock-only execution shell. Its purpose is to prove the project contract
before any cellular-automaton, synthesis, archive, scheduler, memory, or language-model mechanism is
introduced.

## Trust boundary and typed flow

An internal frozen `Task` contains public demonstrations plus oracle-only identifiers and the task seed.
The proposer can receive only a frozen `ProposalContext`, whose `PublicTask` structurally omits the
seed, hidden artifact identifier, exact case-set identifier, rollout suite identifier, and public artifact
content hash. The context has no filesystem or database handle. Phase 0's `MockProposer` receives a
domain-separated proposer RNG seed through `ProposalBudget`; that seed controls proposal generation,
differs from the internal task seed, and is not oracle data. There are no model calls.

The mock proposer emits an opaque typed `RuleExpr` fixture. This is deliberately not the real DSL. The
runner attaches deterministic candidate identity and lineage fields, stores the canonical proposer output
as an immutable JSON artifact, and passes the typed candidate to `MockOracle`. The mock evaluator has no
CA behavior; it returns the `OracleResult` shape required by the future exact oracle and keeps measured
runtime as diagnostic data.

```text
validated YAML -> immutable manifest -> PublicTask -> MockProposer
                                             |              |
                                      context hash     proposal artifact
                                                            |
SQLite state <- deterministic event <- MockOracle <- typed candidate
     |                    |
  resume             payload hash
     |                    |
 results.json <- recomputed metrics <- replay from proposal artifacts
```

## Persistence and lifecycle

Strict YAML validation rejects missing/unknown keys, invalid types, unsupported Phase 0 implementations,
unsafe run roots, and invalid budgets before the run root is created. A valid start writes an immutable
canonical `manifest.json`, then initializes `run.sqlite3` in WAL mode. SQLite owns mutable lifecycle
state, internal task metadata, candidates, evaluations, and the append-only event ledger. Each candidate,
evaluation, event, and next-step update is one transaction. Proposal artifacts are written first using
exclusive/content-checked creation; an interrupted transaction can therefore leave only an idempotent
orphan proposal file, never a partially committed event.

Interruption is supported both through `KeyboardInterrupt` and the deterministic `--interrupt-after`
test control. Resume reads the frozen configuration from the manifest, checks the configured run root,
and continues from `next_step`. Completion freezes `results.json`. Reporting reads only the manifest,
ledger, and results artifact.

## Determinism and replay

Canonical JSON v1 uses UTF-8, sorted object keys, compact separators, finite numbers, and SHA-256. An
event payload hash includes schema version, logical step, task ID, deterministic candidate identity,
proposal artifact hash, parent IDs, proposer/operator IDs, public context hash, and all correctness and
feedback fields from the oracle result. It excludes run ID, timestamps, absolute paths, runtime and
memory measurements, operating-system/hardware data, process IDs, random UUIDs, and Git metadata.

Replay opens SQLite read-only, validates every stored payload hash and sequence, loads each immutable
proposal artifact, reconstructs and validates the candidate identity, re-evaluates it with the recorded
mock oracle version, and compares every deterministic payload byte-for-byte. It then recomputes the
frozen result summary. Replay contains no proposer instance or fallback generation path; its report
records `proposer_invocations: 0`.

Split labels (`training`, `development`, `validation`, and `test`) are immutable metadata in both
internal and public task types. Phase 0 does not generate splits, deduplicate semantics, enforce an
experimental split policy, or run a split-based experiment.
