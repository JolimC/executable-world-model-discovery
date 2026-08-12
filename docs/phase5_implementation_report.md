# Phase 5 implementation report

## Outcome

Phase 5 engineering is complete within the authorized F0 scope. The implementation adds a defensible
structural-family prerequisite, a separate typed SQLite memory, deterministic proposer-safe retrieval,
an evidence-gated learned-primitive language, exact condition C/D isolation, no-cost execution,
provider-disabled replay, budget forecasting, and a transfer-matrix report. Phase 6 was not started.

Scientific status is narrower. One primitive passed the predeclared development promotion gate, but the
paired C/D run is a deterministic recoding smoke over the same development families and is not an
independent H3 experiment. The later one-request live training canary passed compatibility checks but is
not scientific evidence. There were no sealed-test accesses. H3 remains unconfirmed.

## Structural transfer prerequisite

`phase5-f0-structural-transfer-v1` defines six reproducible reference-program grammar strata before
evaluation: two source-training selectors, two development target compositions, and two sealed-test
compositions. Two variants per family produce twelve tasks. Exhaustive eight-case hashes prove twelve
unique semantics and no duplicate across roles. The sealed tasks were generated and indexed, but their
evaluator artifacts and outcomes were never accessed.

The family is a structural generative distinction, not an ECA rule number, task identifier, seed,
descriptor, archive bin, or result. Public tasks contain only ordinary F0 mechanics, uniform-state
demonstrations, opaque task ID, public split, and disabled-query status. Generator family, reference AST,
semantic identity, exact ECA realization, and split proof remain evaluator-only. Consumed Phase 2,
Phase 3, and Phase 4 evidence is barred from memory admission.

## Typed memory and retrieval

The schema-1 memory database uses `WAL`, `synchronous=FULL`, transactions, foreign keys, SQLite
integrity checks, explicit version refusal, and deterministic export. It stores typed episodic,
hypothesis, search-lesson, primitive-proposal, and self-model categories; immutable evidence; scopes;
applicability; definition cost; provenance; support/validation/counter links; and append-only proposal,
promotion, rejection, scoping, and invalidation events. Content identity is recomputed during audit.

Evidence must be present in the frozen eligible catalog. Training support, development validation, and
test refusal are role checked. Independent counts use distinct task IDs and family IDs. Duplicate
observations and retries cannot create independent support. Counterevidence prevents promotion and
invalidates already promoted memory.

Snapshots structurally omit evaluator metadata. Retrieval is stable, scope-aware, and bounded by item,
UTF-8 byte, and conservative token limits. Its artifact records the public query identity, eligible set,
integer scores, tie order, selected IDs, exclusions, rendered bytes, and snapshot hash.

## Learned primitive language

Definitions are typed, canonical, bounded base-DSL `BitExpr` values with zero parameters. A learned call
contains only the definition's content hash. Expansion is hygienic and occurs before canonicalization
and exact evaluation. `TruthTable`, arbitrary Python, evaluator handles, unknown definitions, and
semantic copies of built-in `Parity`/`Majority` are rejected.

The library is a counted sequence of length-delimited base prefix codes. Calls use a reserved prefix and
an Elias-gamma registry index. Library and program decoders reject truncation, extension,
noncanonical order, unknown indices, and streams that do not re-encode identically. The complete library
definition code is 50 bits in the smoke.

## No-cost results and presentable artifact

Training evidence came from four tasks across two source families. The frozen selector primitive was
then evaluated once on four development tasks across two wholly held-out families, without refitting:

| Quantity | Bits |
|---|---:|
| Base programs | 192 |
| Learned invocation programs | 96 |
| Gross savings | 96 |
| Complete library definition | 50 |
| Net held-out gain | **46** |

All four programs expanded exactly and passed the exact oracle, so the development promotion gate
passed. The subsequent two-seed C/D machinery smoke produced eight pairs. C and D were both exact per
evaluation; D retrieved the promoted record on every row. Its aggregate was 384 versus 192 program bits,
192 gross saved, 50 definition bits shown separately and charged once, for +142 net bits. The memory
prompt added 6,024 UTF-8 bytes across the eight D requests. Model tokens, published cost, and actual cash
were all zero.

The transfer matrix is
`artifacts/phase5/no-cost-smoke-v1/transfer-matrix.json`. Each of the two target families records 96
gross bits saved, zero definition cost allocated to the cell, and no negative transfer. Aggregate
definition and net values are separate. Raw pairs, retrieval records, bound request identities,
condition manifests, development gate, memory export/snapshot, primitive registry, analysis, forecast,
and report are siblings under `artifacts/phase5/no-cost-smoke-v1/`.

The provider-disabled replay verified sixteen condition rows, reported +142 net smoke gain, and
reproduced report hash
`520d937e70d4220bb161e5df98486c3a5c793ddbdd26aaa7b90d2df08696463e`
and summary hash
`31d9f724227475fd244bc291248407fc93a615daaa68f9a71e2acca702f53c9e`.

## Budget and authority

The supported ledger reports the carried Phase 4 published-rate amount `$6.52680755`, reconciled cash
`$4.65` with `user-reported-unverified` verification, zero uncertain usage, zero active reservation,
and `$95.35` current authorizable headroom under the `$100` personal ceiling.

The pending Phase 5 exposure partition forecasts at most 256 requests. Each reserves 12,000 input-bound
tokens plus 2,048 output tokens and `$0.007096`; sixteen requests per child reserve `$0.113536`, and the
complete development forecast is `$1.816576` published-rate equivalent with a 7,200-second runtime
bound. Fully unreconciled, the cash upper bound would be `$6.466576`, leaving `$93.533424`. This fit does
not authorize spending. `phase5-development.pending.yaml` requires user review of the new exposure
partition and explicit authorization of the live run. `phase5-test.sealed.yaml` separately denies test
model and oracle authority.

### Live-stage preparation addendum (2026-08-12)

The next-stage preparation is now complete without provider calls. The supported ledger audit is
unchanged: `$4.65` reconciled cash marked `user-reported-unverified`, `$0` uncertainty, `$0` active
reservations, and `$95.35` authorizable headroom. Exposure policy v2 explicitly allocates `$0.01` to a
training compatibility canary, `$5` to development, and `$9.99` to the still-sealed test, preserving the
`$15` Phase 5 total and the parent v2 cash policy.

`phase5-canary.yaml` is a one-task, one-seed, one-request condition-D compatibility gate with a
`$0.007096` worst-case forecast. `phase5-development.yaml` freezes four development tasks, two seeds,
matched C/D rotating order, sixteen requests per child, sixteen children, and the existing `$1.816576`
forecast. Both bind tracked copies of the promoted memory snapshot and primitive registry, exact prompt
and schema, `gpt-5-mini-2025-08-07`, Responses API, default tier, low reasoning, 2,048 output tokens,
budgets, cache, ledger, and analysis plan. OpenAI's current model documentation still lists that exact
snapshot and labels it deprecated; no substitution was made.

At preparation time, both pending authority declarations denied model calls, oracle access,
exposure-policy review, and live-run authority. Provider-disabled preflights passed; the largest
constructed identities were 8,580 and 8,583 bytes under the declared 12,000 conservative bound. A
live-command refusal test reached no credentials, backend, or oracle. The development runner also
requires the exact canary registry and a recorded `passed-live-training-canary` summary before dispatch.
At that point no live or sealed-test outcome had been accessed. Sequential canary plus development
worst-case exposure was `$1.823672`, leaving `$93.526328` headroom if both were fully unreconciled after
the current cash checkpoint.

### Live canary outcome (2026-08-12)

The user authorized only the training canary. Its frozen experiment hash was
`09d543e54bf61149f7fb167c259ed6b31a9ac1508756c3a3c3849fa8270546b6`. The runner made exactly one
physical provider request to the requested `gpt-5-mini-2025-08-07` snapshot on the default tier. The
response used 1,587 input tokens and 212 output tokens, including 64 reasoning tokens, and produced one
schema-valid candidate with zero invalid candidates. The canary ended as
`passed-live-training-canary`; this is a compatibility result, not H3 evidence.

The ledger recorded `$0.00082075` at published rates. That amount is unreconciled usage, not a new
provider-dashboard cash observation; the latest actual-cash checkpoint remains the user-reported,
unverified `$4.65`. The resulting cash upper bound is `$4.65082075`, with `$95.34917925` authorizable
headroom, zero uncertainty, and zero active reservations.

With provider credentials and the live gate removed from the replay environment, replay returned
`verified-provider-disabled`, zero provider calls, zero oracle accesses, and zero sealed-test accesses.
Request hashes match across the request, response, and result artifacts, and requested versus returned
model and service tier match. The development authority remains pending, and no development artifact
was created.

## Verification

The final no-cost commands were:

```console
unset OPENAI_API_KEY WMS_ALLOW_LIVE_MODEL
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest -q
.venv/bin/python -m world_model_search.cli phase5 dry-run
.venv/bin/python -m world_model_search.cli phase5 smoke
.venv/bin/python -m world_model_search.cli phase5 replay
.venv/bin/python -m world_model_search.cli phase5 live-dry-run \
  --experiment experiments/phase5-canary.yaml \
  --authority experiments/phase5-canary.pending.yaml
.venv/bin/python -m world_model_search.cli phase5 live-dry-run \
  --experiment experiments/phase5-development.yaml \
  --authority experiments/phase5-development.pending.yaml
```

Formatting and lint passed, strict mypy passed over 74 source files, and all 130 tests passed. Dry-run
proved 12/12 semantic uniqueness and forecast fit. Smoke completed with zero provider
calls and zero sealed-test accesses. Replay completed as `verified-provider-disabled`.

## Gate disposition and limitations

| Gate | Status | Evidence |
|---|---|---|
| Preservation/backward compatibility | Passed | Complete 130-test suite; Phase 4 ledgers/schemas untouched |
| Family protocol and leakage | Passed for smoke | Whole structural families, 12 unique semantics, public-boundary tests, zero sealed accesses |
| Typed-memory integrity | Passed | Version/provenance/content/scope/bound/counter/corruption tests and deterministic export |
| Primitive promotion/MDL | Passed on development gate | All correct; 96 gross minus 50 definition = +46 net bits |
| Matched isolation | Passed for no-cost smoke | Exact C/D manifest and prompt differencing |
| Scope and contradiction | Passed | Independent-count, counterevidence, invalidation/rejection/empty-library tests |
| Replay and report | Passed | Provider-disabled no-cost reproduction and live-canary bundle verification |
| Budget authority | Passed for authorized canary only | One canary call recorded; development and sealed-test declarations still deny authority |
| Scientific status | Passed | F0-only labels; paired smoke not H3 evidence; H3 unconfirmed |

The strongest limitation is that the family system is deliberately constructed from reference-program
grammar within the eight-case F0 universe. It demonstrates controlled structural transfer, not broad
world-model learning. The positive primitive result is development evidence selected by its promotion
gate. The paired recoding smoke reuses those families, has no model calls, and has no family-stratified
uncertainty estimate. A scientific C/D claim requires the separately reviewed live protocol followed by
a predeclared, one-time sealed test. Null or negative future outcomes must leave H3 unconfirmed without
changing families, endpoints, gates, or budgets.
