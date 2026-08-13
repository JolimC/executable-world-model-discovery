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

At initial Phase 5 preparation, the supported ledger reported the carried Phase 4 published-rate amount `$6.52680755`, reconciled cash
`$4.65` with `user-reported-unverified` verification, zero uncertain usage, zero active reservation,
and `$95.35` current authorizable headroom under the `$100` personal ceiling.

The then-pending Phase 5 exposure partition forecast at most 256 requests. Each reserves 12,000 input-bound
tokens plus 2,048 output tokens and `$0.007096`; sixteen requests per child reserve `$0.113536`, and the
complete development forecast is `$1.816576` published-rate equivalent with a 7,200-second runtime
bound. Fully unreconciled, the cash upper bound would be `$6.466576`, leaving `$93.533424`. This fit does
did not authorize spending. The user later supplied the required development authorization. The
separate `phase5-test.sealed.yaml` continues to deny test model and oracle authority.

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
model and service tier match. At that checkpoint, the development authority was still pending and no
development artifact had been created.

### Live C/D development outcome and final freeze (2026-08-12)

Before development authorization, the user reported the OpenAI project “world_model_search” all-time
dashboard total as `$4.65` at 2026-08-12 01:10 CDT, under “amount billed after credits.” The append-only
checkpoint remains `user-reported-unverified` and covers the finalized canary reservation. It restores
the reconciled cash upper bound to `$4.65` before the development run without changing the canary's
published-rate record.

The exact frozen pilot completed 16 children and all 256 requests: 128 requests per condition. It
recorded 255 schema-valid candidates, one invalid condition-C candidate, 255 oracle calls, zero cache
hits, and zero sealed-test accesses. Published-rate usage was `$0.13670200`: `$0.07349640` for C and
`$0.06320560` for D. The post-run cash upper bound is `$4.78670200`; the actual provider-dashboard total
remains the earlier `$4.65` checkpoint until another observation is supplied. There is zero uncertainty,
zero active reservation, and `$95.21329800` remaining authorizable headroom.

Neither arm solved any of its eight children exactly. Condition C had mean best score 5.5; D had 5.0.
The paired D-C score estimate is -0.5 with family-stratified task-cluster 95% interval [-2, 1], two-sided
`p=0.49595`, and no Holm rejection. D improved one task by four points on both seeds, but lost two points
on each of the other three tasks on both seeds. D retrieval precision was 1.0. D used 38,656 more input
tokens and 4,491 fewer output tokens, for 34,165 more total tokens, while its shorter outputs made its
published-rate total `$0.01029080` lower than C. These efficiency differences do not rescue the failed
correctness gate.

Provider-disabled replay returned `verified-provider-disabled` for all 16 children and 256 artifact
chains. The final analysis marks the primary endpoint
`failed-correctness-no-comparable-exact-pairs`: the mechanically displayed -50-bit definition charge is
not interpretable as held-out transfer gain when neither arm has an exact program. H3 is unconfirmed and
the development pilot does not support it.

The final freeze intentionally performs no refit. Memory content and primitive definitions are
unchanged; only registry metadata is rebound to the final analysis plan. The freeze manifest hash is
`5cd248d6bb49ae5b83af33688c9f65d30583e0a5bb4b0e887b35c80f1fed706c`. The final memory hash is
`9ab7e6ced0cdaa7a2764edd9638b1e5d2af781d511343ecf05a420408fd7bad4`, primitive registry hash is
`14b2610ceac37482f6daead896f89f4d634adf03166a54edf857ec08d299bbd8`, and analysis-plan hash is
`3c0d05942746ed4ccdb63f2410e1d1eb4ea880c4714a1a7891356caed18e404a`. The sealed declaration binds
those identities but still denies model and test-oracle authority.

### Experience-memory v2 retrospective preparation (2026-08-12)

The revised memory mechanism now reuses the frozen Phase 4 `uniform-diverse-archive-v1` arm instead of
rerunning training search. A provider-disabled extractor audited all 20 condition-C runs over ten tasks,
selected the first exact-producing request in each of nine solving runs, and paired each successful
lineage with the three non-exact siblings from the same request and selected parent. The resulting nine
contrast records contain 27 matched unsuccessful lineages over five exact-solving tasks. Source tasks
are explicitly redesignated from their original Phase 4 development role to retrospective v2 training;
all come from one generator family, so no cross-generator-family claim is made.

Assignment follows the representation family of the consequential selected parent cell. Three
induction groups have adequate multi-task support: `mixed` (three tasks), `position-specific` (three),
and `threshold` (two). The `conditional` group has one task and is retained only for audit. Exactly one
strict structured-output request requesting one lesson is frozen for each eligible family. Corpus hash
is `cca04ce74820fdfda46b33eaeafd1cbecdec006163b3e65d76192536459758b8`; preparation manifest hash is
`dfe033327ce82ee31ad389aba7f3b2675d3e4c197f23f86f1814c8ecc44c187d`. The three-request conservative
published-rate ceiling is `$0.021299`. All packages deny provider dispatch, and zero induction requests
were executed.

Promotion is now a two-stage prospective design: sole-lesson arms versus a shared matched control,
followed by fresh-task confirmation of the promoted bundle. Exposure minima produce an inconclusive
status when too few matching cells are selected. Only a passing bundle can freeze a development
snapshot. The prospective task registry, adaptive task/cost cap, induction/validation authorization,
and development authorization remain pending.

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

Formatting and lint passed, strict mypy passed over 78 source files, and all 139 tests passed. Dry-run
proved 12/12 semantic uniqueness and forecast fit. Smoke completed with zero provider
calls and zero sealed-test accesses. Replay completed as `verified-provider-disabled`.

## Gate disposition and limitations

| Gate | Status | Evidence |
|---|---|---|
| Preservation/backward compatibility | Passed | Complete 139-test suite; Phase 4 ledgers/schemas untouched |
| Family protocol and leakage | Passed for smoke | Whole structural families, 12 unique semantics, public-boundary tests, zero sealed accesses |
| Typed-memory integrity | Passed | Version/provenance/content/scope/bound/counter/corruption tests and deterministic export |
| Primitive promotion/MDL | Passed on development gate | All correct; 96 gross minus 50 definition = +46 net bits |
| Matched isolation | Passed for no-cost smoke | Exact C/D manifest and prompt differencing |
| Scope and contradiction | Passed | Independent-count, counterevidence, invalidation/rejection/empty-library tests |
| Replay and report | Passed | Provider-disabled no-cost, canary, and 256-request development verification |
| Budget authority | Passed for authorized canary/development | All usage finalized; sealed test still denies authority |
| Scientific status | Failed to support H3 on development | Zero exact solves in both arms; D-C mean best-score difference -0.5; H3 unconfirmed |

The strongest limitation is that the family system is deliberately constructed from reference-program
grammar within the eight-case F0 universe. It demonstrates controlled structural transfer, not broad
world-model learning. The positive primitive result is development evidence selected by its promotion
gate. The paired recoding smoke reuses those families, has no model calls, and has no family-stratified
uncertainty estimate. A scientific C/D claim requires the separately reviewed live protocol followed by
a predeclared, one-time sealed test. Null or negative future outcomes must leave H3 unconfirmed without
changing families, endpoints, gates, or budgets.
