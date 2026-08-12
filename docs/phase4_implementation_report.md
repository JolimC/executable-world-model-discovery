# Phase 4 implementation report

## Outcome

**Engineering complete; development pilot complete; confirmatory locked test waived; H1/H2 remain
unconfirmed.** The real A/B/C engine, scripted transport, target-blind
random baseline, bounded request/cache/retry/budget lifecycle, offline replay, paired analysis, dry-run,
and frozen reporting execute without an API key or network connection.

The live compatibility gate passed on the authorized revised training canary. The exact
`gpt-5-mini-2025-08-07` snapshot accepted the original two Responses calls, which exhausted their
512-output-token allowance, and the corrected 2,048-token call, which returned the exact two-item strict
batch. The development pilot subsequently completed all 60 children and its provider-disabled replay,
artifact integrity, request/token/oracle accounting, cost reconciliation, variance analysis, and power
forecast passed. The user then deliberately waived the planned locked confirmatory test. No
model/provider/tier substitution was made, no locked registry was frozen, and no test outcome was
accessed.

## Contract map

| Concern | Frozen contract |
|---|---|
| Package | 0.4.0, Python 3.12+, `openai==2.53.0`, `PyYAML==6.0.2` |
| Model | `gpt-5-mini-2025-08-07`, `v1/responses`, default tier, low reasoning |
| Output | strict batch schema v1; exact role and item count; complete typed AST documents |
| Prompts | direct public-task v1; iterative parent-score v1; exploit-only role schedule |
| Feedback | parent-associated score-only v1; no runtime, rollout, errors, semantics, or hidden data |
| Cache/retry | exact request cache v1; one identical bounded retry |
| Search | A direct, B incumbent revision, C uniform archive revision; seven charged initial candidates |
| Persistence | config 4, manifest/database 5, event/results 4, candidate/request/budget v1 |
| Money | integer nano-USD; $100 project, $30 Phase 4, $0.25/$9.75/$20 stage partitions |
| Inference | H1 B-A, H2 C-B; task-cluster bootstrap plus task-seed sensitivity; enforced Holm decisions |

The official rate table reverified on August 11, 2026 is $0.25/MTok uncached input,
$0.025/MTok cached input, and $2/MTok output. Reasoning tokens are included in output. The local amount
is a conservative published-rate estimate, while a separately configured dashboard budget is the
external billing backstop. The recorded price-policy hash is
`120ca1d0cb66d23230ff8267d4c0eb492421e8de55dc4e1e97950e5cd7fc93fa`.

The `PHASE4-LIVE-CANARY` run records one logical call, one identical bounded retry, two physical HTTP
200 responses, 2,800 input tokens, 1,280 cached-input tokens, 1,024 output tokens including 832 reasoning
tokens, and zero valid proposal items. The first response was truncated JSON; the second consumed its
entire output allowance as reasoning and returned no candidate text. Both attempts are charged schema
failures. Their conservative cost estimates are $0.001374 and $0.001086, totaling $0.00246 with zero
uncertain charge.

The cumulative ledger reconciles $0.005676 reserved as $0.00246 actual plus $0.003216 released, with no
active reservation. Replay with provider access disabled made zero proposer/provider calls and reproduced
deterministic summary hash `76a8591e515991a3ff52afb574790885d39bf7dd4d66efbcab0719f6cd2e855c`.
The frozen report is `artifacts/reports/PHASE4-LIVE-CANARY/summary.json`, canonical SHA-256
`b14778d30b1151f1a0a99c8f923fc7b6cccc1034bb57cf58849f8bfd929df4b3`.

The authorized versioned correction changes the canary `max_output_tokens` from 512 to 2,048, its
retry-aware output cap from 2,048 to 4,096, total-token cap from 52,048 to 54,096, and cache namespace to
`phase4-canary-output-2048-v2`. `PHASE4-LIVE-CANARY-V2` made one HTTP 200 call with 1,400 input tokens
(1,280 cached) and 524 output tokens, including 384 reasoning tokens. It returned exactly two schema-valid
ASTs; both were evaluated, with one recorded canonical/semantic duplicate. No retry or uncertain charge
occurred.

The corrected request reserved $0.00591025 and reconciled to a conservative $0.00111 estimate, releasing
$0.00480025. Across both canaries the ledger reconciles $0.01158625 reserved as $0.00357 actual plus
$0.00801625 released, with zero uncertain charge and no active reservation. Promotional credits are not
used to lower these estimates or any hard ceiling. Provider-disabled replay made zero proposer/provider
calls and reproduced deterministic summary hash
`23bba3d953eec518c129e5a226ad3aa82167ac40638b69023720eb90dbf6d6cc`. The corrected report is
`artifacts/reports/PHASE4-LIVE-CANARY-V2/summary.json`, canonical SHA-256
`5be29b06b2434b771a0301c92fbc66c399b71655403848bff5ac9464a461850f`.

## Development pilot and scientific closeout

`phase4-primary-pilot-v2` completed 60/60 children over ten development tasks, two seeds, and three
conditions. Each child used 256 charged evaluations. The recorded run contains 3,780 logical calls,
3,797 physical attempts, 17 bounded schema-failure retries, 14,940 valid and zero invalid proposal
items, and 15,360 exact-oracle invocations. Provider-disabled replay passed for all 60 children with
zero proposer/provider calls. Independent artifact/hash audit errors were zero.

The normalized exact-solve AUC means were A/direct `0.2015625`, B/single-incumbent `0.1927734375`, and
C/uniform-diverse-archive `0.2892578125`, each over 20 paired task/seed rows. H1 (B-A) was not supported:
estimate `-0.0087890625`, task-clustered 95% interval `[-0.208203125, 0.1470703125]`, Holm-adjusted
`p=0.9293070693`. H2 (C-B) was positive but inconclusive: estimate `0.096484375`, task-clustered 95%
interval `[-0.093359375, 0.326171875]`, Holm-adjusted `p=0.7347265273`. H2 is not established.

The pilot's published-rate equivalent is `$6.52323755`; cumulative Phase 4 published-rate equivalent,
including both canaries, is **$6.5268** (exactly `$6.52680755`). Uncertain usage is zero and no active
reservation exists. The user separately reported **$4.65** on the provider dashboard. This is an
**unverified dashboard reconciliation**, because no provider export or invoice was supplied; it does
not replace or reduce the conservative local ledger.

The original protocol called for a later locked test. After reviewing the pilot and its power/cost
forecast, the user deliberately waived that test and chose to close Phase 4 with development evidence
only. This disclosed protocol change is scientifically acceptable, but it leaves H1/H2 unconfirmed and
cannot be used to call H2 established. The pending declaration remains non-executable with both model
calls and test-oracle access prohibited. Test-oracle invocations were zero, every access ledger records
`test_consumed: false`, and test outcomes were never accessed.

## Safety and authority

Live use requires all of:

1. `--allow-live-model`;
2. `WMS_ALLOW_LIVE_MODEL=1`;
3. `OPENAI_API_KEY`, read only inside the adapter;
4. exact config/model/price hashes;
5. a valid cumulative ledger and successful worst-case reservation.

Fake execution, tests, dry-run, replay, and reports do not read the key or call a provider. A durable
responded batch can finish offline. A dispatched request without durable response is retained as
usage-uncertain and is never duplicated. Test authority accepts only a frozen opaque ID set and freeze
hash. No such registry was frozen; the closed test declaration records the user waiver and permits
neither oracle nor model access.

Retries are restricted to a provider-established zero-usage rate limit or a charged malformed envelope.
Ambiguous transport usage is terminal. A request that cannot fit its next worst-case reservation stops
as `cost-cap-exhausted` before dispatch and remains replayable/reportable.

## Reproduction

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

If the cumulative ledger file is missing but paid run databases/artifacts remain, the explicit recovery
command verifies their policy and content hashes before atomically installing a reconstructed ledger:

```console
uv run --locked wms ledger rebuild
```

It refuses orphan paid responses, policy mismatches, corrupt hashes, or an in-progress rebuild. It
cannot infer externally deleted local spend; a reconciled opening balance and the provider dashboard are
required in that loss scenario.

The fake aggregate's deterministic summary hash is
`56cc7c3236910da239f72722c553aa46e4ebe094fe33510420e6054864398320`. The generated presentable
artifact is `artifacts/reports/phase4-fake-smoke-v1/phase4-artifact.json`, SHA-256
`73598cfd022c027f30feab54766d4e578f942c753e4350b4c9ce045574c6fbad`.

Fake H1 and H2 are both zero with degenerate machinery-test intervals, clustered two-sided p-values
`1.0`, Holm-adjusted p-values `1.0`, and no rejection. They validate null-result export but support no
scientific inference. Before live execution, pilot v2's dry run planned 60 children (10 development
tasks x two seeds x three conditions), 256 evaluations per child, batch size four, and up to 63 logical
model calls per child. It verified the exact successful canary record/hash and all hierarchical budget
ceilings without hidden access or provider calls. The subsequent authorized live run completed those
60 children; its results and closeout disposition are recorded above.

Final CI reports 92 files formatted, Ruff clean, strict mypy clean over 64 source files, and 97 tests
passed. Representative Phase 0, 2, and 3 interrupt/resume/replay/report flows also pass; their replay
paths record zero proposer invocations. The documented `PHASE4-FAKE-SMOKE` run records 11 events, two
scripted physical calls, 15,254 input and 160 output tokens, 11 oracle evaluations, zero dollars, and
deterministic summary hash
`0f8c1a6e48321bc2969437863a40b547d93c4bc5f65954a4a4f5f50019255a31`.

## Limitation and phase boundary

The repository still contains only F0: binary one-dimensional radius-one elementary cellular automata.
F1/F2 were not added as incidental LLM work. This small 256-semantics universe and frozen grammar do not
support claims of broad world-model discovery or LLM superiority.

No cross-task memory, learned primitive, interestingness scheduler, active query, partial observability,
scaffold modification, or Phase 5-8 mechanism was implemented. In particular, no Phase 5 mechanism has
been implemented yet.
