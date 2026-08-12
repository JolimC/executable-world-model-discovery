# Dual published-rate and cash budget policy

## Purpose

`published-rate-and-reconciled-cash-v2` separates scientific cost accounting from the
user's personal cash limit. It does not modify the closed Phase 4 policy, ledger, run
artifacts, or reported `$6.52680755` published-rate equivalent.

The two records have different jobs:

- Every provider response continues to record exact token usage and its cost under the
  versioned published price table. Promotions never reduce this scientific cost record.
- The personal `$100` ceiling is enforced against the latest reconciled provider-billed
  cash plus every later request at its full published-rate cost.

The enforcement amount is:

```text
latest reconciled cumulative cash
+ published-rate actual cost after that checkpoint
+ retained uncertain usage after that checkpoint
+ active worst-case reservations after that checkpoint
+ configured safety buffer
```

The v2 policy retains the existing request, child, stage, and Phase 4 exposure ceilings.
Only the project-lifetime enforcement basis changes. A promotion can therefore make
cumulative published-equivalent usage exceed `$100`, but only a reconciliation checkpoint
can release the corresponding cash headroom. Future promotions are never presumed.

## Versioned opening carry-forward

[`configs/project-dual-budget-policy-v2.yaml`](../configs/project-dual-budget-policy-v2.yaml)
starts a new ledger rather than editing `local_state/project-cost-ledger.sqlite3`. Its
opening record is:

| Field | Value |
|---|---:|
| Phase 4 published-rate equivalent | `$6.52680755` |
| Provider cash observation | `$4.65` |
| Cash verification | `user-reported-unverified` |
| Uncertain usage | `$0` |
| Active reservation | `$0` |
| Personal cash ceiling | `$100` |

The `$4.65` observation is not represented as an invoice or provider export and does not
rewrite Phase 4 cost. It is an explicitly scoped operational checkpoint supplied by the
user.

## Inspecting the budget

The command is local-only and makes no provider call:

```console
.venv/bin/wms ledger status \
  --policy configs/project-dual-budget-policy-v2.yaml \
  --ledger local_state/project-dual-budget-ledger.sqlite3
```

On first use, this creates the separate schema-v2 ledger from the immutable opening
carry-forward. The output reports the published-rate balance, reconciled cash, unreconciled
actual/uncertain/reserved exposure, cash upper bound, remaining authorizable cash, latest
reservation sequence, and latest contiguous sequence eligible for reconciliation.

Closed Phase 4 configurations intentionally continue to reference the v1 policy and ledger.
A future live configuration opts into dual enforcement together, never one path at a time:

```yaml
phase4:
  price_policy: configs/project-dual-budget-policy-v2.yaml
  ledger: local_state/project-dual-budget-ledger.sqlite3
```

The runner records the v2 policy hash and enforcement basis in each new manifest. Existing
manifest hashes are unaffected.

The complete opening-plus-checkpoint history is available without provider access:

```console
.venv/bin/wms ledger cash-history
```

## Appending a provider checkpoint

After checking the provider dashboard, identify exactly which local reservation sequence
the cumulative amount covers. Then append, rather than overwrite, the observation:

```console
.venv/bin/wms ledger reconcile-cash \
  --billed-usd 7.12 \
  --observed-at 2026-08-15T14:30:00-05:00 \
  --scope "OpenAI project dashboard through reservation sequence 123" \
  --through-sequence 123 \
  --verification user-reported-unverified
```

`--billed-usd` accepts a nonnegative decimal with at most nine fractional digits and is
converted exactly to integer nano-USD. `--observed-at` must be an ISO date or an offset
datetime. Verification may be `user-reported-unverified`, `provider-export-verified`, or
`invoice-verified`.

When there is no active or uncertain request and the dashboard is known to cover all
current finalized usage, this explicit shortcut is available:

```console
.venv/bin/wms ledger reconcile-cash \
  --billed-usd 7.12 \
  --observed-at 2026-08-15T14:30:00-05:00 \
  --scope "OpenAI project dashboard through all current finalized usage" \
  --through-current-finalized
```

Do not use the shortcut if dashboard posting may lag. Retaining recent requests outside
the checkpoint deliberately double-protects the cash limit until their billing is known.
Coverage cannot move backward or include active/uncertain reservations. A lower cumulative
cash observation is rejected unless `--allow-decrease` explicitly records a refund or
correction.

## Scientific and operational boundaries

- Reconciliation changes future cash headroom, never historical token or published cost.
- Reservations remain transactional and concurrency-safe. Each new request must fit both
  its local exposure caps and the personal cash upper bound.
- Cost information may change authorization between experiments. It must not change
  conditions, samples, or hypotheses inside a frozen comparison.
- If the ceiling is reached during a run, the existing replayable cost-cap stop remains the
  correct behavior.
- A missing schema-v2 ledger must be restored from backup after it has paid records. Legacy
  Phase 4 artifacts cannot reconstruct later provider cash checkpoints.
- This accounting infrastructure is not a Phase 5 search or memory mechanism.
