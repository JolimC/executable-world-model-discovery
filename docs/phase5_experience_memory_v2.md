# Phase 5 v2: experience-derived cross-task memory

## Status

The provider-disabled retrospective training preparation is complete. Phase 4 condition C was reused;
no training search was rerun and no canonical/reference AST was read. Three lesson-induction requests
are frozen but unexecuted. Prospective validation and the development pilot remain unauthorized.

This mechanism is separate from the completed Phase 5 v1 pilot and does not reinterpret its result.

## Retrospective training evidence

The source is the frozen `uniform-diverse-archive-v1` arm of `phase4-primary-pilot-v2`: 20 runs over
10 tasks, all from the evaluator family `elementary-radius1-binary`. These Phase 4 tasks originally had
the `development` split. They are explicitly redesignated as **v2 retrospective training only** and
cannot be used again for v2 validation, development, or test.

For each of the nine runs that solved exactly, the extractor selects the request that produced the
first exact child. It records:

- the complete candidate/score lineage ending in that exact child;
- the three non-exact sibling lineages produced by the same four-candidate request from the same
  selected parent;
- the selected parent cell, request index, run hash, original split, and evaluator-only source family.

This produces nine matched contrast records and 27 matched unsuccessful lineages over five distinct
solved tasks. The unsuccessful siblings let the inducer compare what changed successfully with changes
that failed under the same immediate search context. They reduce local survivor bias, although they do
not erase the broader limitation that the source corpus contains only runs/tasks from one generator
family and was selected retrospectively.

## Representation-family assignment and induction

Evidence is assigned to the public representation family of the **selected parent cell for the
consequential request**. It is not assigned from the exact child's final syntax. The resulting support
is:

| Parent representation family | Contrast records | Independent tasks | Matched failures | Induction |
|---|---:|---:|---:|---|
| `mixed` | 3 | 3 | 9 | eligible |
| `position-specific` | 3 | 3 | 9 | eligible |
| `threshold` | 2 | 2 | 6 | eligible |
| `conditional` | 1 | 1 | 3 | retained for audit; ineligible |

Exactly one strict structured-output induction package is prepared for each eligible family. Each
package contains only that family's sanitized matched contrasts and requests one strongest lesson.
Generator-family labels, reference programs, semantic hashes, error locations, hidden artifacts, and
sealed data are absent. The prepared packages set `provider_dispatch_authorized` to false.

The frozen preparation manifest is
`artifacts/phase5-experience-v2/retrospective-training/manifest.json`. It binds corpus hash
`cca04ce74820fdfda46b33eaeafd1cbecdec006163b3e65d76192536459758b8` and reports three prepared,
zero executed provider requests. Their combined conservative published-rate ceiling is `$0.021299`;
that forecast is not spending authority.

## Two-stage prospective validation

The current mechanism-development claim is deliberately within the same broad generator family. A
second generator family is therefore not required for this v2 pilot, and no cross-generator-family
generalization claim is allowed. All validation tasks must nevertheless be new and semantically
disjoint from Phase 4, both validation stages, development, and sealed test as applicable.

Stage 1 screens lessons individually. One shared memoryless control run is paired with separate
treatment arms, and each treatment arm contains exactly one lesson. A lesson is promoted only after it
was actually retrieved on at least four distinct prospective tasks, has positive mean paired normalized
exact-AUC, and causes no exact-solve regression. Insufficient retrieval exposure is `inconclusive`, not
evidence that the lesson failed.

Stage 2 confirms the joint bundle on fresh tasks not used in Stage 1. It compares empty memory with
cell-conditioned retrieval over all Stage-1-promoted lessons. The bundle needs memory exposure on at
least four distinct tasks, exposure of every included lesson on at least two tasks, positive mean paired
normalized exact-AUC, and no exact-solve regression. Only a passing bundle can produce the frozen
development memory snapshot; its snapshot hash binds the Stage 2 validation-pair hashes.

Validation may add predeclared fresh-task blocks until those exposure minima or a frozen task/cost cap
is reached. The complete block schedule and maximum spend must be frozen before the first prospective
call, which avoids silently expanding a weak result after looking at outcomes.

## Cell-conditioned retrieval

After the uniform scheduler chooses an occupied archive cell, retrieval filters the frozen snapshot to
lessons whose assigned representation family exactly matches that cell. It then orders lessons by
validation gain and lesson ID, applies item/byte/conservative-token bounds, and injects one canonical
JSON block into the iterative proposal prompt. There is no global fallback. The memoryless arm receives
the corresponding empty block, and matched-prompt checks permit no other difference.

## What remains before development

1. Authorize and execute the three frozen lesson-induction requests.
2. Freeze fresh validation tasks, adaptive exposure blocks, request ceilings, and a cost forecast.
3. Authorize and run individual-lesson screening, then fresh-task bundle confirmation.
4. If the bundle passes, freeze its memory snapshot and a separate fresh development registry.
5. Run a very small engineering canary to verify routing/accounting, then the development pilot.

A smaller **scientific** development pilot is not recommended. Family-conditioned retrieval is sparse,
and the previous eight-pair Phase 5 pilot produced zero exact solves in both arms. Fewer than eight
matched task-seed pairs would be especially likely to yield no exact events or too few family-specific
retrieval exposures. A one-task engineering canary is still worthwhile, but it should not be analyzed as
the pilot.

No provider call, prospective validation run, development pilot, or sealed-test access is authorized by
the current files.
