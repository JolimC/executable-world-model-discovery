"""Frozen Phase 2 language and analysis version identifiers."""

DSL_VERSION = "binary-ca-radius1-dsl-v1"
CANDIDATE_SCHEMA_VERSION = 1
INTERPRETER_VERSION = "binary-ca-radius1-interpreter-v1"
CANONICALIZER_VERSION = "binary-ca-canonicalizer-v1"
SEMANTIC_HASH_VERSION = "elementary-local-semantics-v1"
PREFIX_CODE_VERSION = "binary-ca-prefix-v1"
RESIDUAL_CODE_VERSION = "enumerative-residual-gamma-v1"
RANK_VERSION = "correctness-first-rank-v1"
ENUMERATOR_VERSION = "cost-ordered-semantic-first-v1"
TRUTH_TABLE_BASELINE_VERSION = "elementary-truth-table-v1"
ANALYSIS_VERSION = "phase2-elementary-analysis-v1"
ANALYSIS_ARTIFACT_VERSION = "phase2-analysis-bundle-v1"

# Phase 3 extends the search protocol without changing any Phase 2 language meaning.
PHASE3_CONFIG_SCHEMA_VERSION = 3
PHASE3_CANDIDATE_IDENTITY_VERSION = "phase3-candidate-identity-v1"
PHASE3_OPERATOR_VERSION = "typed-local-operators-v1"
PHASE3_RNG_VERSION = "sha256-counter-streams-v1"
PHASE3_DESCRIPTOR_VERSION = "public-probe-descriptor-v1"
PHASE3_ARCHIVE_VERSION = "map-elites-lineage-reserve-v1"
PHASE3_INCUMBENT_VERSION = "single-incumbent-v1"
PHASE3_SCHEDULER_VERSION = "uniform-sorted-branches-v1"
PHASE3_BUDGET_VERSION = "charged-oracle-evaluation-v1"
PHASE3_INITIALIZATION_VERSION = "public-dsl-baselines-v1"
PHASE3_ANALYSIS_VERSION = "paired-exact-auc-bootstrap-v1"
PHASE3_EVENT_SCHEMA_VERSION = 3
PHASE3_RESULTS_SCHEMA_VERSION = 3
# Schema 4 adds an optional, non-replay-stable timing table while retaining the
# deterministic schema-3 event/result contract for old frozen runs.
PHASE3_DATABASE_SCHEMA_VERSION = 4
PHASE3_MANIFEST_SCHEMA_VERSION = 4
PHASE3_PROPOSAL_ARTIFACT_VERSION = "phase3-proposal-attempt-v1"
PHASE3_LINEAGE_VERSION = "phase3-lineage-dag-v1"
PHASE3_EXPERIMENT_SCHEMA_VERSION = 1
