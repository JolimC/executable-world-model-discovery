"""Offline preparation of Phase 4 condition-C experience for Phase 5 v2.

This module has no model backend and no oracle dependency.  It reads the frozen Phase 4
SQLite ledgers, identifies the first exact-solving request in each condition-C run, pairs the
successful child with non-exact siblings from that same request and selected parent, and emits
one proposer-safe lesson-induction package per eligible parent representation family.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from world_model_search.domain.types import ProposalRole, SplitLabel
from world_model_search.dsl.json_schema import DslCandidateDocument
from world_model_search.errors import ConfigurationError, PersistenceError
from world_model_search.memory.experience import (
    ExperienceLessonProposal,
    ExperienceLineageStep,
    MatchedUnsuccessfulLineage,
    SuccessfulLineageEvidence,
)
from world_model_search.model.backends import LiveOptIn, OpenAIResponsesBackend
from world_model_search.model.ledger import ProjectLedger
from world_model_search.model.phase5_experience_prompts import (
    LESSON_INDUCTION_PROMPT_VERSION,
    lesson_induction_json_schema,
    parse_lesson_induction_response,
    render_lesson_induction_prompt,
)
from world_model_search.model.policy import load_price_policy
from world_model_search.model.types import ModelDispatchError, ModelRequest, ModelResponse
from world_model_search.persistence.artifacts import read_text_artifact, write_content_artifact
from world_model_search.search.archive import (
    ArchiveCoordinate,
    ArchiveLayer,
    RepresentationFamily,
)
from world_model_search.serialization import (
    JsonObject,
    canonical_json,
    parse_json_object,
    sha256_json,
    sha256_text,
)

RETROSPECTIVE_CORPUS_VERSION = "phase5-phase4-c-retrospective-contrast-corpus-v1"
INDUCTION_PACKAGE_VERSION = "phase5-family-induction-package-v1"
INDUCTION_REQUEST_IDENTITY_VERSION = "phase5-experience-induction-request-v1"
PREPARATION_MANIFEST_VERSION = "phase5-experience-preparation-manifest-v1"
SOURCE_EXPERIMENT_ID = "phase4-primary-pilot-v2"
SOURCE_CONDITION_ID = "uniform-diverse-archive-v1"
SOURCE_GENERATOR_FAMILY_ID = "elementary-radius1-binary"
SOURCE_REGISTRY_PATH = Path("experiments/phase4-primary-pilot.yaml")
SOURCE_EXPERIMENT_ROOT = Path("artifacts/experiments/phase4-primary-pilot-v2")
SOURCE_RUNS_ROOT = Path("artifacts/phase4-runs/phase4-primary-pilot-v2")
SOURCE_REGISTRY_HASH = "4c05276973bb3b2c97e0c17387649ae27fb015a598ed75c3866f23770c965804"
SOURCE_MANIFEST_HASH = "7e0215fce11e0bb89a7bf9244edaa48f83cf80d86f1bb1f8f1bd9b563f8614eb"
SOURCE_RAW_ROWS_HASH = "f72ce478b9c43719d67b25524244951482d480837be8412f722d93be586fd222"
MINIMUM_INDUCTION_TASKS = 2
INDUCTION_MAX_OUTPUT_TOKENS = 2_048
UNCACHED_INPUT_NANO_USD_PER_TOKEN = 250
OUTPUT_NANO_USD_PER_TOKEN = 2_000
ONE_REQUEST_CEILING_NANO_USD = 10_000_000
EXPERIENCE_INDUCTION_RESULT_VERSION = "phase5-experience-induction-results-v1"


def _read_object(path: Path) -> JsonObject:
    try:
        return parse_json_object(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PersistenceError(f"cannot read frozen Phase 4 source artifact: {path}") from exc


def _integer_field(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{label} must be an integer")
    return value


def _verify_text_hash(path: Path, expected: str) -> None:
    try:
        actual = sha256_text(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PersistenceError(f"cannot read frozen source artifact: {path}") from exc
    if actual != expected:
        raise ConfigurationError(f"frozen Phase 4 source hash differs: {path}")


def _string_list(value: object, label: str) -> tuple[str, ...]:
    try:
        parsed: object = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise PersistenceError(f"{label} is not JSON") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise PersistenceError(f"{label} must be a string array")
    return tuple(parsed)


def _coordinate(value: object) -> ArchiveCoordinate:
    try:
        raw: object = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise PersistenceError("Phase 4 archive coordinate is not JSON") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "descriptor_version",
        "size_bin",
        "representation_family",
        "error_signature_cluster",
        "layer",
    }:
        raise PersistenceError("Phase 4 archive coordinate is malformed")
    if raw.get("descriptor_version") != "public-probe-descriptor-v1":
        raise PersistenceError("Phase 4 descriptor version differs")
    try:
        return ArchiveCoordinate(
            size_bin=str(raw["size_bin"]),
            representation_family=RepresentationFamily(str(raw["representation_family"])),
            error_signature_cluster=str(raw["error_signature_cluster"]),
            layer=ArchiveLayer(str(raw["layer"])),
        )
    except ValueError as exc:
        raise PersistenceError("Phase 4 archive coordinate has an unknown enum") from exc


def _result(value: object) -> JsonObject:
    try:
        parsed: object = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise PersistenceError("Phase 4 evaluation result is not JSON") from exc
    if not isinstance(parsed, dict):
        raise PersistenceError("Phase 4 evaluation result must be an object")
    return cast(JsonObject, parsed)


@dataclass(frozen=True, slots=True)
class Phase4RetrospectiveCorpus:
    evidence: tuple[SuccessfulLineageEvidence, ...]
    source_run_count: int
    exact_run_count: int
    source_task_count: int

    def __post_init__(self) -> None:
        if self.source_run_count != 20 or self.source_task_count != 10:
            raise ValueError("Phase 4 retrospective source cardinality differs")
        if self.exact_run_count != len(self.evidence) or self.exact_run_count != 9:
            raise ValueError("Phase 4 first-exact evidence cardinality differs")
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("retrospective corpus contains duplicate evidence")
        if {item.generator_family_id for item in self.evidence} != {SOURCE_GENERATOR_FAMILY_ID}:
            raise ValueError("retrospective corpus is not the declared single source family")

    def induction_groups(
        self, *, minimum_tasks: int = MINIMUM_INDUCTION_TASKS
    ) -> dict[RepresentationFamily, tuple[SuccessfulLineageEvidence, ...]]:
        if minimum_tasks < 2:
            raise ValueError("lesson induction needs at least two independent tasks")
        grouped: dict[RepresentationFamily, list[SuccessfulLineageEvidence]] = defaultdict(list)
        for item in self.evidence:
            grouped[item.archive_representation_family].append(item)
        return {
            family: tuple(sorted(items, key=lambda item: item.evidence_id))
            for family, items in sorted(grouped.items(), key=lambda pair: pair[0].value)
            if len({item.task_id for item in items}) >= minimum_tasks
        }

    def family_summary(self) -> JsonObject:
        grouped: dict[RepresentationFamily, list[SuccessfulLineageEvidence]] = defaultdict(list)
        for item in self.evidence:
            grouped[item.archive_representation_family].append(item)
        return cast(
            JsonObject,
            {
                family.value: {
                    "contrast_count": len(items),
                    "task_count": len({item.task_id for item in items}),
                    "matched_unsuccessful_lineage_count": sum(
                        len(item.matched_unsuccessful_lineages) for item in items
                    ),
                    "induction_eligible": len({item.task_id for item in items})
                    >= MINIMUM_INDUCTION_TASKS,
                }
                for family, items in sorted(grouped.items(), key=lambda pair: pair[0].value)
            },
        )

    def to_value(self) -> JsonObject:
        return cast(
            JsonObject,
            {
                "corpus_version": RETROSPECTIVE_CORPUS_VERSION,
                "designation": "retrospective-single-source-family-training",
                "source": {
                    "experiment_id": SOURCE_EXPERIMENT_ID,
                    "condition_id": SOURCE_CONDITION_ID,
                    "generator_family_id": SOURCE_GENERATOR_FAMILY_ID,
                    "original_task_split": SplitLabel.DEVELOPMENT.value,
                    "phase5_memory_role": SplitLabel.TRAINING.value,
                    "registry_path": str(SOURCE_REGISTRY_PATH),
                    "registry_hash": SOURCE_REGISTRY_HASH,
                    "experiment_manifest_hash": SOURCE_MANIFEST_HASH,
                    "raw_rows_hash": SOURCE_RAW_ROWS_HASH,
                },
                "selection": {
                    "successful_lineage": "first-exact-child-per-solving-run",
                    "consequential_revision": "request-that-produced-first-exact-child",
                    "family_assignment": "selected-parent-cell-representation-family",
                    "matched_unsuccessful": "all-nonexact-siblings-from-same-request-and-parent",
                    "canonical_reference_ast_used": False,
                },
                "counts": {
                    "condition_c_runs": self.source_run_count,
                    "condition_c_tasks": self.source_task_count,
                    "exact_solving_runs": self.exact_run_count,
                    "exact_solving_tasks": len({item.task_id for item in self.evidence}),
                    "contrast_records": len(self.evidence),
                    "matched_unsuccessful_lineages": sum(
                        len(item.matched_unsuccessful_lineages) for item in self.evidence
                    ),
                },
                "family_summary": self.family_summary(),
                "evidence": [item.evaluator_value() for item in self.evidence],
                "limitations": [
                    "retrospective-source-selection",
                    "single-generator-family",
                    "phase4-development-tasks-redesignated-as-v2-training-only",
                    "no-cross-generator-family-generalization-claim",
                ],
            },
        )

    @property
    def corpus_hash(self) -> str:
        return sha256_json(self.to_value())


def _extract_run(run_directory: Path, raw_row: JsonObject) -> SuccessfulLineageEvidence | None:
    database_path = run_directory / "run.sqlite3"
    results = _read_object(run_directory / "results.json")
    if results.get("status") != "completed" or results.get("condition_id") != SOURCE_CONDITION_ID:
        raise PersistenceError("Phase 4 source run is not a completed condition-C run")
    if results.get("deterministic_summary_hash") != raw_row.get("deterministic_summary_hash"):
        raise PersistenceError("Phase 4 run result differs from experiment raw row")
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise PersistenceError(f"cannot open Phase 4 run database: {database_path}") from exc
    try:
        task_row = connection.execute("SELECT * FROM task").fetchone()
        if task_row is None:
            raise PersistenceError("Phase 4 run has no task record")
        if str(task_row["internal_family_id"]) != SOURCE_GENERATOR_FAMILY_ID:
            raise PersistenceError("Phase 4 source generator family differs")
        if str(task_row["split"]) != SplitLabel.DEVELOPMENT.value:
            raise PersistenceError("Phase 4 source task split differs")
        candidate_rows = {
            str(row["candidate_id"]): dict(row)
            for row in connection.execute("SELECT * FROM candidate")
        }
        evaluation_rows = {
            str(row["candidate_id"]): dict(row)
            for row in connection.execute("SELECT * FROM evaluation ORDER BY evaluation_index")
        }
        transition_rows = {
            int(row["evaluation_index"]): dict(row)
            for row in connection.execute("SELECT * FROM archive_transition")
        }
        exact_rows = [
            row
            for row in evaluation_rows.values()
            if _result(row["result_json"]).get("exact") is True
            and isinstance(row.get("request_index"), int)
        ]
        if not exact_rows:
            return None
        first_exact = min(exact_rows, key=lambda row: int(row["evaluation_index"]))
        request_index = int(first_exact["request_index"])
        request_row = connection.execute(
            "SELECT * FROM model_request WHERE request_index=?", (request_index,)
        ).fetchone()
        if request_row is None:
            raise PersistenceError("first exact evaluation has no model request")
        selected_parents = _string_list(
            request_row["ordered_parent_ids_json"], "selected request parents"
        )
        if len(selected_parents) != 1:
            raise PersistenceError("condition-C experience requires one selected parent")
        selected_parent = selected_parents[0]

        def step(candidate_id: str) -> ExperienceLineageStep:
            try:
                candidate_row = candidate_rows[candidate_id]
                evaluation_row = evaluation_rows[candidate_id]
                transition_row = transition_rows[int(evaluation_row["evaluation_index"])]
            except KeyError as exc:
                raise PersistenceError("Phase 4 lineage record is incomplete") from exc
            document = DslCandidateDocument.from_json(
                canonical_json(
                    {
                        "candidate_schema_version": 1,
                        "dsl_version": "binary-ca-radius1-dsl-v1",
                        "ast": json.loads(str(candidate_row["canonical_ast_json"])),
                    }
                )
            )
            score = _result(evaluation_row["result_json"])
            integer_names = ("local_errors", "local_cases", "ast_bits", "residual_bits")
            if any(
                isinstance(score.get(name), bool) or not isinstance(score.get(name), int)
                for name in integer_names
            ):
                raise PersistenceError("Phase 4 score record is malformed")
            return ExperienceLineageStep(
                candidate_id=candidate_id,
                ordered_parent_ids=_string_list(
                    candidate_row["parent_ids_json"], "candidate parent IDs"
                ),
                ast=document.ast,
                local_errors=cast(int, score["local_errors"]),
                local_cases=cast(int, score["local_cases"]),
                exact=score.get("exact") is True,
                ast_bits=cast(int, score["ast_bits"]),
                residual_bits=cast(int, score["residual_bits"]),
                archive_coordinate=_coordinate(transition_row["coordinate_json"]),
            )

        def lineage(candidate_id: str) -> tuple[ExperienceLineageStep, ...]:
            ordered: list[ExperienceLineageStep] = []
            seen: set[str] = set()
            active: set[str] = set()

            def visit(current: str) -> None:
                if current in seen:
                    return
                if current in active:
                    raise PersistenceError("Phase 4 lineage contains a cycle")
                active.add(current)
                current_step = step(current)
                for parent_id in current_step.ordered_parent_ids:
                    visit(parent_id)
                active.remove(current)
                seen.add(current)
                ordered.append(current_step)

            visit(candidate_id)
            return tuple(ordered)

        successful_steps = lineage(str(first_exact["candidate_id"]))
        matched: list[MatchedUnsuccessfulLineage] = []
        siblings = connection.execute(
            """SELECT evaluation.candidate_id,evaluation.result_json,candidate.parent_ids_json
               FROM evaluation JOIN candidate USING(candidate_id)
               WHERE evaluation.request_index=? ORDER BY evaluation.item_ordinal""",
            (request_index,),
        )
        for sibling in siblings:
            candidate_id = str(sibling["candidate_id"])
            score = _result(sibling["result_json"])
            parents = _string_list(sibling["parent_ids_json"], "sibling parent IDs")
            if score.get("exact") is not True and selected_parent in parents:
                matched.append(MatchedUnsuccessfulLineage(lineage(candidate_id)))
        if len(matched) != 3:
            raise PersistenceError("first-exact request does not have three matched failures")
        source_record_hash = sha256_json(
            {
                "source_record_version": "phase4-first-exact-request-contrast-v1",
                "run_id": run_directory.name,
                "request_index": request_index,
                "selected_parent_candidate_id": selected_parent,
                "successful_lineage": [item.proposer_value() for item in successful_steps],
                "matched_unsuccessful_lineages": [item.evaluator_value() for item in matched],
            }
        )
        run_hash = results.get("deterministic_summary_hash")
        search_seed = raw_row.get("search_seed")
        task_id = raw_row.get("task_id")
        if (
            not isinstance(run_hash, str)
            or not isinstance(search_seed, int)
            or not isinstance(task_id, str)
        ):
            raise PersistenceError("Phase 4 raw row identity is malformed")
        return SuccessfulLineageEvidence(
            task_id=task_id,
            generator_family_id=str(task_row["internal_family_id"]),
            role=SplitLabel.TRAINING,
            source_task_split=SplitLabel(str(task_row["split"])),
            search_seed=search_seed,
            consequential_request_index=request_index,
            selected_parent_candidate_id=selected_parent,
            steps=successful_steps,
            matched_unsuccessful_lineages=tuple(matched),
            run_hash=run_hash,
            artifact_hash=source_record_hash,
        )
    except PersistenceError:
        raise
    except (sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
        raise PersistenceError(f"cannot extract Phase 4 experience: {run_directory.name}") from exc
    finally:
        connection.close()


def extract_phase4_condition_c_corpus(*, repository_root: Path) -> Phase4RetrospectiveCorpus:
    """Build the immutable retrospective corpus without provider or oracle access."""

    _verify_text_hash(repository_root / SOURCE_REGISTRY_PATH, SOURCE_REGISTRY_HASH)
    experiment_root = repository_root / SOURCE_EXPERIMENT_ROOT
    _verify_text_hash(experiment_root / "experiment-manifest.json", SOURCE_MANIFEST_HASH)
    raw_path = experiment_root / "analysis" / "raw-rows.json"
    _verify_text_hash(raw_path, SOURCE_RAW_ROWS_HASH)
    raw = _read_object(raw_path)
    rows = raw.get("rows")
    if not isinstance(rows, list):
        raise PersistenceError("Phase 4 raw rows artifact is malformed")
    condition_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("condition_id") == SOURCE_CONDITION_ID
    ]
    if len(condition_rows) != 20:
        raise PersistenceError("Phase 4 condition-C source run count differs")
    evidence: list[SuccessfulLineageEvidence] = []
    for row in sorted(condition_rows, key=lambda item: str(item.get("run_id"))):
        run_id = row.get("run_id")
        if not isinstance(run_id, str):
            raise PersistenceError("Phase 4 raw row has no run ID")
        item = _extract_run(repository_root / SOURCE_RUNS_ROOT / run_id, row)
        if item is not None:
            evidence.append(item)
    return Phase4RetrospectiveCorpus(
        evidence=tuple(sorted(evidence, key=lambda item: item.evidence_id)),
        source_run_count=len(condition_rows),
        exact_run_count=len(evidence),
        source_task_count=len({str(row.get("task_id")) for row in condition_rows}),
    )


def prepare_phase5_experience_induction(
    *,
    repository_root: Path,
    output_root: Path,
    requested_lessons_per_family: int = 1,
) -> JsonObject:
    """Freeze corpus and one unexecuted induction package per eligible representation family."""

    if output_root.is_absolute() or ".." in output_root.parts:
        raise ConfigurationError("experience output root must be repository-relative without '..'")
    if requested_lessons_per_family < 1:
        raise ConfigurationError("requested lessons per family must be positive")
    corpus = extract_phase4_condition_c_corpus(repository_root=repository_root)
    destination = repository_root / output_root
    corpus_path = destination / "retrospective-corpus.json"
    corpus_artifact_hash = write_content_artifact(corpus_path, canonical_json(corpus.to_value()))
    packages: list[JsonObject] = []
    for family, evidence in corpus.induction_groups().items():
        prompt = render_lesson_induction_prompt(
            evidence=evidence,
            requested_lessons=requested_lessons_per_family,
            representation_family=family,
        )
        schema = lesson_induction_json_schema(
            requested_lessons=requested_lessons_per_family,
            representation_family=family,
        )
        request = ModelRequest(
            backend_id="openai-responses-sdk-v1",
            provider_id="openai",
            resolved_model="gpt-5-mini-2025-08-07",
            endpoint="v1/responses",
            service_tier="default",
            prompt_template="family-lineage-contrast-induction",
            prompt_version=LESSON_INDUCTION_PROMPT_VERSION,
            rendered_input=prompt,
            structured_schema_name=f"phase5_{family.value.replace('-', '_')}_lessons_v1",
            structured_schema_version=1,
            structured_schema=schema,
            role=ProposalRole.EXPLOIT,
            requested_batch_size=requested_lessons_per_family,
            settings={
                "reasoning": {"effort": "low"},
                "max_output_tokens": INDUCTION_MAX_OUTPUT_TOKENS,
                "store": False,
                "truncation": "disabled",
            },
        )
        conservative_input_token_bound = request.conservative_input_token_bound
        maximum_nano_usd = (
            conservative_input_token_bound * UNCACHED_INPUT_NANO_USD_PER_TOKEN
            + INDUCTION_MAX_OUTPUT_TOKENS * OUTPUT_NANO_USD_PER_TOKEN
        )
        if maximum_nano_usd > ONE_REQUEST_CEILING_NANO_USD:
            raise ConfigurationError("prepared induction request exceeds one-request ceiling")
        identity: JsonObject = {
            "package_version": INDUCTION_PACKAGE_VERSION,
            "request_identity_version": INDUCTION_REQUEST_IDENTITY_VERSION,
            "prompt_version": LESSON_INDUCTION_PROMPT_VERSION,
            "archive_representation_family": family.value,
            "source_corpus_hash": corpus.corpus_hash,
            "source_evidence_ids": [item.evidence_id for item in evidence],
            "requested_lessons": requested_lessons_per_family,
            "rendered_input_utf8": prompt,
            "structured_output": {
                "name": f"phase5_{family.value.replace('-', '_')}_lessons_v1",
                "strict": True,
                "schema": schema,
            },
            "model_contract": {
                "backend": "openai-responses-sdk-v1",
                "provider": "openai",
                "model": "gpt-5-mini-2025-08-07",
                "endpoint": "v1/responses",
                "service_tier": "default",
                "reasoning_effort": "low",
                "max_output_tokens": INDUCTION_MAX_OUTPUT_TOKENS,
                "store": False,
                "truncation": "disabled",
            },
            "budget": {
                "conservative_input_token_bound": conservative_input_token_bound,
                "maximum_output_tokens": INDUCTION_MAX_OUTPUT_TOKENS,
                "maximum_published_rate_nano_usd": maximum_nano_usd,
                "one_request_ceiling_nano_usd": ONE_REQUEST_CEILING_NANO_USD,
            },
            "provider_dispatch_authorized": False,
        }
        package_hash = write_content_artifact(
            destination / "induction" / f"{family.value}.request.json",
            canonical_json(identity),
        )
        packages.append(
            {
                "archive_representation_family": family.value,
                "request_artifact": str(output_root / "induction" / f"{family.value}.request.json"),
                "request_artifact_hash": package_hash,
                "source_task_count": len({item.task_id for item in evidence}),
                "source_contrast_count": len(evidence),
                "conservative_input_token_bound": conservative_input_token_bound,
                "maximum_published_rate_nano_usd": maximum_nano_usd,
            }
        )
    prepared_families = {str(item["archive_representation_family"]) for item in packages}
    excluded = [
        {
            "archive_representation_family": family,
            "reason": "fewer-than-two-independent-source-tasks",
        }
        for family, value in corpus.family_summary().items()
        if isinstance(value, dict)
        and value.get("induction_eligible") is False
        and family not in prepared_families
    ]
    manifest: JsonObject = cast(
        JsonObject,
        {
            "manifest_version": PREPARATION_MANIFEST_VERSION,
            "status": "retrospective-training-prepared-induction-not-authorized",
            "source_corpus_artifact": str(output_root / "retrospective-corpus.json"),
            "source_corpus_artifact_hash": corpus_artifact_hash,
            "source_corpus_hash": corpus.corpus_hash,
            "induction_policy": {
                "one_request_per_representation_family": True,
                "requested_lessons_per_family": requested_lessons_per_family,
                "minimum_independent_source_tasks": MINIMUM_INDUCTION_TASKS,
            },
            "prepared_induction_requests": packages,
            "excluded_representation_families": excluded,
            "provider_requests_prepared": len(packages),
            "provider_requests_executed": 0,
            "maximum_induction_published_rate_nano_usd": sum(
                cast(int, item["maximum_published_rate_nano_usd"]) for item in packages
            ),
            "provider_dispatch_authorized": False,
        },
    )
    manifest_hash = write_content_artifact(destination / "manifest.json", canonical_json(manifest))
    return cast(
        JsonObject,
        {
            "status": manifest["status"],
            "output_root": str(output_root),
            "manifest_hash": manifest_hash,
            "corpus_hash": corpus.corpus_hash,
            "prepared_induction_requests": len(packages),
            "provider_requests_executed": 0,
        },
    )


def execute_phase5_experience_induction(
    *,
    repository_root: Path,
    output_root: Path,
    price_policy_path: Path,
    ledger_path: Path,
    allow_live_model: bool,
) -> JsonObject:
    """Execute the three frozen per-family packages with durable ledger accounting."""

    destination = repository_root / output_root
    manifest = _read_object(destination / "manifest.json")
    if manifest.get("manifest_version") != PREPARATION_MANIFEST_VERSION:
        raise ConfigurationError("experience induction manifest version differs")
    package_rows = manifest.get("prepared_induction_requests")
    if not isinstance(package_rows, list) or len(package_rows) != 3:
        raise ConfigurationError("experience induction requires exactly three frozen packages")
    corpus = extract_phase4_condition_c_corpus(repository_root=repository_root)
    if manifest.get("source_corpus_hash") != corpus.corpus_hash:
        raise ConfigurationError("experience induction corpus hash differs")
    catalog = {item.evidence_id: item for item in corpus.evidence}
    policy = load_price_policy(repository_root / price_policy_path)
    backend = OpenAIResponsesBackend(opt_in=LiveOptIn.resolve(allow_live_model))
    proposals: list[ExperienceLessonProposal] = []
    response_rows: list[JsonObject] = []
    with ProjectLedger(repository_root / ledger_path, policy) as ledger:
        for package_row in package_rows:
            if not isinstance(package_row, dict):
                raise ConfigurationError("experience induction package row is malformed")
            relative = Path(str(package_row.get("request_artifact")))
            package_text = read_text_artifact(repository_root / relative)
            if sha256_text(package_text) != package_row.get("request_artifact_hash"):
                raise ConfigurationError("experience induction package hash differs")
            package = parse_json_object(package_text)
            family = RepresentationFamily(str(package.get("archive_representation_family")))
            structured = package.get("structured_output")
            model = package.get("model_contract")
            budget = package.get("budget")
            if (
                not isinstance(structured, dict)
                or not isinstance(model, dict)
                or not isinstance(budget, dict)
            ):
                raise ConfigurationError("experience induction package sections are malformed")
            schema = structured.get("schema")
            if not isinstance(schema, dict):
                raise ConfigurationError("experience induction schema is malformed")
            request = ModelRequest(
                backend_id=str(model.get("backend")),
                provider_id=str(model.get("provider")),
                resolved_model=str(model.get("model")),
                endpoint=str(model.get("endpoint")),
                service_tier=str(model.get("service_tier")),
                prompt_template="family-lineage-contrast-induction",
                prompt_version=str(package.get("prompt_version")),
                rendered_input=str(package.get("rendered_input_utf8")),
                structured_schema_name=str(structured.get("name")),
                structured_schema_version=1,
                structured_schema=schema,
                role=ProposalRole.EXPLOIT,
                requested_batch_size=_integer_field(
                    package.get("requested_lessons"), "requested lessons"
                ),
                settings={
                    "reasoning": {"effort": str(model.get("reasoning_effort"))},
                    "max_output_tokens": _integer_field(
                        model.get("max_output_tokens"), "maximum output tokens"
                    ),
                    "store": False,
                    "truncation": "disabled",
                },
            )
            response_path = destination / "induction" / f"{family.value}.response.json"
            if response_path.is_file():
                response_artifact = _read_object(response_path)
                response = ModelResponse.from_deterministic_value(response_artifact.get("response"))
            else:
                maximum = _integer_field(
                    budget.get("maximum_published_rate_nano_usd"), "maximum request cost"
                )
                independently_computed = policy.price.maximum_cost(
                    input_token_bound=request.conservative_input_token_bound,
                    max_output_tokens=_integer_field(
                        model.get("max_output_tokens"), "maximum output tokens"
                    ),
                )
                if maximum != independently_computed or maximum > ONE_REQUEST_CEILING_NANO_USD:
                    raise ConfigurationError("experience induction cost binding differs")
                reservation_id = sha256_text(
                    f"phase5-experience-induction-v1\0{family.value}\0{request.request_hash}"
                )
                ledger.reserve(
                    reservation_id=reservation_id,
                    run_id="phase5-experience-v2-induction",
                    stage="development",
                    request_hash=request.request_hash,
                    amount_nano_usd=maximum,
                    child_cap_nano_usd=150_000_000,
                )
                try:
                    response = backend.dispatch(request)
                except ModelDispatchError as exc:
                    failure: JsonObject = {
                        "artifact_version": "phase5-experience-induction-failure-v1",
                        "family": family.value,
                        "request_hash": request.request_hash,
                        "error": exc.error.to_value(),
                    }
                    write_content_artifact(
                        destination / "induction" / f"{family.value}.failure.json",
                        canonical_json(failure),
                    )
                    if exc.error.usage_uncertain:
                        ledger.mark_uncertain(reservation_id=reservation_id, failure_record=failure)
                    else:
                        ledger.reconcile(
                            reservation_id=reservation_id,
                            actual_nano_usd=0,
                            usage_record={**failure, "actual_nano_usd": 0},
                        )
                    raise
                response_artifact = {
                    "artifact_version": "phase5-experience-induction-response-v1",
                    "family": family.value,
                    "request_hash": request.request_hash,
                    "response": response.deterministic_value(),
                    "diagnostics": {"provider_latency_ns": response.provider_latency_ns},
                }
                response_hash = write_content_artifact(
                    response_path, canonical_json(response_artifact)
                )
                actual = policy.price.cost(response.usage)
                ledger.reconcile(
                    reservation_id=reservation_id,
                    actual_nano_usd=actual,
                    usage_record={
                        "run_id": "phase5-experience-v2-induction",
                        "family": family.value,
                        "request_hash": request.request_hash,
                        "response_hash": response_hash,
                        "usage": response.usage.to_value(),
                        "actual_nano_usd": actual,
                        "price_policy_hash": policy.content_hash,
                    },
                )
            parsed = parse_lesson_induction_response(
                response.raw_text,
                evidence_catalog=catalog,
                requested_lessons=_integer_field(
                    package.get("requested_lessons"), "requested lessons"
                ),
                representation_family=family,
            )
            proposals.extend(parsed)
            response_rows.append(
                {
                    "archive_representation_family": family.value,
                    "request_hash": request.request_hash,
                    "response_artifact": str(response_path.relative_to(repository_root)),
                    "usage": response.usage.to_value(),
                    "lesson_proposal_ids": [item.proposal_id for item in parsed],
                }
            )
    value: JsonObject = cast(
        JsonObject,
        {
            "result_version": EXPERIENCE_INDUCTION_RESULT_VERSION,
            "source_corpus_hash": corpus.corpus_hash,
            "lessons": [
                item.identity_value() | {"proposal_id": item.proposal_id} for item in proposals
            ],
            "responses": response_rows,
        },
    )
    artifact_hash = write_content_artifact(
        destination / "induction" / "frozen-lessons.json", canonical_json(value)
    )
    return {
        "status": "lesson-induction-complete",
        "lesson_count": len(proposals),
        "frozen_lessons_artifact_hash": artifact_hash,
        "families": [item.archive_representation_family.value for item in proposals],
    }
