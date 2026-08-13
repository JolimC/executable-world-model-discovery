"""Vendor-neutral Phase 4 LLM proposer, prompts, validation, and request identity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from world_model_search.domain.types import (
    CandidateSummary,
    ProposalBudget,
    ProposalContext,
    ProposalRole,
)
from world_model_search.dsl.ast import AstLimits, BitExpr
from world_model_search.memory.experience import ExperienceRetrievalRecord
from world_model_search.model.cache import ExactResponseCache
from world_model_search.model.phase5_experience_prompts import (
    EXPERIENCE_SEARCH_PROMPT_VERSION,
    inject_experience_memory,
)
from world_model_search.model.prompts import ParentScoreFeedback, render_prompt
from world_model_search.model.schema import (
    BATCH_SCHEMA_NAME,
    BATCH_SCHEMA_VERSION,
    BatchItem,
    CandidateBatch,
    candidate_batch_json_schema,
    parse_candidate_batch,
)
from world_model_search.model.types import ModelBackend, ModelRequest, ModelResponse
from world_model_search.serialization import JsonObject

LLM_PROPOSER_VERSION = "vendor-neutral-llm-proposer-v1"


@dataclass(frozen=True, slots=True)
class LLMProposal:
    ordinal: int
    role: ProposalRole
    source_ast: BitExpr
    canonical_ast: BitExpr
    request_hash: str
    submitted_document: JsonObject
    operator_id: str


@dataclass(frozen=True, slots=True)
class LLMParsedResponse:
    batch: CandidateBatch
    proposals: tuple[LLMProposal, ...]


class LLMProposer:
    """Own every domain-facing concern while delegating only transport to a backend."""

    proposer_id = "llm"
    proposer_version = LLM_PROPOSER_VERSION

    def __init__(
        self,
        *,
        backend: ModelBackend,
        resolved_model: str,
        endpoint: str,
        service_tier: str,
        settings: JsonObject,
        limits: AstLimits,
        allowed_macros: frozenset[str],
        cache: ExactResponseCache | None = None,
    ) -> None:
        self.backend = backend
        self.resolved_model = resolved_model
        self.endpoint = endpoint
        self.service_tier = service_tier
        self.settings = settings
        self.limits = limits
        self.allowed_macros = allowed_macros
        self.cache = cache
        self.last_cache_hit = False

    def build_request(
        self,
        *,
        task: object,
        role: ProposalRole,
        batch_size: int,
        parent: CandidateSummary | None = None,
        feedback: ParentScoreFeedback | None = None,
    ) -> ModelRequest:
        from world_model_search.domain.types import PublicTask

        if not isinstance(task, PublicTask):
            raise TypeError("LLM proposer accepts only PublicTask")
        template, version, rendered = render_prompt(
            task=task,
            role=role,
            requested_batch_size=batch_size,
            parent=parent,
            feedback=feedback,
        )
        return ModelRequest(
            backend_id=self.backend.backend_id,
            provider_id=self.backend.provider_id,
            resolved_model=self.resolved_model,
            endpoint=self.endpoint,
            service_tier=self.service_tier,
            prompt_template=template,
            prompt_version=version,
            rendered_input=rendered,
            structured_schema_name=BATCH_SCHEMA_NAME,
            structured_schema_version=BATCH_SCHEMA_VERSION,
            structured_schema=candidate_batch_json_schema(role=role, batch_size=batch_size),
            role=role,
            requested_batch_size=batch_size,
            settings=self.settings,
        )

    def dispatch(self, request: ModelRequest, *, allow_cache: bool = True) -> ModelResponse:
        self.last_cache_hit = False
        if allow_cache and self.cache is not None:
            cached = self.cache.get(request)
            if cached is not None:
                self.last_cache_hit = True
                return cached
        response = self.backend.dispatch(request)
        if self.cache is not None:
            self.cache.put(request, response)
        return response

    def build_experience_request(
        self,
        *,
        task: object,
        role: ProposalRole,
        batch_size: int,
        parent: CandidateSummary,
        feedback: ParentScoreFeedback,
        retrieval: ExperienceRetrievalRecord,
    ) -> ModelRequest:
        """Build a v2 iterative request bound to one selected archive cell and snapshot."""

        from world_model_search.domain.types import PublicTask

        if not isinstance(task, PublicTask):
            raise TypeError("LLM proposer accepts only PublicTask")
        _template, _version, base = render_prompt(
            task=task,
            role=role,
            requested_batch_size=batch_size,
            parent=parent,
            feedback=feedback,
        )
        rendered = inject_experience_memory(base_prompt=base, retrieval=retrieval)
        return ModelRequest(
            backend_id=self.backend.backend_id,
            provider_id=self.backend.provider_id,
            resolved_model=self.resolved_model,
            endpoint=self.endpoint,
            service_tier=self.service_tier,
            prompt_template="iterative-experience-memory",
            prompt_version=EXPERIENCE_SEARCH_PROMPT_VERSION,
            rendered_input=rendered,
            structured_schema_name=BATCH_SCHEMA_NAME,
            structured_schema_version=BATCH_SCHEMA_VERSION,
            structured_schema=candidate_batch_json_schema(role=role, batch_size=batch_size),
            role=role,
            requested_batch_size=batch_size,
            settings=self.settings,
        )

    def parse_response(self, request: ModelRequest, response: ModelResponse) -> LLMParsedResponse:
        if response.request_hash != request.request_hash:
            raise ValueError("model response does not match exact request identity")
        batch = parse_candidate_batch(
            response.raw_text,
            expected_role=request.role,
            requested_batch_size=request.requested_batch_size,
            limits=self.limits,
            allowed_macros=self.allowed_macros,
        )
        operator_id = "llm-direct-v1" if request.prompt_template == "direct" else "llm-revision-v1"
        proposals = tuple(
            self._proposal(item, request=request, operator_id=operator_id)
            for item in batch.items
            if item.accepted
        )
        return LLMParsedResponse(batch, proposals)

    @staticmethod
    def _proposal(item: BatchItem, *, request: ModelRequest, operator_id: str) -> LLMProposal:
        if item.source_ast is None or item.canonical_ast is None or item.submitted_document is None:
            raise ValueError("cannot convert a rejected batch item to a proposal")
        return LLMProposal(
            ordinal=item.ordinal,
            role=request.role,
            source_ast=item.source_ast,
            canonical_ast=item.canonical_ast,
            request_hash=request.request_hash,
            submitted_document=item.submitted_document,
            operator_id=operator_id,
        )

    def propose(self, context: ProposalContext, budget: ProposalBudget) -> Sequence[LLMProposal]:
        """Shared protocol entry point for stateless direct sampling contract tests."""

        if context.parents or context.feedback:
            raise ValueError("iterative LLM proposals require typed parent score feedback")
        role = ProposalRole(budget.operator_id or ProposalRole.EXPLOIT.value)
        request = self.build_request(
            task=context.task,
            role=role,
            batch_size=budget.max_candidates,
        )
        return self.parse_response(request, self.dispatch(request)).proposals
