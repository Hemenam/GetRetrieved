import threading
from uuid import uuid4

from hrlearnium import evaluation
from hrlearnium.config import Settings
from hrlearnium.ingestion import IngestionError, parse_docx
from hrlearnium.models import ModelGateway, ModelUnavailable
from hrlearnium.policy import (
    conversational_reply,
    is_bare_followup,
    is_followup,
    non_answer_reply,
    policy_check,
)
from hrlearnium.reranking import CrossEncoderReranker, Reranker, rerank
from hrlearnium.retrieval import Candidate, filter_relevance, retrieve
from hrlearnium.schemas import Principal, QueryRequest, QueryResponse, Selection, render_answer
from hrlearnium.storage import Storage, StorageConflict


class ConversationNotFound(Exception):
    pass


class ServiceBusy(Exception):
    pass


class CourseService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        gateway: ModelGateway,
        reranker: Reranker | None = None,
    ):
        self.settings, self.storage, self.gateway = settings, storage, gateway
        self.reranker = (
            reranker or CrossEncoderReranker(settings)
            if settings.retrieval_mode == "hybrid_rerank"
            else None
        )
        # Bounded model work per process; shared SQL limits protect users/tenants across workers.
        self._model_slots = threading.BoundedSemaphore(2)

    def ingest(
        self,
        principal: Principal,
        course: str,
        filename: str,
        data: bytes,
        document_id: str | None = None,
    ):
        current = (
            self.storage.get_document(principal.tenant_id, course, document_id)
            if document_id
            else None
        )
        if document_id and current is None:
            raise LookupError("Document not found")
        parsed = parse_docx(data, filename)
        if len(parsed.full_text) > self.settings.max_document_characters:
            raise IngestionError("Document text exceeds the configured limit")
        if not parsed.passages:
            raise IngestionError("Document contains no usable course passages")
        vectors, model = None, None
        if self.settings.model_backend != "literal" and self.settings.retrieval_mode in {
            "hybrid",
            "hybrid_rerank",
        }:
            if not self._model_slots.acquire(blocking=False):
                raise ServiceBusy()
            try:
                model = self.gateway.identity()
                texts = [passage.chapter_title + "\n" + passage.text for passage in parsed.passages]
                vectors = self.gateway.embed(texts)
                if self.gateway.identity() != model:
                    raise ModelUnavailable("Embedding model changed during ingestion")
            finally:
                self._model_slots.release()
        return self.storage.save_document(
            principal.tenant_id,
            course,
            filename,
            parsed,
            data,
            vectors,
            model,
            max_course_passages=self.settings.max_course_passages,
            document_id=document_id,
            expected_version=current.version if current else None,
        )

    def query(
        self, principal: Principal, course: str, request: QueryRequest, request_id: str
    ) -> QueryResponse:
        with evaluation.collect(self.settings, request.include_evaluation, self.reranker) as trace:
            result = self._query(principal, course, request, request_id)
            if trace is not None:
                # Loading the reranker may resolve a pinned Hub revision during this query.
                if self.reranker:
                    identity = self.reranker.identity()
                    trace["reranker_resolved_revision"] = identity.get("resolved_revision")
                    trace["configuration"]["reranker"] = {
                        key: value for key, value in identity.items() if key != "resolved_revision"
                    }
                result.evaluation = trace
            return result

    def _query(
        self, principal: Principal, course: str, request: QueryRequest, request_id: str
    ) -> QueryResponse:
        tenant = principal.tenant_id
        conversation_id = str(request.conversation_id) if request.conversation_id else str(uuid4())
        previous = []
        if request.conversation_id:
            previous = self.storage.conversation_questions(
                tenant,
                course,
                principal.sub,
                conversation_id,
                self.settings.conversation_ttl_seconds,
            )
            if previous is None:
                raise ConversationNotFound()
        mode = (
            "lexical" if self.settings.model_backend == "literal" else self.settings.retrieval_mode
        )
        social_reply = conversational_reply(request.question)
        if social_reply is not None:
            reason, answer = social_reply
            # Social turns share the scoped conversation but never become evidence history.
            self.storage.save_conversation(
                tenant,
                course,
                principal.sub,
                conversation_id,
                previous,
                self.settings.conversation_ttl_seconds,
            )
            return QueryResponse(
                request_id=request_id,
                status="conversation",
                answer=answer,
                excerpts=[],
                conversation_id=conversation_id,
                reason_code=reason,
                retrieval_mode=mode,
                response_mode=request.response_mode,
            )
        if request.response_mode == "explained" and self.settings.model_backend == "literal":
            raise ModelUnavailable("Explanation requires a configured LLM backend")
        decision = policy_check(request.question)
        candidate_ids = set()
        if decision is None:
            passages, models = self.storage.search_passages(tenant, course)
            evaluation.corpus(passages)
            if not passages:
                decision = Selection(
                    status="refused", passage_ids=[], reason_code="insufficient_evidence"
                )
            elif is_bare_followup(request.question) and not previous:
                decision = Selection(
                    status="clarification", passage_ids=[], reason_code="ambiguous"
                )
            else:
                if not self._model_slots.acquire(blocking=False):
                    raise ServiceBusy()
                try:
                    search_question = request.question
                    if previous and is_followup(request.question):
                        search_question = previous[-1] + "\n" + request.question
                    if mode == "full_context":
                        if (
                            sum(len(p.text) + len(p.chapter_title) for p in passages)
                            > self.settings.max_context_characters
                        ):
                            raise ModelUnavailable(
                                "Course exceeds the full-context limit; use hybrid retrieval"
                            )
                        candidates = [
                            Candidate(passage=p, lexical_score=0, semantic_score=None, rank_score=0)
                            for p in passages
                        ]
                    else:
                        vector = None
                        if mode in {"hybrid", "hybrid_rerank"}:
                            identity = self.gateway.identity()
                            if models != {identity} or any(p.embedding is None for p in passages):
                                raise ModelUnavailable(
                                    "Course embeddings are missing or stale; reindex the course"
                                )
                            with evaluation.stage("embedding"):
                                vector = self.gateway.embed([search_question])[0]
                            if self.gateway.identity() != identity:
                                raise ModelUnavailable("Embedding model changed during retrieval")
                        try:
                            with evaluation.stage("retrieval"):
                                candidates = retrieve(
                                    search_question,
                                    passages,
                                    query_embedding=vector,
                                    # Apply floors BEFORE top-k; weak high-ranked matches must
                                    # not hide eligible evidence further down the fused list.
                                    limit=len(passages),
                                )
                        except ValueError as error:
                            raise ModelUnavailable(
                                "Invalid retrieval embeddings; reindex the course"
                            ) from error
                        evaluation.candidates("retrieved_candidates", candidates)
                        if mode in {"hybrid", "hybrid_rerank"}:
                            before_count = len(candidates)
                            with evaluation.stage("relevance_filter"):
                                candidates = filter_relevance(
                                    candidates,
                                    min_cosine=self.settings.retrieval_min_cosine,
                                    min_bm25=self.settings.retrieval_min_bm25,
                                )
                            evaluation.relevance_filter(before_count, len(candidates))
                        candidates = candidates[
                            : (
                                self.settings.rerank_pool_limit
                                if mode == "hybrid_rerank"
                                else self.settings.candidate_limit
                            )
                        ]
                        evaluation.candidates("eligible_candidates", candidates)
                        if mode == "hybrid_rerank":
                            with evaluation.stage("reranking"):
                                candidates = rerank(
                                    search_question,
                                    candidates,
                                    self.reranker,
                                    limit=self.settings.candidate_limit,
                                    min_score=self.settings.reranker_min_score,
                                )
                        context_size, bounded = 0, []
                        for candidate in candidates:
                            cost = len(candidate.passage.text) + len(
                                candidate.passage.chapter_title
                            )
                            if context_size + cost > self.settings.max_context_characters:
                                continue
                            bounded.append(candidate)
                            context_size += cost
                        candidates = bounded
                    candidate_ids = {candidate.passage.id for candidate in candidates}
                    evaluation.candidates("selector_candidates", candidates)
                    with evaluation.stage("selection"):
                        decision = (
                            self.gateway.select(request.question, previous, candidates)
                            if candidates
                            else Selection(
                                status="refused",
                                passage_ids=[],
                                reason_code="insufficient_evidence",
                            )
                        )
                finally:
                    self._model_slots.release()
        # Evidence IDs must be retrieved candidates, authorized current records and exact spans.
        if (
            len(decision.passage_ids) != len(set(decision.passage_ids))
            or len(decision.passage_ids) > self.settings.max_excerpts
            or not set(decision.passage_ids).issubset(candidate_ids)
            or (
                decision.status == "answered"
                and (not decision.passage_ids or decision.reason_code != "supported")
            )
            or (decision.status != "answered" and decision.passage_ids)
            or (decision.status == "clarification" and decision.reason_code != "ambiguous")
            or (decision.status == "refused" and decision.reason_code in {"supported", "ambiguous"})
        ):
            raise ModelUnavailable("Invalid evidence selection")
        excerpts = self.storage.get_excerpts(
            tenant, course, decision.passage_ids, current_only=True
        )
        if len(excerpts) != len(decision.passage_ids):
            raise StorageConflict("Course content changed during the query; retry")
        # Sorting by source metadata keeps multi-passage quotations in source order.
        excerpts.sort(
            key=lambda e: (e.citation.document_id, e.citation.version, e.citation.source_start)
        )
        explanation = None
        if decision.status == "answered" and request.response_mode == "explained":
            if not self._model_slots.acquire(blocking=False):
                raise ServiceBusy()
            try:
                with evaluation.stage("explanation"):
                    explanation = self.gateway.explain(request.question, previous, excerpts)
                allowed = {excerpt.id for excerpt in excerpts}
                if any(
                    len(statement.citation_ids) != len(set(statement.citation_ids))
                    or not set(statement.citation_ids).issubset(allowed)
                    for statement in explanation.statements
                ):
                    raise ModelUnavailable("Explanation references invalid evidence")
                with evaluation.stage("verification"):
                    supported = self.gateway.verify_explanation(
                        request.question, previous, excerpts, explanation
                    )
                if type(supported) is not bool:
                    raise ModelUnavailable("Invalid explanation verification")
                if not supported:
                    explanation, excerpts = None, []
                    decision = Selection(
                        status="refused", passage_ids=[], reason_code="insufficient_evidence"
                    )
            finally:
                self._model_slots.release()
            # Generation and checking can take time. Never return evidence retired or replaced
            # during these calls, even though old revision citations remain readable separately.
            current = self.storage.get_excerpts(
                tenant, course, decision.passage_ids, current_only=True
            )
            if {e.id for e in current} != {e.id for e in excerpts}:
                raise StorageConflict("Course content changed during the query; retry")
        answer = render_answer(excerpts, explanation)
        if decision.status != "answered":
            answer = non_answer_reply(request.question, decision.status, decision.reason_code)
        # Rejected questions never become context for subsequent questions.
        questions = previous + [request.question] if decision.status == "answered" else previous
        self.storage.save_conversation(
            tenant,
            course,
            principal.sub,
            conversation_id,
            questions,
            self.settings.conversation_ttl_seconds,
        )
        return QueryResponse(
            request_id=request_id,
            status=decision.status,
            answer=answer,
            excerpts=excerpts,
            conversation_id=conversation_id,
            reason_code=decision.reason_code,
            retrieval_mode=mode,
            response_mode=request.response_mode,
            explanation=explanation,
        )
