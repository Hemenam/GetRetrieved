import threading
from uuid import uuid4

from hrlearnium import evaluation
from hrlearnium.access import require_access
from hrlearnium.answer_contracts import unsupported_literal_quotes
from hrlearnium.core_config import CoreSettings
from hrlearnium.course_metadata import metadata_request, metadata_response
from hrlearnium.grounding import (
    brief_selection,
    explanation_covers_question,
    filter_brief_rules,
    prune_incidental_statements,
    review_supports_statements,
    search_variants,
    selection_covers_question,
    supplement_general_rule,
    validate_plan,
    wants_persian_search,
)
from hrlearnium.ingestion import IngestionError, parse_docx
from hrlearnium.language import (
    explanation_language_matches,
    language_scope,
    response_language,
    strip_language_directive,
)
from hrlearnium.models import ModelGateway, ModelUnavailable
from hrlearnium.policy import (
    conversational_reply,
    is_bare_followup,
    is_followup,
    non_answer_reply,
    policy_check,
)
from hrlearnium.request_planning import (
    canonical_route,
    compile_route,
    course_routing_context,
    original_request_plan,
    request_plan_error,
    search_plan,
    verified_request_plan,
)
from hrlearnium.request_units import compile_units, is_course_overview, request_units
from hrlearnium.reranking import CrossEncoderReranker, Reranker, rerank
from hrlearnium.retrieval import Candidate, filter_relevance, fuse_query_candidates, retrieve
from hrlearnium.schemas import (
    AnswerPart,
    CompleteSelection,
    Explanation,
    Principal,
    QueryRequest,
    QueryResponse,
    QuestionPlan,
    RequestPlan,
    RequestPlanVerification,
    RetrievalOption,
    RetrievalOptions,
    Selection,
    SpanAnswerReview,
    render_answer,
    render_parts,
)
from hrlearnium.source_rules import (
    attribute_source_scope,
    explicit_rule_conflict,
    source_status_preserved,
)
from hrlearnium.storage import Storage, StorageConflict


class ConversationNotFound(Exception):
    pass


class ServiceBusy(Exception):
    pass


class CourseService:
    def __init__(
        self,
        settings: CoreSettings,
        storage: Storage,
        gateway: ModelGateway,
        reranker: Reranker | None = None,
    ):
        self.settings, self.storage, self.gateway = settings, storage, gateway
        # Construction is cheap: runtime/model loading remains lazy, even with a UI override.
        self._optional_reranker = reranker or CrossEncoderReranker(settings)
        self.reranker = (
            self._optional_reranker if settings.retrieval_mode == "hybrid_rerank" else None
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
        require_access(principal, course, "content:write")
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

    def retrieval_options(self, principal: Principal, course: str) -> RetrievalOptions:
        """Authorized prerequisite checks, not an inference test or an indexing operation."""
        require_access(principal, course, "query")
        passages, models = self.storage.search_passages(principal.tenant_id, course)
        embedded = sum(p.embedding is not None for p in passages)
        common = []
        if self.settings.model_backend == "literal":
            common.append("Configure an LLM backend; the literal backend is diagnostic only.")
        elif self.settings.model_backend == "openai_compatible" and not all(
            (
                self.settings.api_base_url.strip(),
                self.settings.api_key.get_secret_value().strip(),
                self.settings.api_model.strip(),
            )
        ):
            common.append("Configure the server's model API URL, key and chat model.")
        if not passages:
            common.append("Import course content before testing retrieval.")
        full = list(common)
        if (
            sum(len(p.text) + len(p.chapter_title) for p in passages)
            > self.settings.max_context_characters
        ):
            full.append(
                "This course exceeds the full-context character limit. Use hybrid retrieval."
            )
        hybrid = list(common)
        identity = None
        if (
            self.settings.model_backend == "openai_compatible"
            and not self.settings.api_embedding_model.strip()
        ):
            hybrid.append("Set HR_API_EMBEDDING_MODEL on the server, then reindex this course.")
        elif self.settings.model_backend != "literal" and not common:
            try:
                identity = self.gateway.identity()
            except ModelUnavailable:
                hybrid.append(
                    "The embedding model configuration or inventory is unavailable. Check the server, then reindex."
                )
        if passages and embedded != len(passages):
            hybrid.append(
                f"Course embeddings are missing for {len(passages) - embedded} of {len(passages)} chunks. Reindex this course."
            )
        elif passages and identity is not None and models != {identity}:
            hybrid.append(
                "Course embeddings use a different model or revision. Reindex this course."
            )
        ranked = list(hybrid)
        if self.settings.rerank_pool_limit < self.settings.candidate_limit:
            ranked.append("HR_RERANK_POOL_LIMIT must be at least HR_CANDIDATE_LIMIT.")
        if not self.settings.reranker_model.strip():
            ranked.append("Configure HR_RERANKER_MODEL on the server.")
        ranker_status = (
            self._optional_reranker.readiness()
            if hasattr(self._optional_reranker, "readiness")
            else {"ready": True}
        )
        if not ranker_status.get("ready"):
            if ranker_status.get("missing_dependencies"):
                ranked.append(
                    "Install the optional hrlearnium-chatbot[rerank] runtime on the server."
                )
            else:
                ranked.append(
                    "Reranker model files are not cached locally. Download the configured model or configure a local model directory."
                )
        warnings = [
            "Prerequisites only; provider access and answer quality are verified by an actual query."
        ]
        ranker_warnings = warnings + [
            "Reranker inference is checked on the first ranked query; cached-file checks do not prove inference works."
        ]
        if ranker_status.get("download_required") and not self.settings.reranker_local_files_only:
            ranker_warnings.append(
                "The first ranked query may download model files because server downloads are enabled."
            )
        return RetrievalOptions(
            default_mode="lexical"
            if self.settings.model_backend == "literal"
            else self.settings.retrieval_mode,
            passage_count=len(passages),
            embedded_passage_count=embedded,
            modes=[
                RetrievalOption(
                    mode="full_context", available=not full, reasons=full, warnings=warnings
                ),
                RetrievalOption(
                    mode="hybrid", available=not hybrid, reasons=hybrid, warnings=warnings
                ),
                RetrievalOption(
                    mode="hybrid_rerank",
                    available=not ranked,
                    reasons=ranked,
                    warnings=ranker_warnings,
                ),
            ],
        )

    def query(
        self,
        principal: Principal,
        course: str,
        request: QueryRequest,
        request_id: str | None = None,
    ) -> QueryResponse:
        require_access(principal, course, "query")
        request_id = request_id or str(uuid4())
        mode = request.retrieval_mode or self.settings.retrieval_mode
        if self.settings.model_backend == "literal":
            if request.retrieval_mode is not None:
                raise ModelUnavailable("Retrieval overrides require an LLM backend")
            mode = "lexical"
        if mode == "hybrid_rerank" and (
            not self.settings.reranker_model.strip()
            or self.settings.rerank_pool_limit < self.settings.candidate_limit
        ):
            raise ModelUnavailable("Invalid reranking configuration")
        ranker = self._optional_reranker if mode == "hybrid_rerank" else None
        with (
            evaluation.collect(self.settings, request.include_evaluation, ranker) as trace,
            language_scope(request.question) as language,
        ):
            if trace is not None:
                trace["configuration"]["retrieval_mode"] = mode
                trace["target_language"] = language.language
            result = self._query(principal, course, request, request_id, mode, ranker)
            if trace is not None:
                # Loading the reranker may resolve a pinned Hub revision during this query.
                if ranker:
                    identity = ranker.identity()
                    trace["reranker_resolved_revision"] = identity.get("resolved_revision")
                    trace["configuration"]["reranker"] = {
                        key: value for key, value in identity.items() if key != "resolved_revision"
                    }
                result.evaluation = trace
            return result

    def course_outline(self, principal: Principal, course: str):
        require_access(principal, course, "query")
        return self.storage.course_outline(principal.tenant_id, course)

    def list_documents(self, principal: Principal, course: str):
        require_access(principal, course, "content:read")
        return self.storage.list_documents(principal.tenant_id, course)

    def get_excerpt(self, principal: Principal, course: str, excerpt_id: str):
        require_access(principal, course, "query")
        excerpts = self.storage.get_excerpts(principal.tenant_id, course, [excerpt_id])
        return excerpts[0] if excerpts else None

    def retire_document(self, principal: Principal, course: str, document_id: str):
        require_access(principal, course, "content:write")
        return self.storage.retire_document(principal.tenant_id, course, document_id)

    def close(self):
        """Close this component's model clients at worker shutdown, not after each request."""
        self.gateway.close()

    def _query(self, principal, course, request, request_id, mode, ranker) -> QueryResponse:
        """Plan once, route structural requests, isolate independent content questions."""
        previous = []
        conversation_id = str(request.conversation_id) if request.conversation_id else str(uuid4())
        if request.conversation_id:
            previous = self.storage.conversation_questions(
                principal.tenant_id,
                course,
                principal.sub,
                conversation_id,
                self.settings.conversation_ttl_seconds,
            )
            if previous is None:
                raise ConversationNotFound()
        policy = policy_check(request.question)
        if (
            conversational_reply(request.question) is not None
            or (
                policy is not None
                and (
                    policy.reason_code == "policy_override"
                    or len(request_units(request.question)) == 1
                )
            )
            or (is_bare_followup(request.question) and not previous)
        ):
            return self._query_single(principal, course, request, request_id, mode, ranker)
        operation = metadata_request(strip_language_directive(request.question))
        if operation:
            outline = self.storage.course_outline(principal.tenant_id, course)
            result = metadata_response(
                outline,
                request,
                request_id=request_id,
                conversation_id=conversation_id,
                mode=mode,
                operation=operation,
            )
            self.storage.save_conversation(
                principal.tenant_id,
                course,
                principal.sub,
                conversation_id,
                previous + [request.question] if result.status == "answered" else previous,
                self.settings.conversation_ttl_seconds,
            )
            return result
        if mode == "lexical":
            return self._query_single(principal, course, request, request_id, mode, ranker)
        passages, _ = self.storage.search_passages(principal.tenant_id, course)
        evaluation.corpus(passages)
        if not passages:
            return self._query_single(principal, course, request, request_id, mode, ranker)
        outline = self.storage.course_outline(principal.tenant_id, course)
        if callable(getattr(self.gateway, "route_units", None)):
            units = request_units(request.question)
            translate = (
                mode in {"hybrid", "hybrid_rerank"}
                and self.settings.translate_search_queries
                and wants_persian_search(request.question)
            )
            if len(units) > 8:
                plan = compile_units(units, None, response_language(request.question))
            else:
                if not self._model_slots.acquire(blocking=False):
                    raise ServiceBusy()
                try:
                    proposed = None
                    with evaluation.stage("unit_routing"):
                        try:
                            proposed = self.gateway.route_units(
                                units,
                                previous,
                                course_routing_context(outline),
                                translate=translate,
                            )
                            plan = compile_units(
                                units,
                                proposed,
                                response_language(request.question),
                                translate=translate,
                            )
                        except (ValueError, ModelUnavailable) as error:
                            if isinstance(
                                error, ModelUnavailable
                            ) and error.diagnostic_code not in {
                                "structured_response_invalid",
                                "invalid_question_plan",
                            }:
                                raise
                            plan = compile_units(units, None, response_language(request.question))
                    evaluation.unit_routing(units, proposed, plan)
                finally:
                    self._model_slots.release()
            evaluation.request_plan(plan)
        elif callable(getattr(self.gateway, "route_request", None)):
            plan, translate = self._prepare_route(request.question, previous, mode, outline)
        else:
            # Compatibility for existing in-process gateway integrations. The shipped
            # hosted and Ollama gateways both use the simplified v8 routing protocol.
            plan, translate = self._prepare_request_plan(request.question, previous, mode)
        if plan.needs_clarification:
            return self._plan_clarification(
                principal, course, request, conversation_id, previous, request_id, mode
            )
        parts, excerpts = [], {}
        for index, group in enumerate(plan.groups):
            question = group.original_text
            if conversational_reply(question) is not None:
                raise ModelUnavailable(
                    "Question plan contains a non-substantive part", code="invalid_question_plan"
                )
            part_request = request.model_copy(
                update={"question": question, "conversation_id": None}
            )
            try:
                with evaluation.collect(
                    self.settings, request.include_evaluation, ranker
                ) as part_trace:
                    if group.intent == "metadata":
                        result = metadata_response(
                            outline,
                            part_request,
                            request_id=request_id,
                            conversation_id=conversation_id,
                            mode=mode,
                            operation=(group.metadata_field, group.chapter_number),
                        )
                    else:
                        child_plan = search_plan(group)
                        # Translation is a best-effort search hint, never answer authority.
                        validate_plan(child_plan, translated=bool(child_plan.persian_query))
                        result = self._query_single(
                            principal,
                            course,
                            part_request,
                            request_id,
                            mode,
                            ranker,
                            prepared_plan=child_plan,
                            history_override=previous,
                            persist=False,
                            conversation_id_override=conversation_id,
                        )
            except ModelUnavailable as error:
                if len(plan.groups) == 1 or error.diagnostic_code not in {
                    "answer_verification_failed",
                    "invalid_answer_language",
                    "invalid_answer_review",
                    "structured_response_invalid",
                    "invalid_evidence_brief",
                    "answer_source_budget_exceeded",
                    "invalid_answer_citations",
                }:
                    raise
                # An independently failed draft must not erase supported siblings or
                # impersonate a valid course refusal. Never expose the failed draft.
                result = QueryResponse(
                    request_id=request_id,
                    status="error",
                    answer=non_answer_reply(question, "error", error.diagnostic_code),
                    excerpts=[],
                    conversation_id=conversation_id,
                    reason_code=error.diagnostic_code,
                    retrieval_mode=mode,
                    response_mode=request.response_mode,
                )
                if part_trace is not None:
                    part_trace["operational_error"] = error.diagnostic_code
            evaluation.part_result(index, question, part_trace, single=len(plan.groups) == 1)
            if len(plan.groups) == 1:
                if (
                    self.storage.course_outline(principal.tenant_id, course).snapshot
                    != outline.snapshot
                ):
                    raise StorageConflict("Course content changed during the query; retry")
                self.storage.save_conversation(
                    principal.tenant_id,
                    course,
                    principal.sub,
                    conversation_id,
                    previous + [question] if result.status == "answered" else previous,
                    self.settings.conversation_ttl_seconds,
                )
                return result
            for excerpt in result.excerpts:
                if excerpt.id in excerpts and excerpts[excerpt.id] != excerpt:
                    raise StorageConflict("Course evidence changed during the query")
                excerpts[excerpt.id] = excerpt
            parts.append(
                AnswerPart(
                    question=question,
                    status=result.status,
                    answer_kind=result.answer_kind,
                    answer=result.answer,
                    reason_code=result.reason_code,
                    excerpt_ids=[e.id for e in result.excerpts],
                    explanation=result.explanation,
                    metadata=result.metadata,
                )
            )
        if self.storage.course_outline(principal.tenant_id, course).snapshot != outline.snapshot:
            raise StorageConflict("Course content changed during the query; retry")
        current = self.storage.get_excerpts(
            principal.tenant_id, course, list(excerpts), current_only=True
        )
        if {e.id for e in current} != set(excerpts):
            raise StorageConflict("Course content changed during the query; retry")
        accepted = [part for part in parts if part.status == "answered"]
        for part in accepted:
            if part.answer_kind == "content" and part.explanation is not None:
                part.answer = render_answer(list(excerpts.values()), part.explanation)
        status = (
            "answered"
            if len(accepted) == len(parts)
            else "partial"
            if accepted
            else "error"
            if any(p.status == "error" for p in parts)
            else "clarification"
            if all(p.status == "clarification" for p in parts)
            else "refused"
        )
        # Only supported independent questions enter history, never rejected siblings.
        self.storage.save_conversation(
            principal.tenant_id,
            course,
            principal.sub,
            conversation_id,
            previous + [p.question for p in accepted],
            self.settings.conversation_ttl_seconds,
        )
        result = QueryResponse(
            request_id=request_id,
            status=status,
            answer=render_parts(parts, list(excerpts.values()), request.response_mode),
            excerpts=list(excerpts.values()),
            conversation_id=conversation_id,
            reason_code="all_parts_supported"
            if status == "answered"
            else "some_parts_unsupported"
            if accepted
            else "part_execution_failed"
            if status == "error"
            else "no_parts_supported",
            retrieval_mode=mode,
            response_mode=request.response_mode,
            answer_kind="multi",
            parts=parts,
        )
        return result

    def _prepare_route(self, question, previous, mode, outline):
        """Interpret only. No model-generated requirements or pre-retrieval abstention."""
        translate = (
            mode in {"hybrid", "hybrid_rerank"}
            and self.settings.translate_search_queries
            and wants_persian_search(question)
        )
        context = course_routing_context(outline)
        route, failure = None, None
        if not self._model_slots.acquire(blocking=False):
            raise ServiceBusy()
        try:
            for attempt in range(2):
                check = None
                try:
                    with evaluation.stage("routing_repair" if attempt else "routing"):
                        proposed = (
                            self.gateway.repair_route(
                                question,
                                previous,
                                context,
                                route,
                                failure,
                                translate=translate,
                            )
                            if attempt
                            else self.gateway.route_request(
                                question,
                                previous,
                                context,
                                translate=translate,
                            )
                        )
                    route = proposed
                    route = canonical_route(route, question)
                    plan = compile_route(route, response_language(question), translate=translate)
                    with evaluation.stage("route_verification"):
                        check = self.gateway.verify_route(question, previous, context, route)
                    if not verified_request_plan(check):
                        failure = "route_verification_rejected"
                    else:
                        evaluation.routing_attempt(attempt, route, None, check)
                        evaluation.request_plan(plan)
                        return plan, translate
                except ValueError:
                    failure = "route_alignment_or_operation_invalid"
                except ModelUnavailable as error:
                    if error.diagnostic_code not in {
                        "structured_response_invalid",
                        "invalid_question_plan",
                    }:
                        raise
                    failure = error.diagnostic_code
                evaluation.routing_attempt(attempt, route, failure, check)
            # Only the model's proposed execution plan is discarded. The student's full
            # request still goes through normal selection, scope and answer verification.
            plan = original_request_plan(question, response_language(question))
            evaluation.routing_fallback(failure)
            evaluation.request_plan(plan)
            return plan, False
        finally:
            self._model_slots.release()

    def _prepare_request_plan(self, question, previous, mode):
        """One repair at most. Internal plan failures never ask the learner to rephrase."""
        translate = (
            mode in {"hybrid", "hybrid_rerank"}
            and self.settings.translate_search_queries
            and wants_persian_search(question)
        )
        target = response_language(question)
        plan, failure = None, None
        recoverable = {"structured_response_invalid", "invalid_question_plan"}
        if not self._model_slots.acquire(blocking=False):
            raise ServiceBusy()
        try:
            for attempt in range(2):
                check = None
                try:
                    with evaluation.stage("planning_repair" if attempt else "planning"):
                        if attempt:
                            plan = self.gateway.repair_request_plan(
                                question,
                                previous,
                                plan,
                                failure,
                                translate=translate,
                                target_language=target,
                            )
                        else:
                            plan = self.gateway.plan_request(
                                question, previous, translate=translate, target_language=target
                            )
                    failure = request_plan_error(plan, question, target, translate=translate)
                    if isinstance(plan, RequestPlan):
                        evaluation.request_plan(plan)
                    if failure is None and plan.needs_clarification and not attempt:
                        failure = "proposed_clarification_needs_review"
                    if failure is None:
                        with evaluation.stage(
                            "plan_repair_verification" if attempt else "plan_verification"
                        ):
                            check = self.gateway.verify_request_plan(question, previous, plan)
                        if isinstance(check, RequestPlanVerification):
                            evaluation.request_plan_verification(check)
                        if not verified_request_plan(check):
                            failure = "plan_verification_rejected"
                except ModelUnavailable as error:
                    if error.diagnostic_code not in recoverable:
                        raise
                    failure = error.diagnostic_code
                evaluation.request_plan_attempt(attempt, plan, failure, check)
                if failure is None:
                    return plan, translate
                # Pass controlled diagnostics, never raw provider error text, to the repair.
                if isinstance(check, RequestPlanVerification):
                    failure += ":" + ",".join(
                        name for name, valid in check.model_dump().items() if not valid
                    )
            raise ModelUnavailable(
                "Request planning failed after one repair", code="invalid_question_plan"
            )
        finally:
            self._model_slots.release()

    def _plan_clarification(
        self, principal, course, request, conversation_id, previous, request_id, mode
    ):
        self.storage.save_conversation(
            principal.tenant_id,
            course,
            principal.sub,
            conversation_id,
            previous,
            self.settings.conversation_ttl_seconds,
        )
        return QueryResponse(
            request_id=request_id,
            status="clarification",
            answer=non_answer_reply(request.question, "clarification", "incomplete_question_plan"),
            excerpts=[],
            conversation_id=conversation_id,
            reason_code="incomplete_question_plan",
            retrieval_mode=mode,
            response_mode=request.response_mode,
        )

    def _query_single(
        self,
        principal: Principal,
        course: str,
        request: QueryRequest,
        request_id: str,
        mode: str,
        ranker: Reranker | None,
        *,
        prepared_plan: QuestionPlan | None = None,
        history_override: list[str] | None = None,
        persist: bool = True,
        conversation_id_override: str | None = None,
    ) -> QueryResponse:
        tenant = principal.tenant_id
        conversation_id = conversation_id_override or (
            str(request.conversation_id) if request.conversation_id else str(uuid4())
        )
        previous = list(history_override or [])
        if request.conversation_id and history_override is None:
            previous = self.storage.conversation_questions(
                tenant,
                course,
                principal.sub,
                conversation_id,
                self.settings.conversation_ttl_seconds,
            )
            if previous is None:
                raise ConversationNotFound()
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
        plan, complete_selection = None, None
        repair_candidates, evidence_brief = None, None
        answer_budget = (
            self.settings.overview_max_excerpts
            if (
                callable(getattr(self.gateway, "prepare_evidence", None))
                and is_course_overview(request.question)
            )
            else self.settings.max_excerpts
        )
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
                    identity = None
                    if mode in {"hybrid", "hybrid_rerank"}:
                        identity = self.gateway.identity()
                        if models != {identity} or any(p.embedding is None for p in passages):
                            raise ModelUnavailable(
                                "Course embeddings are missing or stale; reindex the course"
                            )
                    if mode == "full_context" and (
                        sum(len(p.text) + len(p.chapter_title) for p in passages)
                        > self.settings.max_context_characters
                    ):
                        raise ModelUnavailable(
                            "Course exceeds the full-context limit; use hybrid retrieval"
                        )
                    variants = [search_question]
                    if mode != "lexical":
                        translate = (
                            mode in {"hybrid", "hybrid_rerank"}
                            and self.settings.translate_search_queries
                            and wants_persian_search(request.question)
                        )
                        if prepared_plan is None:
                            with evaluation.stage("planning"):
                                plan = self.gateway.plan(
                                    request.question, previous, translate=translate
                                )
                                validate_plan(plan, translated=translate)
                        else:
                            plan = prepared_plan
                        variants = search_variants(plan, search_question)
                        evaluation.question_plan(plan, variants)
                    if mode == "full_context":
                        candidates = [
                            Candidate(passage=p, lexical_score=0, semantic_score=None, rank_score=0)
                            for p in passages
                        ]
                    else:
                        vectors = [None] * len(variants)
                        if mode in {"hybrid", "hybrid_rerank"}:
                            with evaluation.stage("embedding"):
                                vectors = self.gateway.embed(variants)
                            if len(vectors) != len(variants):
                                raise ModelUnavailable("Invalid query embedding count")
                            if self.gateway.identity() != identity:
                                raise ModelUnavailable("Embedding model changed during retrieval")
                        try:
                            with evaluation.stage("retrieval"):
                                candidates = fuse_query_candidates(
                                    [
                                        retrieve(
                                            query,
                                            passages,
                                            query_embedding=vector,
                                            limit=len(passages),
                                        )
                                        for query, vector in zip(variants, vectors, strict=True)
                                    ]
                                )
                        except ValueError as error:
                            raise ModelUnavailable(
                                "Invalid retrieval embeddings; reindex the course"
                            ) from error
                        evaluation.candidates("retrieved_candidates", candidates)
                        if mode == "hybrid":
                            before_count = len(candidates)
                            with evaluation.stage("relevance_filter"):
                                candidates = filter_relevance(
                                    candidates,
                                    min_cosine=self.settings.retrieval_min_cosine,
                                    min_bm25=self.settings.retrieval_min_bm25,
                                )
                            evaluation.relevance_filter(before_count, len(candidates))
                            if is_course_overview(request.question):
                                candidates = sorted(
                                    candidates,
                                    key=lambda c: (
                                        c.passage.section_kind != "introduction",
                                        c.passage.chapter_number or 0
                                        if c.passage.section_kind == "introduction"
                                        else 0,
                                        -c.rank_score,
                                    ),
                                )
                            if callable(getattr(self.gateway, "review_grounded_answer", None)):
                                # Bounded recovery context from the SAME positive, gated
                                # ranking, before top-k truncation. No new search/model call.
                                repair_candidates, size = [], 0
                                for item in candidates[: self.settings.rerank_pool_limit]:
                                    cost = len(item.passage.text) + len(item.passage.chapter_title)
                                    if size + cost <= self.settings.max_context_characters:
                                        repair_candidates.append(item)
                                        size += cost
                                evaluation.candidates(
                                    "answer_recovery_candidates", repair_candidates
                                )
                        # Reranked mode keeps positive ranked matches until cross-encoder
                        # scoring. An uncalibrated embedding floor must not hide evidence
                        # from the reranker. Pool/context limits still bound the work.
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
                                    plan.persian_query or plan.search_query,
                                    candidates,
                                    ranker,
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
                    if (
                        callable(getattr(self.gateway, "prepare_evidence", None))
                        and mode != "lexical"
                    ):
                        candidates = (
                            repair_candidates if repair_candidates is not None else candidates
                        )
                        candidate_ids = {c.passage.id for c in candidates}
                    evaluation.candidates("selector_candidates", candidates)
                    with evaluation.stage("selection"):
                        if not candidates:
                            decision = Selection(
                                status="refused",
                                passage_ids=[],
                                reason_code="insufficient_evidence",
                            )
                        elif plan is None:
                            decision = self.gateway.select(request.question, previous, candidates)
                        elif callable(getattr(self.gateway, "prepare_evidence", None)):
                            evidence_brief = self.gateway.prepare_evidence(
                                request.question,
                                previous,
                                candidates,
                                plan,
                                max_sources=answer_budget,
                            )
                            evidence_brief = filter_brief_rules(evidence_brief, candidates)
                            evidence_brief = supplement_general_rule(
                                evidence_brief,
                                candidates,
                                request.question,
                                answer_budget,
                            )
                            evaluation.evidence_brief(evidence_brief)
                            complete_selection = brief_selection(
                                evidence_brief,
                                candidates,
                                max_excerpts=answer_budget,
                                question=request.question,
                            )
                            decision = selection_covers_question(
                                complete_selection,
                                plan,
                                candidate_ids=candidate_ids,
                                max_excerpts=answer_budget,
                            )
                            evaluation.requirement_selection(complete_selection, decision.status)
                        else:
                            complete_selection = self.gateway.select_complete(
                                request.question, previous, candidates, plan
                            )
                            decision = selection_covers_question(
                                complete_selection,
                                plan,
                                candidate_ids=candidate_ids,
                                max_excerpts=answer_budget,
                            )
                            evaluation.requirement_selection(complete_selection, decision.status)
                finally:
                    self._model_slots.release()
        # Evidence IDs must be retrieved candidates, authorized current records and exact spans.
        if (
            len(decision.passage_ids) != len(set(decision.passage_ids))
            or len(decision.passage_ids) > answer_budget
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
                explanation, excerpts, complete_selection = self._verified_answer(
                    principal,
                    course,
                    request,
                    previous,
                    excerpts,
                    plan,
                    complete_selection,
                    candidates,
                    repair_candidates=repair_candidates,
                    evidence_brief=evidence_brief,
                    max_sources=answer_budget,
                )
                decision = complete_selection.selection
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
        if persist:
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

    def _verified_answer(
        self,
        principal,
        course,
        request,
        previous,
        excerpts,
        plan,
        selection,
        candidates,
        *,
        repair_candidates=None,
        evidence_brief=None,
        max_sources=None,
    ):
        """One evidence-bound repair. Never relabel a bad draft as absent course evidence."""
        target = response_language(request.question)
        draft, verification, feedback = None, None, None
        language_repairs = 0
        local_issues = []
        answer_budget = max_sources or self.settings.max_excerpts
        span_review = getattr(self.gateway, "review_grounded_answer", None)
        context = None
        recovery_pool = repair_candidates if repair_candidates is not None else candidates
        if callable(span_review):
            candidates = repair_candidates if repair_candidates is not None else candidates
            if evidence_brief is not None:
                # The pre-draft brief already examined the broad eligible pool.
                # Keep repair inside its approved source set and global budget.
                candidates = [c for c in candidates if c.passage.id in {e.id for e in excerpts}]
            ids = [c.passage.id for c in candidates]
            context = self.storage.get_excerpts(principal.tenant_id, course, ids, current_only=True)
            if {e.id for e in context} != set(ids):
                raise StorageConflict("Course evidence changed during answer review")
        for attempt in range(2):
            pool = excerpts
            if attempt:
                pruned = (
                    prune_incidental_statements(draft, feedback, excerpts, plan)
                    if evidence_brief is not None
                    else None
                )
                if pruned is None and not callable(
                    getattr(self.gateway, "repair_explanation", None)
                ):
                    break
                # Same authorized, relevance-gated candidate pool; not new retrieval,
                # external facts, another tenant/course or an unbounded source expansion.
                if evidence_brief is not None and pruned is None:
                    with evaluation.stage("answer_evidence_repair"):
                        evidence_brief = self.gateway.prepare_evidence(
                            request.question,
                            previous,
                            recovery_pool,
                            plan,
                            max_sources=answer_budget,
                            feedback={
                                "draft": draft.model_dump(),
                                "review": feedback.model_dump(),
                                "local_contract_failures": local_issues,
                            },
                        )
                        evidence_brief = filter_brief_rules(evidence_brief, recovery_pool)
                        evidence_brief = supplement_general_rule(
                            evidence_brief, recovery_pool, request.question, answer_budget
                        )
                        evaluation.evidence_brief(evidence_brief)
                        selection = brief_selection(
                            evidence_brief,
                            recovery_pool,
                            max_excerpts=answer_budget,
                            question=request.question,
                        )
                        if selection.selection.status != "answered":
                            raise ModelUnavailable(
                                "Repair could not establish support",
                                code="answer_verification_failed",
                            )
                        ids = selection.selection.passage_ids
                        context = self.storage.get_excerpts(
                            principal.tenant_id, course, ids, current_only=True
                        )
                        candidates = [c for c in recovery_pool if c.passage.id in ids]
                ids = [c.passage.id for c in candidates]
                pool = (
                    context
                    if context is not None
                    else self.storage.get_excerpts(
                        principal.tenant_id, course, ids, current_only=True
                    )
                )
                if {e.id for e in pool} != set(ids):
                    raise StorageConflict("Course evidence changed during answer repair")
                with evaluation.stage("answer_repair"):
                    draft = (
                        pruned
                        if pruned is not None
                        else (
                            self.gateway.explain_from_brief(
                                request.question, previous, pool, plan, evidence_brief
                            )
                            if evidence_brief is not None
                            else self.gateway.repair_explanation(
                                request.question,
                                previous,
                                pool,
                                draft,
                                plan,
                                feedback or verification,
                            )
                        )
                    )
            else:
                with evaluation.stage("explanation"):
                    draft = (
                        self.gateway.explain_from_brief(
                            request.question, previous, excerpts, plan, evidence_brief
                        )
                        if evidence_brief is not None
                        else self.gateway.explain_complete(
                            request.question, previous, excerpts, plan, evidence_map=selection
                        )
                    )
            allowed = {e.id for e in pool}
            if not isinstance(draft, Explanation) or any(
                len(s.citation_ids) != len(set(s.citation_ids))
                or not set(s.citation_ids) <= allowed
                for s in draft.statements
            ):
                raise ModelUnavailable(
                    "Explanation references invalid evidence", code="invalid_answer_citations"
                )
            used = {i for s in draft.statements for i in s.citation_ids}
            if not used or len(used) > answer_budget:
                raise ModelUnavailable(
                    "Explanation exceeds the source budget", code="answer_source_budget_exceeded"
                )
            if attempt:
                excerpts = sorted(
                    [e for e in pool if e.id in used],
                    key=lambda e: (
                        e.citation.document_id,
                        e.citation.version,
                        e.citation.source_start,
                    ),
                )
                selection = CompleteSelection(
                    selection=Selection(
                        status="answered",
                        reason_code="supported",
                        passage_ids=[e.id for e in excerpts],
                    ),
                    coverage=[
                        {"requirement_index": i, "passage_ids": [e.id for e in excerpts]}
                        for i in range(len(plan.requirements))
                    ],
                )
            before_language_repair = draft
            if evidence_brief is not None:
                draft = attribute_source_scope(draft, excerpts, target)
                before_language_repair = draft
            initial_language_valid = explanation_language_matches(draft, excerpts, target)
            if not initial_language_valid:
                if language_repairs:
                    raise ModelUnavailable(
                        "Answer language repair exhausted", code="invalid_answer_language"
                    )
                language_repairs += 1
                original_citations = [s.citation_ids for s in draft.statements]
                with evaluation.stage("language_repair"):
                    draft = self.gateway.repair_language(
                        request.question, previous, excerpts, draft
                    )
                if (
                    not isinstance(draft, Explanation)
                    or [s.citation_ids for s in draft.statements] != original_citations
                    or not explanation_language_matches(draft, excerpts, target)
                ):
                    raise ModelUnavailable(
                        "Answer language repair failed", code="invalid_answer_language"
                    )
            evaluation.language_result(target, initial_language_valid)
            with evaluation.stage("answer_reverification" if attempt else "verification"):
                repair_context = (
                    {"language_repair_of": before_language_repair}
                    if not initial_language_valid
                    else {}
                )
                review_method = getattr(self.gateway, "review_answer", None)
                if callable(span_review):
                    review_method = span_review
                    repair_context["context_excerpts"] = context
                review = (
                    review_method if callable(review_method) else self.gateway.verify_complete
                )(
                    request.question,
                    previous,
                    excerpts,
                    draft,
                    plan,
                    evidence_map=selection,
                    **repair_context,
                )
                if callable(review_method):
                    if callable(span_review) and not isinstance(review, SpanAnswerReview):
                        raise ModelUnavailable("Invalid span answer review")
                    supported_spans = review_supports_statements(review, draft, excerpts)
                    verification, feedback = review.verification, review
                else:
                    supported_spans = True
                    verification = feedback = review
                supported = supported_spans and explanation_covers_question(
                    verification, plan, draft, selection
                )
                if evidence_brief is not None:
                    local_issues = []
                    for statement in draft.statements:
                        own = [e for e in excerpts if e.id in statement.citation_ids]
                        if not source_status_preserved(statement.text, own):
                            local_issues.append("example_or_exercise_requires_explicit_attribution")
                            supported = False
                        if any(explicit_rule_conflict(statement.text, e.text) for e in own):
                            local_issues.append("explicit_source_instruction_polarity_reversed")
                            supported = False
                        if unsupported_literal_quotes(statement.text, own):
                            local_issues.append("attributed_quote_missing_from_own_source")
                            supported = False
            evaluation.requirement_verification(verification, supported)
            evaluation.answer_attempt(
                attempt,
                draft,
                verification,
                supported,
                review=feedback,
                local_failures=local_issues,
            )
            if supported:
                return draft, excerpts, selection
            if not initial_language_valid:
                raise ModelUnavailable(
                    "Language repair failed evidence verification", code="invalid_answer_language"
                )
        raise ModelUnavailable(
            "Unable to verify an answer after bounded repair", code="answer_verification_failed"
        )
