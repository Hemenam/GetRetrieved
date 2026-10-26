import hashlib
import json
import math
import re
import time
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from hrlearnium.core_config import CoreSettings
from hrlearnium.detailed_review import DetailedSpanReview, normalize_detailed_review
from hrlearnium.evaluation import (
    ERROR_CODES,
    provider_attempt,
    provider_usage,
    review_contract_check,
    stage,
)
from hrlearnium.fact_contracts import (
    FactExplanationBase,
    bind_writer_facts,
    fact_explanation_model,
    normalize_precise_brief,
    precise_brief_model,
)
from hrlearnium.language import response_context
from hrlearnium.policy import (
    ANSWER_REPAIR_INSTRUCTIONS,
    ANSWER_REVIEW_INSTRUCTIONS,
    BRIEF_WRITING_INSTRUCTIONS,
    COMPLETE_SELECTION_INSTRUCTIONS,
    COMPLETE_VERIFICATION_INSTRUCTIONS,
    DETAILED_SPAN_REVIEW_INSTRUCTIONS,
    EVIDENCE_BRIEF_INSTRUCTIONS,
    EXPLANATION_INSTRUCTIONS,
    LANGUAGE_REPAIR_INSTRUCTIONS,
    PLAN_VERIFICATION_INSTRUCTIONS,
    PLANNING_INSTRUCTIONS,
    REQUEST_PLAN_REPAIR_INSTRUCTIONS,
    REQUEST_PLAN_VERIFICATION_INSTRUCTIONS,
    REQUEST_PLANNING_INSTRUCTIONS,
    ROUTE_VERIFICATION_INSTRUCTIONS,
    ROUTING_INSTRUCTIONS,
    SELECTOR_INSTRUCTIONS,
    UNIT_ROUTING_INSTRUCTIONS,
    VERIFICATION_INSTRUCTIONS,
)
from hrlearnium.retrieval import Candidate
from hrlearnium.schemas import (
    AnswerReview,
    CompleteSelection,
    CompleteVerification,
    EvidenceBrief,
    EvidenceDecisionEnvelopeBase,
    Excerpt,
    Explanation,
    ExplanationVerification,
    PlanVerification,
    QuestionPlan,
    RequestPlan,
    RequestPlanVerification,
    RoutePlan,
    Selection,
    SpanAnswerReview,
    TranslatedQuestionPlan,
    UnitRoutes,
    UntranslatedQuestionPlan,
)
from hrlearnium.selection_contract import decision_model
from hrlearnium.source_spans import source_spans
from hrlearnium.text import normalize_persian


class ModelUnavailable(Exception):
    """Infrastructure/protocol error, distinct from a valid evidence-based refusal."""

    def __init__(self, message: str, *, code="model_unavailable", provider_status=None):
        super().__init__(message)
        self.diagnostic_code = (
            code if isinstance(code, str) and code in ERROR_CODES else "model_unavailable"
        )
        self.provider_status = (
            provider_status
            if type(provider_status) is int and 400 <= provider_status <= 599
            else None
        )


class ModelGateway(Protocol):
    def review_grounded_answer(
        self,
        question,
        previous,
        excerpts,
        explanation,
        plan,
        *,
        evidence_map=None,
        language_repair_of=None,
        context_excerpts=(),
    ) -> SpanAnswerReview: ...
    def route_request(self, question, history, context, *, translate) -> RoutePlan: ...
    def repair_route(
        self, question, history, context, proposed, failure, *, translate
    ) -> RoutePlan: ...
    def verify_route(self, question, history, context, route) -> RequestPlanVerification: ...
    def repair_explanation(
        self, question, history, excerpts, draft, plan, verification
    ) -> Explanation: ...
    def review_answer(
        self,
        question,
        previous,
        excerpts,
        explanation,
        plan,
        *,
        evidence_map=None,
        language_repair_of=None,
    ) -> AnswerReview: ...
    def plan_request(
        self, question: str, previous_questions: list[str], *, translate: bool, target_language: str
    ) -> RequestPlan: ...
    def repair_request_plan(
        self, question, previous_questions, plan, failure, *, translate, target_language
    ) -> RequestPlan: ...
    def verify_request_plan(
        self, question: str, previous_questions: list[str], plan: RequestPlan
    ) -> RequestPlanVerification: ...
    def repair_language(
        self, question, previous_questions, excerpts, explanation
    ) -> Explanation: ...
    def verify_plan(
        self, question: str, previous_questions: list[str], plan: QuestionPlan
    ) -> PlanVerification: ...
    def plan(
        self, question: str, previous_questions: list[str], *, translate: bool
    ) -> QuestionPlan: ...
    def select_complete(
        self,
        question: str,
        previous_questions: list[str],
        candidates: list[Candidate],
        plan: QuestionPlan,
    ) -> CompleteSelection: ...
    def explain_complete(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        plan: QuestionPlan,
        *,
        evidence_map: CompleteSelection | None = None,
    ) -> Explanation: ...
    def verify_complete(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        explanation: Explanation,
        plan: QuestionPlan,
        *,
        evidence_map: CompleteSelection | None = None,
        language_repair_of: Explanation | None = None,
    ) -> CompleteVerification: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def select(
        self, question: str, previous_questions: list[str], candidates: list[Candidate]
    ) -> Selection: ...
    def explain(
        self, question: str, previous_questions: list[str], excerpts: list[Excerpt]
    ) -> Explanation: ...
    def verify_explanation(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        explanation: Explanation,
    ) -> bool: ...
    def identity(self) -> str | None: ...
    def readiness(self) -> dict: ...
    def close(self) -> None: ...


StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


def _strict_schema(value):
    """OpenAI strict schemas require every object property, including fields with defaults."""
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _strict_schema(item) for key, item in value.items() if key != "default"}
    if result.get("type") == "object":
        result["required"] = list(result.get("properties", {}))
        result["additionalProperties"] = False
    return result


def _normalized_vectors(values, count: int, dimensions: int | None):
    if not isinstance(values, list) or len(values) != count:
        raise ModelUnavailable("Invalid embedding count")
    vectors = []
    for vector in values:
        try:
            if (
                not isinstance(vector, list)
                or not 1 <= len(vector) <= 16384
                or any(
                    type(value) not in (float, int) or not math.isfinite(value) for value in vector
                )
            ):
                raise ValueError("Invalid embedding vector")
            scale = max(abs(float(value)) for value in vector)
            if not scale:
                raise ValueError("Zero embedding vector")
            scaled = [float(value) / scale for value in vector]
            norm = math.sqrt(sum(value * value for value in scaled))
        except (ValueError, OverflowError, TypeError):
            raise ModelUnavailable("Invalid embedding vector") from None
        if dimensions is not None and dimensions != len(vector):
            raise ModelUnavailable("Embedding dimensions changed")
        dimensions = len(vector)
        vectors.append([value / norm for value in scaled])
    return vectors, dimensions


def _content_limit(model: type[BaseModel]) -> int:
    if model in {RequestPlan, RoutePlan, UnitRoutes}:
        return 32000
    return (
        24000
        if model in {Explanation, AnswerReview, SpanAnswerReview, DetailedSpanReview, EvidenceBrief}
        or issubclass(model, (FactExplanationBase, EvidenceDecisionEnvelopeBase))
        else 1000
        if model is ExplanationVerification
        else 12000
    )


def _validate_structured(content: str, model: type[StructuredResult]) -> StructuredResult:
    # Enforce required fields locally too: json_object mode does not enforce a provider schema.
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != set(model.model_fields):
        raise ValueError("Structured response has missing or extra fields")
    return model.model_validate(value)


class _EvidenceGateway:
    settings: CoreSettings

    def route_units(self, units, history, context, *, translate):
        schema = UnitRoutes.model_json_schema()
        schema["$defs"]["UnitRoute"]["properties"]["unit_index"] = {
            "type": "integer",
            "enum": list(range(len(units))),
        }
        if not translate:
            schema["$defs"]["UnitRoute"]["properties"]["persian_query"] = {"type": "null"}
        return self._structured(
            UnitRoutes,
            UNIT_ROUTING_INSTRUCTIONS,
            {
                "units": [{"unit_index": i, "text": text} for i, text in enumerate(units)],
                "previous_questions": history,
                "course_outline": context,
                "translate_to_persian": translate,
            },
            schema,
        )

    def prepare_evidence(
        self, question, previous, candidates, plan, *, max_sources=None, feedback=None
    ):
        from hrlearnium.grounding import overview_brief, requested_value_kind
        from hrlearnium.request_units import evidence_obligations, is_course_overview

        budget = max_sources or self.settings.max_excerpts
        if is_course_overview(question) and feedback is None:
            literal = overview_brief(candidates, budget)
            if literal is not None:
                return literal

        spans = source_spans([c.passage for c in candidates])
        if not spans or len(spans) > 450:
            raise ModelUnavailable("Invalid evidence catalog bounds", code="invalid_evidence_brief")
        model = precise_brief_model([c.passage.id for c in candidates], spans, budget)
        result = self._structured(
            model,
            EVIDENCE_BRIEF_INSTRUCTIONS
            + "\nReturn decision. Select the smallest complete source_ids list. Each fact selects ONLY the exact support_spans IDs actually supporting that fact, not every span in a passage. Every span must belong to a selected source; every selected source must support a fact. Preserve explicit numbers, conditions, uncertainty, actors and example/rule scope. Do not add optional background, examples or recaps. requested_values uses the actual literal value and its source_index; for other requests requested_values=[]. Repair feedback by removing unsupported extras or selecting genuinely supporting spans, never merely relabeling a rejected claim.",
            {
                "question": question,
                "previous_questions": previous,
                "requirements": plan.requirements,
                "max_excerpts": budget,
                "answer_obligations": evidence_obligations(question),
                "review_feedback": feedback.model_dump()
                if hasattr(feedback, "model_dump")
                else feedback
                if isinstance(feedback, dict)
                else None,
                "required_value_kind": requested_value_kind(question),
                "source_context": [
                    {
                        "id": c.passage.id,
                        "chapter_title": c.passage.chapter_title,
                        "section_kind": c.passage.section_kind,
                    }
                    for c in candidates
                ],
                "support_spans": self._span_payload(spans),
                **response_context(question),
            },
            model.model_json_schema(),
        )
        decision = result.model_dump(mode="json")["decision"]
        return normalize_precise_brief(decision, spans)

    def explain_from_brief(self, question, previous, excerpts, plan, brief):
        from hrlearnium.request_units import is_course_overview

        payload = self._explanation_payload(question, previous, excerpts)
        payload["approved_facts"] = brief.model_dump()
        for index, fact in enumerate(payload["approved_facts"]["facts"]):
            fact["fact_index"] = index
        spans = {s["id"]: s for s in source_spans(excerpts)}
        by_id = {e.id: e for e in excerpts}
        for fact in payload["approved_facts"]["facts"]:
            kinds = {
                by_id[spans[s]["citation_id"]].citation.section_kind
                for s in fact["span_ids"]
                if s in spans and spans[s]["citation_id"] in by_id
            }
            # Server-owned scope is attached before prose generation, rather than
            # relying on the writer to rediscover it from a distant passage.
            if kinds and kinds <= {"example", "exercise"}:
                scope = (
                    "طبق تمرین پیشنهادی دوره (نه واقعیت تأییدشده شرکت دانشجو): "
                    if "exercise" in kinds
                    else "در مثال آموزشی دوره (نه واقعیت تأییدشده شرکت دانشجو): "
                )
                fact["text"] = scope + fact["text"]
        allowed_spans = {s for f in brief.facts for s in f.span_ids}
        payload["support_spans"] = self._span_payload(
            [s for s in spans.values() if s["id"] in allowed_spans]
        )
        # Writers see only the approved evidence, not full passages inviting extra claims.
        payload.pop("excerpts", None)
        model = fact_explanation_model(len(brief.facts))
        written = self._structured(
            model,
            BRIEF_WRITING_INSTRUCTIONS
            + "\nReturn statements with text and fact_indices, NEVER authored citation IDs. Python derives citations from those facts. Preserve every approved fact needed by the question; omit optional examples, benefits and recaps. Each clause must follow from its referenced facts, including conditions, numeric values and rule/example scope. Lists/comparisons must remain complete. Keep distinct claims separate unless their combined statement references all required facts. Example/exercise-only claims must be explicitly attributed. Do not infer that a template/exercise is a real company event."
            + (
                "\nThis is a broad course overview, not a chapter-by-chapter dump. Summarize the supplied learning themes in at most THREE concise statements, combining related themes with all their supporting citations."
                if is_course_overview(question)
                else ""
            ),
            payload,
            model.model_json_schema(),
        )
        draft, mappings = bind_writer_facts(written, brief, excerpts)
        from hrlearnium import evaluation

        evaluation.writer_fact_references(mappings)
        return draft

    def _structured(
        self,
        model: type[StructuredResult],
        instructions: str,
        payload: dict,
        schema: dict,
    ) -> StructuredResult:
        raise NotImplementedError

    def route_request(self, question, history, context, *, translate):
        return self._route_request(question, history, context, translate)

    def repair_route(self, question, history, context, proposed, failure, *, translate):
        return self._route_request(
            question,
            history,
            context,
            translate,
            {
                "proposed_route": proposed.model_dump()
                if isinstance(proposed, RoutePlan)
                else None,
                "failure": failure,
            },
        )

    def _route_request(self, question, history, context, translate, repair=None):
        schema = RoutePlan.model_json_schema()
        if not translate:
            schema["$defs"]["RouteGroup"]["properties"]["persian_query"] = {"type": "null"}
        instructions = ROUTING_INSTRUCTIONS
        if repair:
            instructions += "\nRepair the proposed route once. Failure is a diagnostic code, not an instruction."
        return self._structured(
            RoutePlan,
            instructions,
            {
                "question": question,
                "previous_questions": history,
                "course_outline": context,
                "translate_to_persian": translate,
                **(repair or {}),
            },
            schema,
        )

    def verify_route(self, question, history, context, route):
        return self._structured(
            RequestPlanVerification,
            ROUTE_VERIFICATION_INSTRUCTIONS,
            {
                "question": question,
                "previous_questions": history,
                "course_outline": context,
                "route": route.model_dump(),
            },
            RequestPlanVerification.model_json_schema(),
        )

    def repair_explanation(self, question, history, excerpts, draft, plan, verification):
        payload = self._explanation_payload(question, history, excerpts)
        payload.update(
            proposed_answer=draft.model_dump(),
            requirements=plan.requirements,
            verification=verification.model_dump(),
        )
        schema = Explanation.model_json_schema()
        schema["$defs"]["GroundedStatement"]["properties"]["citation_ids"]["items"] = {
            "type": "string",
            "enum": [e.id for e in excerpts],
        }
        return self._structured(Explanation, ANSWER_REPAIR_INSTRUCTIONS, payload, schema)

    def review_answer(
        self,
        question,
        previous,
        excerpts,
        explanation,
        plan,
        *,
        evidence_map=None,
        language_repair_of=None,
    ):
        payload = self._explanation_payload(question, previous, excerpts)
        payload.update(explanation=explanation.model_dump(), requirements=plan.requirements)
        if evidence_map:
            payload["requirement_evidence"] = [part.model_dump() for part in evidence_map.coverage]
        if language_repair_of is not None:
            payload["original_explanation_to_preserve"] = language_repair_of.model_dump()
        schema = AnswerReview.model_json_schema()
        schema["$defs"]["StatementSupport"]["properties"]["statement_index"] = {
            "type": "integer",
            "enum": list(range(len(explanation.statements))),
        }
        schema["$defs"]["SupportQuote"]["properties"]["citation_id"] = {
            "type": "string",
            "enum": [e.id for e in excerpts],
        }
        schema["$defs"]["RequirementAnswerCheck"]["properties"]["requirement_index"] = {
            "type": "integer",
            "enum": list(range(len(plan.requirements))),
        }
        schema["$defs"]["RequirementAnswerCheck"]["properties"]["statement_indices"]["items"] = {
            "type": "integer",
            "enum": list(range(len(explanation.statements))),
        }
        return self._structured(AnswerReview, ANSWER_REVIEW_INSTRUCTIONS, payload, schema)

    def review_grounded_answer(
        self,
        question,
        previous,
        excerpts,
        explanation,
        plan,
        *,
        evidence_map=None,
        language_repair_of=None,
        context_excerpts=(),
    ):
        payload = self._explanation_payload(question, previous, excerpts)
        spans = source_spans(excerpts)
        if not spans or len(spans) > 450:
            raise ModelUnavailable("Source span catalog exceeds supported bounds")
        payload.update(
            explanation=explanation.model_dump(),
            requirements=plan.requirements,
            support_spans=self._span_payload(spans),
            candidate_context=[
                self._source_payload(e)
                for e in context_excerpts
                if e.id not in {s.id for s in excerpts}
            ],
        )
        if evidence_map:
            payload["requirement_evidence"] = [p.model_dump() for p in evidence_map.coverage]
        if language_repair_of is not None:
            payload["original_explanation_to_preserve"] = language_repair_of.model_dump()
        schema = DetailedSpanReview.model_json_schema()
        schema["$defs"]["SpanStatementSupport"]["properties"]["span_ids"]["items"] = {
            "type": "string",
            "enum": [s["id"] for s in spans],
        }
        schema["$defs"]["SpanStatementSupport"]["properties"]["statement_index"] = {
            "type": "integer",
            "enum": list(range(len(explanation.statements))),
        }
        schema["$defs"]["RequirementAnswerCheck"]["properties"]["requirement_index"] = {
            "type": "integer",
            "enum": list(range(len(plan.requirements))),
        }
        schema["$defs"]["RequirementAnswerCheck"]["properties"]["statement_indices"]["items"] = {
            "type": "integer",
            "enum": list(range(len(explanation.statements))),
        }
        for attempt in range(2):
            wire = None
            try:
                instructions = DETAILED_SPAN_REVIEW_INSTRUCTIONS
                if attempt:
                    instructions += "\nRepair the malformed REVIEW contract by re-auditing the UNCHANGED draft and evidence. Retain all genuine negative semantic judgments; do not approve merely to satisfy the schema. Do not rewrite the answer."
                    with stage("review_contract_repair"):
                        wire = self._structured(DetailedSpanReview, instructions, payload, schema)
                else:
                    wire = self._structured(DetailedSpanReview, instructions, payload, schema)
                review = normalize_detailed_review(wire, explanation, plan, excerpts)
                review_contract_check(attempt + 1, wire, [], protocol="detailed_span_checks_v12")
                return review
            except ModelUnavailable as error:
                if error.diagnostic_code not in {
                    "structured_response_invalid",
                    "invalid_answer_review",
                }:
                    raise
                review_contract_check(
                    attempt + 1, wire, [error.diagnostic_code], protocol="detailed_span_checks_v12"
                )
                if attempt:
                    raise ModelUnavailable(
                        "Invalid detailed answer review", code="invalid_answer_review"
                    ) from None
                payload["review_contract_feedback"] = {
                    "issues": [error.diagnostic_code],
                    "previous_review": wire.model_dump() if wire is not None else None,
                }

    def plan(
        self, question: str, previous_questions: list[str], *, translate: bool
    ) -> QuestionPlan:
        # A nullable field leaves both choices valid under strict generation. Encode the
        # request's setting in the schema AND local type, rather than hoping the prompt
        # wins and rejecting otherwise valid Persian questions after a billable call.
        model = TranslatedQuestionPlan if translate else UntranslatedQuestionPlan
        return self._structured(
            model,
            PLANNING_INSTRUCTIONS,
            {
                "question": question,
                "previous_questions": previous_questions,
                "translate_to_persian": translate,
            },
            model.model_json_schema(),
        )

    def plan_request(self, question, previous_questions, *, translate, target_language):
        return self._request_plan(question, previous_questions, translate, target_language)

    def repair_request_plan(
        self, question, previous_questions, plan, failure, *, translate, target_language
    ):
        return self._request_plan(
            question,
            previous_questions,
            translate,
            target_language,
            repair={
                "proposed_plan": plan.model_dump() if isinstance(plan, RequestPlan) else None,
                "failure": failure,
            },
        )

    def _request_plan(self, question, previous_questions, translate, target_language, repair=None):
        schema = RequestPlan.model_json_schema()
        schema["properties"]["answer_language"] = {"type": "string", "enum": [target_language]}
        if not translate:
            schema["$defs"]["RequestGroup"]["properties"]["persian_query"] = {"type": "null"}
        return self._structured(
            RequestPlan,
            REQUEST_PLANNING_INSTRUCTIONS + (REQUEST_PLAN_REPAIR_INSTRUCTIONS if repair else ""),
            {
                "question": question,
                "previous_questions": previous_questions,
                "translate_to_persian": translate,
                "target_language": target_language,
                **(repair or {}),
            },
            schema,
        )

    def verify_request_plan(self, question, previous_questions, plan):
        return self._structured(
            RequestPlanVerification,
            REQUEST_PLAN_VERIFICATION_INSTRUCTIONS,
            {
                "question": question,
                "previous_questions": previous_questions,
                "plan": plan.model_dump(),
            },
            RequestPlanVerification.model_json_schema(),
        )

    def repair_language(self, question, previous_questions, excerpts, explanation):
        payload = self._explanation_payload(question, previous_questions, excerpts)
        payload["proposed_answer"] = explanation.model_dump()
        schema = Explanation.model_json_schema()
        schema["properties"]["statements"].update(
            minItems=len(explanation.statements), maxItems=len(explanation.statements)
        )
        schema["$defs"]["GroundedStatement"]["properties"]["citation_ids"]["items"] = {
            "type": "string",
            "enum": [e.id for e in excerpts],
        }
        return self._structured(Explanation, LANGUAGE_REPAIR_INSTRUCTIONS, payload, schema)

    def verify_plan(self, question, previous_questions, plan):
        return self._structured(
            PlanVerification,
            PLAN_VERIFICATION_INSTRUCTIONS,
            {
                "question": question,
                "previous_questions": previous_questions,
                "requirements": plan.requirements,
            },
            PlanVerification.model_json_schema(),
        )

    def select_complete(
        self,
        question: str,
        previous_questions: list[str],
        candidates: list[Candidate],
        plan: QuestionPlan,
    ) -> CompleteSelection:
        if not candidates:
            raise ModelUnavailable("Evidence selection requires candidate passages")
        ids = [c.passage.id for c in candidates]
        model = decision_model(ids, len(plan.requirements), self.settings.max_excerpts)
        schema = model.model_json_schema()
        result = self._structured(
            model,
            COMPLETE_SELECTION_INSTRUCTIONS,
            {
                "question": question,
                "previous_questions": previous_questions,
                "requirements": plan.requirements,
                "max_excerpts": self.settings.max_excerpts,
                "passages": [
                    {
                        "id": c.passage.id,
                        "text": c.passage.text,
                        "chapter_number": c.passage.chapter_number,
                        "chapter_title": c.passage.chapter_title,
                        "section_kind": c.passage.section_kind,
                    }
                    for c in candidates
                ],
            },
            schema,
        )
        decision = result.model_dump(mode="json")["decision"]
        coverage = []
        if decision["status"] == "answered":
            for index in range(len(plan.requirements)):
                references = decision["coverage"][f"r{index}"]
                # Repeated IDs/references are harmless metadata: canonicalize without
                # inventing facts, dropping a part, or exceeding the global budget.
                ids_for_part = list(dict.fromkeys(decision["source_ids"][i] for i in references))
                coverage.append({"requirement_index": index, "passage_ids": ids_for_part})
        ids = list(dict.fromkeys(id_ for part in coverage for id_ in part["passage_ids"]))
        return CompleteSelection(
            selection=Selection(
                status=decision["status"], reason_code=decision["reason_code"], passage_ids=ids
            ),
            coverage=coverage,
        )

    def explain_complete(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        plan: QuestionPlan,
        *,
        evidence_map: CompleteSelection | None = None,
    ) -> Explanation:
        return self.explain(
            question,
            previous_questions,
            excerpts,
            requirements=plan.requirements,
            requirement_evidence=[part.model_dump() for part in evidence_map.coverage]
            if evidence_map
            else None,
        )

    def verify_complete(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        explanation: Explanation,
        plan: QuestionPlan,
        *,
        evidence_map: CompleteSelection | None = None,
        language_repair_of: Explanation | None = None,
    ) -> CompleteVerification:
        payload = self._explanation_payload(question, previous_questions, excerpts)
        payload.update(requirements=plan.requirements, explanation=explanation.model_dump())
        if language_repair_of is not None:
            payload["original_explanation_to_preserve"] = language_repair_of.model_dump()
        if evidence_map is not None:
            payload["requirement_evidence"] = [part.model_dump() for part in evidence_map.coverage]
        schema = CompleteVerification.model_json_schema()
        schema["$defs"]["RequirementAnswerCheck"]["properties"]["requirement_index"] = {
            "type": "integer",
            "enum": list(range(len(plan.requirements))),
        }
        schema["$defs"]["RequirementAnswerCheck"]["properties"]["statement_indices"]["items"] = {
            "type": "integer",
            "enum": list(range(len(explanation.statements))),
        }
        return self._structured(
            CompleteVerification, COMPLETE_VERIFICATION_INSTRUCTIONS, payload, schema
        )

    def select(
        self, question: str, previous_questions: list[str], candidates: list[Candidate]
    ) -> Selection:
        if not candidates:
            raise ModelUnavailable("Evidence selection requires candidate passages")
        schema = Selection.model_json_schema()
        schema["properties"]["passage_ids"]["items"] = {
            "type": "string",
            "enum": [item.passage.id for item in candidates],
        }
        schema["properties"]["passage_ids"]["maxItems"] = self.settings.max_excerpts
        payload = {
            "question": question,
            "previous_questions": previous_questions,
            "passages": [
                {
                    "id": item.passage.id,
                    "text": item.passage.text,
                    "chapter_number": item.passage.chapter_number,
                    "chapter_title": item.passage.chapter_title,
                    "section_kind": item.passage.section_kind,
                }
                for item in candidates
            ],
        }
        return self._structured(Selection, SELECTOR_INSTRUCTIONS, payload, schema)

    @staticmethod
    def _source_payload(e):
        return {
            "id": e.id,
            "text": e.text,
            "chapter_title": e.citation.chapter_title,
            "section_kind": e.citation.section_kind,
        }

    @staticmethod
    def _span_payload(spans):
        return [{key: s[key] for key in ("id", "citation_id", "text")} for s in spans]

    @staticmethod
    def _explanation_payload(question: str, previous_questions: list[str], excerpts: list[Excerpt]):
        if not excerpts:
            raise ModelUnavailable("Explanation requires source excerpts")
        return {
            "question": question,
            "previous_questions": previous_questions,
            "excerpts": [
                {
                    "id": e.id,
                    "text": e.text,
                    "chapter_title": e.citation.chapter_title,
                    "section_kind": e.citation.section_kind,
                }
                for e in excerpts
            ],
            **response_context(question),
        }

    def explain(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        *,
        requirements: list[str] | None = None,
        requirement_evidence: list[dict] | None = None,
    ) -> Explanation:
        payload = self._explanation_payload(question, previous_questions, excerpts)
        if requirements is not None:
            payload["requirements"] = requirements
        if requirement_evidence is not None:
            payload["requirement_evidence"] = requirement_evidence
        schema = Explanation.model_json_schema()
        schema["$defs"]["GroundedStatement"]["properties"]["citation_ids"]["items"] = {
            "type": "string",
            "enum": [excerpt.id for excerpt in excerpts],
        }
        return self._structured(Explanation, EXPLANATION_INSTRUCTIONS, payload, schema)

    def verify_explanation(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        explanation: Explanation,
    ) -> bool:
        payload = self._explanation_payload(question, previous_questions, excerpts)
        payload["explanation"] = explanation.model_dump()
        result = self._structured(
            ExplanationVerification,
            VERIFICATION_INSTRUCTIONS,
            payload,
            ExplanationVerification.model_json_schema(),
        )
        return result.supported


class OllamaGateway(_EvidenceGateway):
    def __init__(self, settings: CoreSettings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=settings.model_timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected JSON object")
            request_payload = kwargs.get("json", {})
            if method == "POST":
                usage = {
                    "prompt_tokens": result.get("prompt_eval_count"),
                    "completion_tokens": result.get("eval_count"),
                }
                provider_usage({"usage": usage}, request_payload.get("model", ""), path)
            return result
        except (httpx.HTTPError, ValueError, KeyError):
            # Do not propagate provider response bodies, source text, URLs or credentials to clients/logs.
            raise ModelUnavailable(
                "Model service unavailable or returned an invalid response"
            ) from None

    def _installed_models(self) -> dict[str, str]:
        result = self._request("GET", "/api/tags", timeout=5)
        try:
            return {item["name"]: item["digest"] for item in result["models"]}
        except (KeyError, TypeError) as error:
            raise ModelUnavailable("Invalid model inventory") from error

    def identity(self) -> str:
        installed = self._installed_models()
        name = self.settings.embedding_model
        digest = installed.get(name) or installed.get(name + ":latest")
        if not isinstance(digest, str) or not digest:
            raise ModelUnavailable("Configured embedding model is not installed")
        return f"ollama:{name}@{digest}"

    def readiness(self) -> dict:
        installed = self._installed_models()
        names = [self.settings.selector_model]
        if self.settings.retrieval_mode in {"hybrid", "hybrid_rerank"}:
            names.append(self.settings.embedding_model)
        return {
            "ready": all(name in installed or name + ":latest" in installed for name in names),
            "backend": "ollama",
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        dimensions = None
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            result = self._request(
                "POST",
                "/api/embed",
                json={
                    "model": self.settings.embedding_model,
                    "input": batch,
                    "truncate": False,
                },
            )
            normalized, dimensions = _normalized_vectors(
                result.get("embeddings"), len(batch), dimensions
            )
            vectors.extend(normalized)
        return vectors

    def _structured(
        self,
        model: type[StructuredResult],
        instructions: str,
        payload: dict,
        schema: dict,
    ) -> StructuredResult:
        result = self._request(
            "POST",
            "/api/chat",
            json={
                "model": self.settings.selector_model,
                "stream": False,
                "think": False,
                "messages": [
                    {
                        "role": "system",
                        "content": instructions + "\nSchema:\n" + json.dumps(schema),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "format": schema,
                "options": {
                    "temperature": 0,
                    "num_ctx": 32768,
                    "num_predict": (
                        4096
                        if issubclass(model, (EvidenceDecisionEnvelopeBase, FactExplanationBase))
                        or model
                        in {
                            Explanation,
                            RequestPlan,
                            RoutePlan,
                            AnswerReview,
                            SpanAnswerReview,
                            DetailedSpanReview,
                            UnitRoutes,
                            EvidenceBrief,
                        }
                        else 2048
                        if issubclass(model, QuestionPlan)
                        or issubclass(model, EvidenceDecisionEnvelopeBase)
                        or model in {CompleteSelection, CompleteVerification}
                        else 700
                    ),
                },
            },
        )
        try:
            if result.get("done") is not True or result.get("done_reason") == "length":
                raise ValueError("Incomplete selection")
            content = result["message"]["content"]
            if not isinstance(content, str) or len(content) > _content_limit(model):
                raise ValueError("Invalid response length")
            return _validate_structured(content, model)
        except (ValidationError, KeyError, TypeError, ValueError):
            raise ModelUnavailable(
                "Model returned an invalid structured evidence response",
                code="structured_response_invalid",
            ) from None

    def close(self) -> None:
        self.client.close()


class OpenAICompatibleGateway(_EvidenceGateway):
    """Explicit Chat Completions/embeddings contract; no provider guessing or fallback."""

    def __init__(self, settings: CoreSettings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            timeout=settings.model_timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    def _missing_config(self, *, embeddings: bool = False) -> list[str]:
        missing = [
            name
            for name, value in (
                ("HR_API_BASE_URL", self.settings.api_base_url),
                ("HR_API_KEY", self.settings.api_key.get_secret_value()),
                ("HR_API_MODEL", self.settings.api_model),
            )
            if not value.strip()
        ]
        if embeddings and not self.settings.api_embedding_model.strip():
            missing.append("HR_API_EMBEDDING_MODEL")
        return missing

    def _require_config(self, *, embeddings: bool = False) -> None:
        if self._missing_config(embeddings=embeddings):
            raise ModelUnavailable("Hosted model configuration is incomplete")

    def readiness(self) -> dict:
        missing = self._missing_config(
            embeddings=self.settings.retrieval_mode in {"hybrid", "hybrid_rerank"}
        )
        return {
            "ready": not missing,
            "backend": "openai_compatible",
            "configuration_complete": not missing,
            "provider_verified": False,
            "check": "configuration_only",
            "missing_settings": missing,
        }

    def identity(self) -> str:
        self._require_config(embeddings=True)
        # No key is persisted. Changing the provider URL or embedding model requires reindexing.
        provider = hashlib.sha256(self.settings.api_base_url.rstrip("/").encode()).hexdigest()
        return (
            f"openai_compatible:{provider}:{self.settings.api_embedding_model}"
            f"@{self.settings.api_embedding_revision}"
        )

    def _request(self, path: str, payload: dict) -> dict:
        self._require_config()
        deadline = time.monotonic() + self.settings.model_timeout_seconds
        transient = (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        )
        for attempt in range(self.settings.api_max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Recheck after backoff too: a suspended/paused host must not start
                # another billable attempt merely because it was eligible earlier.
                raise ModelUnavailable(
                    "Model service unavailable or returned an invalid response",
                    code="provider_timeout",
                ) from None
            code, status, retryable = None, None, False
            try:
                response = self.client.post(
                    self.settings.api_base_url.rstrip("/") + "/" + path,
                    headers={"Authorization": "Bearer " + self.settings.api_key.get_secret_value()},
                    json=payload,
                    follow_redirects=False,
                    timeout=remaining,
                )
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Expected JSON object")
                provider_attempt(payload.get("model", ""), path, attempt + 1, "success")
                provider_usage(result, payload.get("model", ""), path)
                return result
            except httpx.HTTPStatusError as error:
                code, status = "provider_http_error", error.response.status_code
                # Authentication, rate limits, schemas and all other HTTP errors remain
                # explicit failures. This fix retries only transient transport failures.
            except httpx.TimeoutException as error:
                code, retryable = "provider_timeout", isinstance(error, transient)
            except (httpx.HTTPError, httpx.InvalidURL) as error:
                code, retryable = "provider_transport_error", isinstance(error, transient)
            except (ValueError, KeyError):
                code = "provider_invalid_response"
            delay = self.settings.api_retry_backoff_seconds * (2**attempt)
            retry = (
                retryable
                and attempt < self.settings.api_max_retries
                and deadline - time.monotonic() > delay + 0.001
            )
            provider_attempt(
                payload.get("model", ""),
                path,
                attempt + 1,
                code,
                provider_status=status,
                will_retry=retry,
            )
            if not retry:
                raise ModelUnavailable(
                    "Model service unavailable or returned an invalid response",
                    code=code,
                    provider_status=status,
                ) from None
            time.sleep(delay)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._require_config(embeddings=True)
        vectors = []
        dimensions = None
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            result = self._request(
                "embeddings",
                {
                    "model": self.settings.api_embedding_model,
                    "input": batch,
                    "encoding_format": "float",
                },
            )
            data = result.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise ModelUnavailable("Invalid embedding count")
            ordered = [None] * len(batch)
            seen = set()
            for item in data:
                if not isinstance(item, dict):
                    raise ModelUnavailable("Invalid embedding response")
                index = item.get("index")
                if type(index) is not int or not 0 <= index < len(batch) or index in seen:
                    raise ModelUnavailable("Invalid embedding index")
                seen.add(index)
                ordered[index] = item.get("embedding")
            normalized, dimensions = _normalized_vectors(ordered, len(batch), dimensions)
            vectors.extend(normalized)
        return vectors

    def _structured(
        self,
        model: type[StructuredResult],
        instructions: str,
        payload: dict,
        schema: dict,
    ) -> StructuredResult:
        schema = _strict_schema(schema)
        response_format = {"type": "json_object"}
        if self.settings.api_structured_output == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": model.__name__, "strict": True, "schema": schema},
            }
        result = self._request(
            "chat/completions",
            {
                "model": self.settings.api_model,
                "stream": False,
                "store": False,
                "max_completion_tokens": self.settings.api_max_completion_tokens,
                "response_format": response_format,
                "messages": [
                    {
                        "role": "system",
                        "content": instructions + "\nJSON schema:\n" + json.dumps(schema),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            },
        )
        try:
            choices = result["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("Invalid completion count")
            choice = choices[0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete completion")
            message = choice["message"]
            if (
                not isinstance(message, dict)
                or message.get("refusal")
                or message.get("tool_calls")
                or message.get("function_call")
            ):
                raise ValueError("No structured completion")
            content = message["content"]
            if not isinstance(content, str) or len(content) > _content_limit(model):
                raise ValueError("Invalid completion length")
            return _validate_structured(content, model)
        except (ValidationError, KeyError, TypeError, ValueError, AttributeError):
            raise ModelUnavailable(
                "Model returned an invalid structured evidence response",
                code="structured_response_invalid",
            ) from None

    def close(self) -> None:
        self.client.close()


class LiteralGateway:
    """Offline diagnostic: only explicitly requested, exact quoted phrases can be looked up.

    This is NOT a replacement for the semantic answerability gate. It makes no claims
    about a passage answering a natural-language question and never guesses relevance.
    """

    def identity(self) -> None:
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ModelUnavailable("Literal lookup has no semantic model")

    def explain(
        self, question: str, previous_questions: list[str], excerpts: list[Excerpt]
    ) -> Explanation:
        raise ModelUnavailable("Literal lookup cannot generate explanations")

    def verify_explanation(
        self,
        question: str,
        previous_questions: list[str],
        excerpts: list[Excerpt],
        explanation: Explanation,
    ) -> bool:
        raise ModelUnavailable("Literal lookup cannot verify explanations")

    def readiness(self) -> dict:
        return {"ready": True, "backend": "literal", "natural_language_qa": False}

    def select(
        self, question: str, previous_questions: list[str], candidates: list[Candidate]
    ) -> Selection:
        # A full-match grammar prevents a quote from masking an unrelated request or extra advice.
        match = re.fullmatch(
            r'\s*(?:متن دقیق|عبارت دقیق|نقل قول|exact quote)\s*[«"]([^»"\n]{5,300})[»"]\s*[.؟?]?\s*',
            question,
            re.I,
        )
        if not match:
            return Selection(status="refused", passage_ids=[], reason_code="insufficient_evidence")
        phrase = normalize_persian(match.group(1))
        if len(phrase.strip()) < 5 or not any(character.isalnum() for character in phrase):
            return Selection(status="refused", passage_ids=[], reason_code="insufficient_evidence")
        matches = [
            item.passage.id for item in candidates if phrase in normalize_persian(item.passage.text)
        ]
        if len(matches) == 1:
            return Selection(status="answered", passage_ids=matches, reason_code="supported")
        return Selection(
            status="clarification" if matches else "refused",
            passage_ids=[],
            reason_code="ambiguous" if matches else "insufficient_evidence",
        )

    def close(self) -> None:
        pass


def create_gateway(settings: CoreSettings) -> ModelGateway:
    if settings.model_backend == "ollama":
        return OllamaGateway(settings)
    if settings.model_backend == "openai_compatible":
        return OpenAICompatibleGateway(settings)
    if settings.model_backend == "literal":
        return LiteralGateway()
    raise ModelUnavailable("Unsupported model backend")
