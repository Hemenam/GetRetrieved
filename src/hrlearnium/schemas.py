from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$")]
Scope = Literal["query", "content:read", "content:write"]
ResponseMode = Literal["verbatim", "explained"]
RetrievalMode = Literal["full_context", "hybrid", "hybrid_rerank"]
POLICY_VERSION = "grounded-course-v12.2"
AnswerLanguage = Literal["fa", "en"]
MetadataField = Literal[
    "course_title",
    "chapter_count",
    "chapter_titles",
    "chapter_title",
    "first_chapter_title",
    "last_chapter_title",
    "page_number",
    "chapter_sections",
    "instructor",
    "duration",
    "prerequisites",
    "other",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Principal(StrictModel):
    sub: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    tenant_id: Identifier
    course_ids: list[Identifier] = Field(min_length=1, max_length=200)
    scopes: list[Scope] = Field(min_length=1, max_length=3)


class QueryRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "مدل ۳۰-۳۰-۳۰ چطور به اولویت‌بندی در بحران کمک می‌کند؟",
                    "response_mode": "explained",
                }
            ]
        }
    )
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)
    ]
    response_mode: ResponseMode = Field(
        default="explained",
        description="Both modes accept natural questions. Verbatim returns exact source excerpts; "
        "explained (default) returns a natural cited answer; exact evidence remains in excerpts.",
    )
    retrieval_mode: RetrievalMode | None = Field(
        default=None,
        description="Override evidence retrieval for this request only. Omit to use the server default. "
        "Hybrid modes require current course embeddings; reranking also needs its runtime/model files.",
    )
    conversation_id: UUID | None = Field(
        default=None,
        description=(
            "Omit this field for your first question. For a follow-up, copy the conversation_id "
            "returned by a successful request for this same user and course. Do not invent an ID."
        ),
    )
    include_evaluation: bool = Field(
        default=False,
        description="Include timings, ranking and configuration for evaluation tools.",
    )


class RetrievalOption(StrictModel):
    mode: RetrievalMode
    available: bool
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RetrievalOptions(StrictModel):
    default_mode: Literal["lexical", "full_context", "hybrid", "hybrid_rerank"]
    check: Literal["prerequisites_only"] = "prerequisites_only"
    provider_verified: bool = False
    passage_count: int
    embedded_passage_count: int
    modes: list[RetrievalOption]


class Citation(StrictModel):
    document_id: str
    version: int
    document_title: str
    chapter_number: int | None
    chapter_title: str
    section_kind: str
    paragraph_start: int
    paragraph_end: int
    source_start: int
    source_end: int
    source_sha256: str


class Excerpt(StrictModel):
    id: str
    text: str
    citation: Citation


class GroundedStatement(StrictModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)]
    citation_ids: list[str] = Field(min_length=1, max_length=8)


class Explanation(StrictModel):
    statements: list[GroundedStatement] = Field(min_length=1, max_length=8)


class ExplanationVerification(StrictModel):
    supported: bool = Field(strict=True)


class QuestionPlan(StrictModel):
    """Search/decomposition data only; never source facts or executable instructions."""

    search_query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)
    ]
    persian_query: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)]
        | None
    )
    requirements: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=1000)]
    ] = Field(min_length=1, max_length=8)


class PlanVerification(StrictModel):
    all_parts_identified: bool = Field(strict=True)
    independent: bool = Field(strict=True)


class RequestGroup(StrictModel):
    """One answer unit. Original text is authority; search data is not."""

    original_text: Annotated[str, StringConstraints(min_length=2, max_length=2000)]
    kind: Literal["single", "joint"]
    intent: Literal["content", "metadata"]
    metadata_field: MetadataField | None
    chapter_number: Annotated[int, Field(strict=True, ge=1)] | None
    search_query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)
    ]
    persian_query: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)]
        | None
    )
    requirements: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=1000)]
    ] = Field(min_length=1, max_length=8)


class RequestPlan(StrictModel):
    answer_language: AnswerLanguage
    needs_clarification: bool = Field(strict=True)
    groups: list[RequestGroup] = Field(max_length=8)


class RequestPlanVerification(StrictModel):
    all_parts_identified: bool = Field(strict=True)
    no_added_requests: bool = Field(strict=True)
    grouping_valid: bool = Field(strict=True)
    intents_valid: bool = Field(strict=True)


class RouteGroup(StrictModel):
    """Interpretation only. Evidence requirements and response language are server-owned."""

    text: Annotated[str, StringConstraints(min_length=2, max_length=2000)]
    intent: Literal["content", "metadata"]
    metadata_fields: list[MetadataField] = Field(max_length=8)
    chapter_number: Annotated[int, Field(strict=True, ge=1)] | None
    search_query: Annotated[str, StringConstraints(min_length=2, max_length=2000)]
    persian_query: Annotated[str, StringConstraints(min_length=2, max_length=2000)] | None


class RoutePlan(StrictModel):
    # There is intentionally no pre-retrieval "needs_clarification" or generated
    # requirements list. The evidence selector judges ambiguity after searching.
    groups: list[RouteGroup] = Field(min_length=1, max_length=8)


class UnitRoute(StrictModel):
    unit_index: int = Field(strict=True, ge=0, le=7)
    parent_unit_index: Annotated[int, Field(strict=True, ge=0, le=7)] | None = None
    intent: Literal["content", "metadata"]
    metadata_fields: list[MetadataField] = Field(max_length=8)
    chapter_number: Annotated[int, Field(strict=True, ge=1)] | None
    persian_query: Annotated[str, StringConstraints(min_length=2, max_length=1000)] | None


class UnitRoutes(StrictModel):
    units: list[UnitRoute] = Field(min_length=1, max_length=8)


class EvidenceFact(StrictModel):
    text: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    span_ids: list[str] = Field(min_length=1, max_length=450)


class EvidenceValue(StrictModel):
    value: str
    source_id: str


class EvidenceBrief(StrictModel):
    status: Literal["answered", "refused", "clarification"]
    reason_code: Literal[
        "supported",
        "insufficient_evidence",
        "outside_scope",
        "new_advice",
        "ambiguous",
        "example_not_fact",
    ]
    complete: bool = Field(strict=True)
    facts: list[EvidenceFact] = Field(max_length=8)
    requested_values: list[EvidenceValue] = Field(default_factory=list, max_length=8)


class SupportQuote(StrictModel):
    citation_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    quote: Annotated[str, StringConstraints(min_length=1, max_length=1600)]


class StatementSupport(StrictModel):
    statement_index: int = Field(strict=True, ge=0)
    supported: bool = Field(strict=True)
    evidence: list[SupportQuote] = Field(max_length=8)


class UntranslatedQuestionPlan(QuestionPlan):
    """Translation is disabled: require JSON null, not a nullable model choice."""

    persian_query: None


class TranslatedQuestionPlan(QuestionPlan):
    """Translation is enabled: require a nonempty string at both boundaries."""

    persian_query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)
    ]


class RequirementEvidence(StrictModel):
    requirement_index: int = Field(strict=True, ge=0, le=7)
    passage_ids: list[str] = Field(max_length=8)


class CompleteSelection(StrictModel):
    selection: "Selection"
    coverage: list[RequirementEvidence] = Field(max_length=8)


class RefusedEvidenceDecision(StrictModel):
    status: Literal["refused"]
    reason_code: Literal[
        "insufficient_evidence",
        "outside_scope",
        "new_advice",
        "example_not_fact",
        "conflicting_evidence",
        "policy_override",
    ]


class ClarificationEvidenceDecision(StrictModel):
    status: Literal["clarification"]
    reason_code: Literal["ambiguous"]


class EvidenceDecisionEnvelopeBase(StrictModel):
    """Marker for the bounded per-request selection schemas."""


class RequirementAnswerCheck(StrictModel):
    requirement_index: int = Field(strict=True, ge=0, le=7)
    covered: bool = Field(strict=True)
    statement_indices: list[Annotated[int, Field(strict=True, ge=0, le=7)]] = Field(max_length=8)


class CompleteVerification(StrictModel):
    supported: bool = Field(strict=True)
    all_question_parts_identified: bool = Field(strict=True)
    coverage: list[RequirementAnswerCheck] = Field(max_length=8)


class AnswerReview(StrictModel):
    verification: CompleteVerification
    statements: list[StatementSupport] = Field(max_length=8)


class IntentSupport(StrictModel):
    answers_original_question: bool = Field(strict=True)
    actor_action_preserved: bool = Field(strict=True)
    source_stance_preserved: bool = Field(strict=True)
    causal_claims_supported: bool = Field(strict=True)


class SpanStatementSupport(StrictModel):
    statement_index: int = Field(strict=True, ge=0)
    supported: bool = Field(strict=True)
    span_ids: list[str] = Field(max_length=8)


class SpanAnswerReview(StrictModel):
    verification: CompleteVerification
    intent: IntentSupport
    statements: list[SpanStatementSupport] = Field(max_length=8)
    rejection_reasons: list[Annotated[str, StringConstraints(min_length=1, max_length=600)]] = Field(
        default_factory=list, max_length=8
    )


def render_answer(excerpts: list[Excerpt], explanation: Explanation | None = None) -> str:
    if explanation is not None:
        indices = {excerpt.id: index for index, excerpt in enumerate(excerpts, start=1)}
        lines = [
            statement.text + " " + " ".join(f"[{indices[id_]}]" for id_ in statement.citation_ids)
            for statement in explanation.statements
        ]
        return "\n\n".join(lines)
    return "\n\n".join(excerpt.text for excerpt in excerpts)


class OutlineChapter(StrictModel):
    number: int
    title: str
    sections: list[str]


class OutlineDocument(StrictModel):
    document_id: str
    version: int
    title: str
    source_sha256: str
    chapters: list[OutlineChapter]
    warnings: list[str]


class CourseOutline(StrictModel):
    course_id: str
    provenance: Literal["imported_document_headings"] = "imported_document_headings"
    snapshot: str
    documents: list[OutlineDocument]
    warnings: list[str]


class MetadataEvidence(StrictModel):
    field: str
    chapter_number: int | None = None
    language: Literal["fa", "en"]
    outline: CourseOutline


class AnswerPart(StrictModel):
    question: str
    status: Literal["answered", "refused", "clarification", "error"]
    answer_kind: Literal["content", "metadata"] = "content"
    answer: str
    reason_code: str
    excerpt_ids: list[str] = Field(default_factory=list)
    explanation: Explanation | None = None
    metadata: MetadataEvidence | None = None


def render_parts(parts: list[AnswerPart], excerpts: list[Excerpt], response_mode: str) -> str:
    result = []
    for index, part in enumerate(parts, start=1):
        text = part.answer
        if part.answer_kind == "content" and part.status == "answered":
            sources = [e for e in excerpts if e.id in part.excerpt_ids]
            if response_mode == "explained":
                # Use one global citation numbering across the whole response.
                text = render_answer(excerpts, part.explanation)
            else:
                text = render_answer(sources)
        result.append(f"{index}. {part.question}\n{text}")
    return "\n\n".join(result)


class QueryResponse(StrictModel):
    request_id: str
    status: Literal["answered", "partial", "refused", "clarification", "conversation", "error"]
    answer: str
    excerpts: list[Excerpt]
    conversation_id: str
    reason_code: str
    retrieval_mode: Literal["lexical", "hybrid", "hybrid_rerank", "full_context"]
    response_mode: ResponseMode = "explained"
    explanation: Explanation | None = None
    policy_version: str = POLICY_VERSION
    evaluation: dict | None = None
    answer_kind: Literal["content", "metadata", "multi"] = "content"
    metadata: MetadataEvidence | None = None
    parts: list[AnswerPart] = Field(default_factory=list, max_length=8)


class DocumentResponse(StrictModel):
    id: str
    title: str
    filename: str
    version: int
    sha256: str
    active: bool
    passage_count: int
    paragraph_count: int
    warnings: list[str]
    created_at: str
    updated_at: str


class Selection(StrictModel):
    status: Literal["answered", "refused", "clarification"]
    passage_ids: list[str] = Field(default_factory=list, max_length=8)
    reason_code: Literal[
        "supported",
        "insufficient_evidence",
        "outside_scope",
        "new_advice",
        "ambiguous",
        "example_not_fact",
        "conflicting_evidence",
        "policy_override",
    ]
