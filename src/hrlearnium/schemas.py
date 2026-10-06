from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$")]
Scope = Literal["query", "content:read", "content:write"]
ResponseMode = Literal["verbatim", "explained"]
POLICY_VERSION = "grounded-course-v2"


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
                    "response_mode": "verbatim",
                }
            ]
        }
    )
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)
    ]
    response_mode: ResponseMode = Field(
        default="verbatim",
        description="Both modes accept natural questions. Verbatim returns exact source excerpts; "
        "explained also includes a model explanation grounded in those excerpts.",
    )
    conversation_id: UUID | None = Field(
        default=None,
        description=(
            "Omit this field for your first question. For a follow-up, copy the conversation_id "
            "returned by a successful request for this same user and course. Do not invent an ID."
        ),
    )


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


def render_answer(excerpts: list[Excerpt], explanation: Explanation | None = None) -> str:
    answer = "\n\n".join(excerpt.text for excerpt in excerpts)
    if explanation is not None:
        indices = {excerpt.id: index for index, excerpt in enumerate(excerpts, start=1)}
        lines = [
            statement.text + " " + " ".join(f"[{indices[id_]}]" for id_ in statement.citation_ids)
            for statement in explanation.statements
        ]
        answer += "\n\nتوضیح بر اساس متن دوره:\n" + "\n".join(lines)
    return answer


class QueryResponse(StrictModel):
    request_id: str
    status: Literal["answered", "refused", "clarification"]
    answer: str
    excerpts: list[Excerpt]
    conversation_id: str
    reason_code: str
    retrieval_mode: Literal["lexical", "hybrid", "full_context"]
    response_mode: ResponseMode = "verbatim"
    explanation: Explanation | None = None
    policy_version: str = POLICY_VERSION


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
