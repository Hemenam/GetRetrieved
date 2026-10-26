"""Detailed reviewer wire protocol; only Python computes its aggregate verdict.

Source ownership and contract consistency are deterministic. Semantic support,
intent and completeness remain model judgments, not proven by this normalizer.
"""

from typing import Annotated

from pydantic import Field, StringConstraints, ValidationError

from hrlearnium.schemas import (
    CompleteVerification,
    Excerpt,
    Explanation,
    IntentSupport,
    QuestionPlan,
    RequirementAnswerCheck,
    SpanAnswerReview,
    SpanStatementSupport,
    StrictModel,
)
from hrlearnium.source_spans import source_spans


class DetailedSpanReview(StrictModel):
    """One detailed decision per check; no model-authored overall verdict."""

    all_question_parts_identified: bool = Field(strict=True)
    intent: IntentSupport
    coverage: list[RequirementAnswerCheck] = Field(max_length=8)
    statements: list[SpanStatementSupport] = Field(max_length=8)
    rejection_reasons: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]
    ] = Field(max_length=8)


def _invalid(reason):
    # Lazy import permits the shipped gateway to import this wire model without
    # a models -> detailed_review -> models import cycle.
    from hrlearnium.models import ModelUnavailable

    raise ModelUnavailable(
        "Invalid detailed answer review: " + reason, code="invalid_answer_review"
    )


def normalize_detailed_review(wire, explanation, plan, excerpts) -> SpanAnswerReview:
    """Validate the wire response and return the existing internal review model.

    Negative semantic checks are legitimate rejections, not protocol errors.
    Malformed indices, unsupported-span payloads, foreign support and inconsistent
    reasons are protocol errors. No negative vote can be removed or promoted.
    Inputs are not mutated; the returned compatibility object owns its copies.
    """
    if not isinstance(wire, DetailedSpanReview):
        _invalid("wrong_wire_type")
    if not isinstance(explanation, Explanation) or not isinstance(plan, QuestionPlan):
        _invalid("invalid_review_context")
    if not isinstance(excerpts, (list, tuple)) or not all(isinstance(e, Excerpt) for e in excerpts):
        _invalid("invalid_source_context")
    try:
        # Revalidate a serialized copy too: in-process adapters/tests may have
        # used model_copy(update=...) or assigned fields without validation.
        wire = DetailedSpanReview.model_validate(wire.model_dump(warnings=False))
        explanation = Explanation.model_validate(explanation.model_dump(warnings=False))
        plan = QuestionPlan.model_validate(plan.model_dump(warnings=False))
        excerpts = [Excerpt.model_validate(e.model_dump(warnings=False)) for e in excerpts]
    except (ValidationError, TypeError, ValueError):
        _invalid("invalid_field_value")

    statement_count, requirement_count = len(explanation.statements), len(plan.requirements)
    if sorted(row.statement_index for row in wire.statements) != list(range(statement_count)):
        _invalid("statement_checks_must_cover_each_statement_once")
    if sorted(row.requirement_index for row in wire.coverage) != list(range(requirement_count)):
        _invalid("coverage_must_identify_each_requirement_once")
    by_id = {source.id: source for source in excerpts}
    if len(by_id) != len(excerpts):
        _invalid("duplicate_source_ids")
    spans = {span["id"]: span for span in source_spans(excerpts)}
    for row in wire.statements:
        own = explanation.statements[row.statement_index].citation_ids
        if len(set(own)) != len(own) or not set(own) <= set(by_id):
            _invalid("invalid_statement_citations")
        if row.supported != bool(row.span_ids) or len(set(row.span_ids)) != len(row.span_ids):
            _invalid("statement_support_and_spans_disagree")
        if any(
            identifier not in spans or spans[identifier]["citation_id"] not in own
            for identifier in row.span_ids
        ):
            _invalid("support_span_not_owned_by_statement")
    for row in wire.coverage:
        if (
            row.covered != bool(row.statement_indices)
            or len(set(row.statement_indices)) != len(row.statement_indices)
            or any(i >= statement_count for i in row.statement_indices)
        ):
            _invalid("coverage_and_answering_statements_disagree")

    supported = (
        wire.all_question_parts_identified
        and all(wire.intent.model_dump().values())
        and all(row.supported for row in wire.statements)
        and all(row.covered for row in wire.coverage)
    )
    if supported and wire.rejection_reasons:
        _invalid("passing_review_contains_rejection_reasons")
    if not supported and not wire.rejection_reasons:
        _invalid("rejection_requires_a_specific_reason")
    return SpanAnswerReview(
        verification=CompleteVerification(
            supported=supported,
            all_question_parts_identified=wire.all_question_parts_identified,
            coverage=wire.coverage,
        ),
        intent=wire.intent,
        statements=wire.statements,
        rejection_reasons=wire.rejection_reasons,
    ).model_copy(deep=True)
