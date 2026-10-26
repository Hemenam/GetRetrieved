"""Pure offline checks for the nonredundant reviewer wire contract."""

import copy

import pytest
from pydantic import ValidationError

from hrlearnium.detailed_review import DetailedSpanReview, normalize_detailed_review
from hrlearnium.models import ModelUnavailable
from hrlearnium.schemas import Excerpt, Explanation, QuestionPlan, SpanAnswerReview
from hrlearnium.source_spans import source_spans


def source(identifier="source-1", text="The first requested fact is available here."):
    return Excerpt(
        id=identifier,
        text=text,
        citation={
            "document_id": "document-1",
            "version": 1,
            "document_title": "A synthetic course",
            "chapter_number": 1,
            "chapter_title": "One",
            "section_kind": "explanation",
            "paragraph_start": 0,
            "paragraph_end": 0,
            "source_start": 0,
            "source_end": len(text),
            "source_sha256": "a" * 64,
        },
    )


@pytest.fixture
def context():
    excerpts = [source(), source("source-2", "The second requested fact is available here.")]
    explanation = Explanation(
        statements=[
            {"text": "The first requested fact.", "citation_ids": [excerpts[0].id]},
            {"text": "The second requested fact.", "citation_ids": [excerpts[1].id]},
        ]
    )
    plan = QuestionPlan(
        search_query="Two requested facts?", persian_query=None, requirements=["First?", "Second?"]
    )
    spans = source_spans(excerpts)
    wire = DetailedSpanReview(
        all_question_parts_identified=True,
        intent={
            "answers_original_question": True,
            "actor_action_preserved": True,
            "source_stance_preserved": True,
            "causal_claims_supported": True,
        },
        coverage=[
            {"requirement_index": i, "covered": True, "statement_indices": [i]} for i in range(2)
        ],
        statements=[
            {"statement_index": i, "supported": True, "span_ids": [spans[i]["id"]]}
            for i in range(2)
        ],
        rejection_reasons=[],
    )
    return wire, explanation, plan, excerpts


def test_wire_has_no_model_authored_overall_support_or_verification():
    fields = DetailedSpanReview.model_fields
    assert set(fields) == {
        "all_question_parts_identified",
        "intent",
        "coverage",
        "statements",
        "rejection_reasons",
    }
    assert "supported" not in DetailedSpanReview.model_json_schema()["properties"]
    assert all(field.is_required() for field in fields.values())


def test_valid_all_positive_details_return_legacy_model_without_mutating_inputs(context):
    original = copy.deepcopy(context)
    result = normalize_detailed_review(*context)
    assert isinstance(result, SpanAnswerReview)
    assert result.verification.supported
    assert result.verification.all_question_parts_identified
    assert result.verification.coverage == context[0].coverage
    assert result.statements == context[0].statements
    assert result.intent == context[0].intent
    assert result.rejection_reasons == []
    assert context == original
    result.statements[0].span_ids.clear()
    assert context == original


@pytest.mark.parametrize(
    "check",
    [
        "all_parts",
        "answers_original_question",
        "actor_action_preserved",
        "source_stance_preserved",
        "causal_claims_supported",
        "statement",
        "coverage",
    ],
)
def test_each_negative_semantic_check_is_a_valid_rejection_not_protocol_error(context, check):
    wire, explanation, plan, excerpts = context
    if check == "all_parts":
        wire.all_question_parts_identified = False
    elif check == "statement":
        wire.statements[0].supported = False
        wire.statements[0].span_ids = []
    elif check == "coverage":
        wire.coverage[0].covered = False
        wire.coverage[0].statement_indices = []
    else:
        setattr(wire.intent, check, False)
    wire.rejection_reasons = [f"Specific failed check: {check}; its requested claim lacks support."]
    original = wire.model_dump()
    result = normalize_detailed_review(wire, explanation, plan, excerpts)
    assert result.verification.supported is False
    assert result.statements == wire.statements
    assert result.intent == wire.intent
    assert result.rejection_reasons == wire.rejection_reasons
    assert wire.model_dump() == original


@pytest.mark.parametrize(
    "fault",
    [
        "missing_statement",
        "duplicate_statement",
        "foreign_statement",
        "negative_statement",
        "missing_requirement",
        "duplicate_requirement",
        "foreign_requirement",
        "negative_requirement",
        "duplicate_span",
        "unknown_span",
        "other_statements_source",
        "stale_source_span",
        "supported_without_span",
        "rejected_with_span",
        "covered_without_indices",
        "uncovered_with_indices",
        "duplicate_answer_indices",
        "foreign_answer_index",
        "negative_answer_index",
        "passing_with_reason",
        "rejected_without_reason",
        "blank_reason",
        "invalid_boolean",
        "duplicate_source",
        "missing_cited_source",
        "duplicate_citation",
    ],
)
def test_protocol_faults_raise_redacted_invalid_answer_review(context, fault):
    wire, explanation, plan, excerpts = context
    if fault == "missing_statement":
        wire.statements.pop()
    elif fault == "duplicate_statement":
        wire.statements[1].statement_index = 0
    elif fault == "foreign_statement":
        wire.statements[1].statement_index = 2
    elif fault == "negative_statement":
        wire.statements[1].statement_index = -1
    elif fault == "missing_requirement":
        wire.coverage.pop()
    elif fault == "duplicate_requirement":
        wire.coverage[1].requirement_index = 0
    elif fault == "foreign_requirement":
        wire.coverage[1].requirement_index = 2
    elif fault == "negative_requirement":
        wire.coverage[1].requirement_index = -1
    elif fault == "duplicate_span":
        wire.statements[0].span_ids *= 2
    elif fault == "unknown_span":
        wire.statements[0].span_ids = ["private-source-secret-not-a-span"]
    elif fault == "other_statements_source":
        wire.statements[0].span_ids = wire.statements[1].span_ids[:]
    elif fault == "stale_source_span":
        excerpts[0].text += " A revised sentence changes its source identity."
    elif fault == "supported_without_span":
        wire.statements[0].span_ids = []
    elif fault == "rejected_with_span":
        wire.statements[0].supported = False
        wire.rejection_reasons = ["The first statement lacks support."]
    elif fault == "covered_without_indices":
        wire.coverage[0].statement_indices = []
    elif fault == "uncovered_with_indices":
        wire.coverage[0].covered = False
        wire.rejection_reasons = ["The first requirement is incomplete."]
    elif fault == "duplicate_answer_indices":
        wire.coverage[0].statement_indices = [0, 0]
    elif fault == "foreign_answer_index":
        wire.coverage[0].statement_indices = [2]
    elif fault == "negative_answer_index":
        wire.coverage[0].statement_indices = [-1]
    elif fault == "passing_with_reason":
        wire.rejection_reasons = ["The manager's direction is reversed."]
    elif fault == "rejected_without_reason":
        wire.intent.actor_action_preserved = False
    elif fault == "blank_reason":
        wire.intent.actor_action_preserved = False
        wire.rejection_reasons = ["   "]
    elif fault == "invalid_boolean":
        wire.intent.actor_action_preserved = "false"
    elif fault == "duplicate_source":
        excerpts.append(excerpts[0])
    elif fault == "missing_cited_source":
        excerpts.pop(0)
    else:
        explanation.statements[0].citation_ids *= 2
    with pytest.raises(ModelUnavailable) as error:
        normalize_detailed_review(wire, explanation, plan, excerpts)
    assert error.value.diagnostic_code == "invalid_answer_review"
    assert "private-source-secret" not in str(error.value)
    assert excerpts[0].text not in str(error.value)


def test_wrong_wire_type_is_not_silently_treated_as_a_new_protocol(context):
    with pytest.raises(ModelUnavailable) as error:
        normalize_detailed_review(context[0].model_dump(), *context[1:])
    assert error.value.diagnostic_code == "invalid_answer_review"


def test_new_wire_rejects_legacy_aggregate_fields(context):
    payload = context[0].model_dump()
    payload["supported"] = False
    with pytest.raises(ValidationError):
        DetailedSpanReview.model_validate(payload)


def test_coverage_may_reuse_a_statement_for_a_genuine_joint_answer(context):
    wire, explanation, plan, excerpts = context
    for check in wire.coverage:
        check.statement_indices = [0, 1]
    assert normalize_detailed_review(wire, explanation, plan, excerpts).verification.supported


def test_required_rejected_statement_is_preserved_for_existing_repair_logic(context):
    wire, explanation, plan, excerpts = context
    wire.statements[0].supported = False
    wire.statements[0].span_ids = []
    wire.rejection_reasons = [
        "Statement 0 changes a conditional suggestion into a mandatory instruction."
    ]
    result = normalize_detailed_review(wire, explanation, plan, excerpts)
    assert not result.verification.supported
    assert result.verification.coverage[0].statement_indices == [0]
    assert not result.statements[0].supported


def test_optional_rejected_statement_does_not_falsify_complete_requested_coverage(context):
    wire, explanation, plan, excerpts = context
    wire.statements[1].supported = False
    wire.statements[1].span_ids = []
    for check in wire.coverage:
        check.statement_indices = [0]
    wire.rejection_reasons = ["Statement 1 adds an unnecessary unsupported outcome."]
    result = normalize_detailed_review(wire, explanation, plan, excerpts)
    assert not result.verification.supported
    assert result.verification.all_question_parts_identified
    assert all(check.covered for check in result.verification.coverage)
    assert result.verification.coverage == wire.coverage


@pytest.mark.parametrize("part", ["explanation", "plan", "sources", "source_entry", "source_text"])
def test_invalid_context_cannot_escape_as_an_unclassified_exception(context, part):
    wire, explanation, plan, excerpts = context
    if part == "explanation":
        explanation = None
    elif part == "plan":
        plan = None
    elif part == "sources":
        excerpts = None
    elif part == "source_entry":
        excerpts = [None]
    else:
        excerpts[0].text = None
    with pytest.raises(ModelUnavailable) as error:
        normalize_detailed_review(wire, explanation, plan, excerpts)
    assert error.value.diagnostic_code == "invalid_answer_review"


def test_unused_context_source_cannot_supply_a_statements_evidence(context):
    wire, explanation, plan, excerpts = context
    other = source("unselected-source", "A claim that appears elsewhere in the course.")
    excerpts.append(other)
    wire.statements[0].span_ids = [source_spans([other])[0]["id"]]
    with pytest.raises(ModelUnavailable) as error:
        normalize_detailed_review(wire, explanation, plan, excerpts)
    assert error.value.diagnostic_code == "invalid_answer_review"


def test_both_covered_and_supported_indices_may_arrive_out_of_order(context):
    wire, explanation, plan, excerpts = context
    wire.statements.reverse()
    wire.coverage.reverse()
    result = normalize_detailed_review(wire, explanation, plan, excerpts)
    assert result.verification.supported
    assert result.statements == wire.statements
    assert result.verification.coverage == wire.coverage
