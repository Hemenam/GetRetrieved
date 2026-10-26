"""Course-independent contracts; all provider traffic is mocked."""

import json

import pytest
from test_answer_support_v9 import span_review
from test_api import BASE, QUESTION, token, upload
from test_api import api_factory as api_factory
from test_hosted_gateway import completion, excerpt, explanation, gateway, response
from test_hosted_gateway import settings as settings
from test_response_modes import install_explainer
from test_routing_v8 import install_route, route_group

from hrlearnium.answer_contracts import (
    conservative_rejection,
    review_contract_issues,
    unsupported_literal_quotes,
)
from hrlearnium.grounding import prune_incidental_statements, review_supports_statements
from hrlearnium.models import ModelUnavailable
from hrlearnium.schemas import Explanation, QuestionPlan


def plan():
    return QuestionPlan(search_query="A question", persian_query=None, requirements=["A question"])


@pytest.mark.parametrize("verdict", ["pass", "reject", "contradiction", "missing_reason"])
def test_review_consistency_never_flips_a_verdict(verdict):
    draft, source = explanation(), excerpt()
    review = span_review(draft, [source])
    if verdict != "pass":
        review.verification.supported = False
    if verdict in {"reject", "missing_reason"}:
        review.statements[0].supported = False
        review.statements[0].span_ids = []
    if verdict == "reject":
        review.rejection_reasons = ["Statement 0 attributes an unsupported cause."]
    original = review.model_dump()
    assert bool(review_contract_issues(review, len(draft.statements), 1)) == (
        verdict in {"contradiction", "missing_reason"}
    )
    assert review.model_dump() == original


@pytest.mark.parametrize("second", ["pass", "reject", "contradiction"])
def test_gateway_rechecks_unchanged_draft_once_then_fails_closed(settings, second):
    draft, source, calls = explanation(), excerpt(), []
    good = span_review(draft, [source])
    broken = good.model_copy(deep=True)
    broken.verification.supported = False
    rejected = broken.model_copy(deep=True)
    rejected.statements[0].supported = False
    rejected.statements[0].span_ids = []
    rejected.rejection_reasons = ["Statement 0 is not entailed by its source."]

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        if len(calls) == 2:
            assert payload["explanation"] == calls[0]["explanation"]
            assert payload["support_spans"] == calls[0]["support_spans"]
            assert payload["review_contract_feedback"]["previous_review"] == broken.model_dump()
        value = (
            broken
            if len(calls) == 1
            else {"pass": good, "reject": rejected, "contradiction": broken}[second]
        )
        return response(completion(value.model_dump()))

    model = gateway(settings, handler)
    try:
        if second == "contradiction":
            with pytest.raises(ModelUnavailable) as error:
                model.review_grounded_answer("A question", [], [source], draft, plan())
            assert error.value.diagnostic_code == "invalid_answer_review"
        else:
            result = model.review_grounded_answer("A question", [], [source], draft, plan())
            assert result.verification.supported is (second == "pass")
        assert len(calls) == 2
    finally:
        model.close()


@pytest.mark.parametrize(
    "claim,source_text,bad",
    [
        ("مدیر می‌پرسد: «چه کمکی لازم داری؟»", "مدیر می‌پرسد: «چه کمکی لازم داری؟»", False),
        ("در مثال، مدیر می‌پرسد: «چه کمکی لازم داری؟»", "مدیر می‌گوید: کمی استراحت کن.", True),
        ("مدیر می‌گوید: «باید کار را ادامه دهید.»", "نباید کار را ادامه دهید.", True),
        ("The manager asks: “What help do you need?”", "The manager offers a break.", True),
        ("The manager asks: “What help do you need?”", "مدیر می‌پرسد چه کمکی لازم داری؟", False),
        ("مدیر می‌گوید: «لطفاً با تيم تماس بگيريد!»", "لطفا با تیم تماس بگیرید.", False),
        ('The model is called "two-way filter".', "A model explanation.", False),
        ("The manager did not say: “Go home right now.”", "The manager offers a break.", False),
    ],
)
def test_literal_quotes_use_own_source_keep_negation_and_allow_translation(claim, source_text, bad):
    source = excerpt().model_copy(update={"text": source_text})
    assert bool(unsupported_literal_quotes(claim, [source])) is bad


def test_quote_in_different_source_does_not_validate_wrong_example_attribution():
    rule = excerpt().model_copy(update={"id": "rule", "text": "مدیر بپرسید: «چه کمکی لازم داری؟»"})
    example = excerpt().model_copy(update={"id": "example", "text": "مدیر اجازه استراحت می‌دهد."})
    draft = Explanation(
        statements=[
            {"text": "در مثال، مدیر می‌پرسد: «چه کمکی لازم داری؟»", "citation_ids": [example.id]}
        ]
    )
    review = span_review(draft, [example])
    assert not review_supports_statements(review, draft, [rule, example])


def test_consistent_rejected_optional_claim_can_be_pruned_but_required_claim_cannot():
    source = excerpt()
    draft = Explanation(
        statements=[
            {"text": "The requested fact.", "citation_ids": [source.id]},
            {"text": "An unnecessary unsupported benefit.", "citation_ids": [source.id]},
        ]
    )
    review = span_review(draft, [source])
    review.verification.supported = False
    review.rejection_reasons = ["Statement 1 adds an unsupported benefit."]
    review.statements[1].supported = False
    review.statements[1].span_ids = []
    review.verification.coverage[0].statement_indices = [0]
    assert not review_contract_issues(review, 2, 1)
    pruned = prune_incidental_statements(draft, review, [source], plan())
    assert pruned.statements == draft.statements[:1]
    review.verification.coverage[0].statement_indices = [0, 1]
    assert prune_incidental_statements(draft, review, [source], plan()) is None


def test_explicit_rerun_selection_preserves_references_and_rejects_unknown_ids():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_natural_benchmark import explicit_selection

    cases = [{"id": "natural-001", "question": "One?"}, {"id": "natural-002", "question": "Two?"}]
    selected, provenance = explicit_selection("natural-002", cases)
    assert selected == cases[1:] and selected[0] is cases[1]
    assert provenance["selected_case_ids"] == ["natural-002"]
    for value in ("", "natural-003", "natural-001,natural-001"):
        with pytest.raises(ValueError):
            explicit_selection(value, cases)


def test_conservative_aggregate_repair_can_only_reject_never_approve():
    draft, source = explanation(), excerpt()
    review = span_review(draft, [source])
    assert conservative_rejection(review, []) is None
    review.verification.supported = False
    assert conservative_rejection(review, review_contract_issues(review, 1, 1)) is None
    review.verification.supported = True
    review.statements[0].supported = False
    review.statements[0].span_ids = []
    result = conservative_rejection(review, review_contract_issues(review, 1, 1))
    assert result.verification.supported is False
    assert result.statements == review.statements and result.rejection_reasons
    assert not review_contract_issues(result, 1, 1)
    assert review.verification.supported is True


def test_second_contradictory_positive_aggregate_preserves_detailed_rejection(settings):
    draft, source, calls = explanation(), excerpt(), []
    rejected = span_review(draft, [source])
    rejected.statements[0].supported = False
    rejected.statements[0].span_ids = []

    def handler(request):
        calls.append(request)
        return response(completion(rejected.model_dump()))

    model = gateway(settings, handler)
    try:
        result = model.review_grounded_answer("A question", [], [source], draft, plan())
        assert len(calls) == 2 and not result.verification.supported
        assert result.statements == rejected.statements
    finally:
        model.close()


def test_inconsistent_reviewer_error_does_not_erase_independent_sibling(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    other = "مدل ۳۰-۳۰-۳۰ چیست؟"
    install_route(model, [route_group(QUESTION), route_group(other)])
    install_explainer(model)

    def review(question, history, sources, draft, plan, **kwargs):
        if question == QUESTION:
            raise ModelUnavailable("Inconsistent answer reviewer", code="invalid_answer_review")
        return span_review(draft, sources)

    model.review_grounded_answer = review
    result = client.post(
        BASE + "/query", headers=token(), json={"question": QUESTION + " " + other}
    )
    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "partial" and len(body["parts"]) == 2
    assert body["parts"][0]["status"] == "error"
    assert body["parts"][0]["reason_code"] == "invalid_answer_review"
    assert body["parts"][1]["status"] == "answered"
