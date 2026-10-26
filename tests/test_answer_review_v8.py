"""Quote ownership and bounded correction are contracts, not live semantic quality claims."""

import json

import httpx
import pytest
from test_api import BASE, QUESTION, token, upload
from test_api import api_factory as api_factory
from test_hosted_gateway import (
    assert_strict_schema,
    completion,
    excerpt,
    explanation,
    gateway,
    response,
)
from test_hosted_gateway import settings as settings
from test_response_modes import install_explainer
from test_routing_v8 import install_route

from hrlearnium.grounding import review_supports_statements
from hrlearnium.language import language_scope
from hrlearnium.models import OllamaGateway
from hrlearnium.schemas import AnswerReview, Explanation, QuestionPlan, Selection


def review_for(draft, sources, *, supported=True):
    return AnswerReview.model_validate(
        {
            "verification": {
                "supported": supported,
                "all_question_parts_identified": True,
                "coverage": [
                    {
                        "requirement_index": 0,
                        "covered": supported,
                        "statement_indices": [0] if supported else [],
                    }
                ],
            },
            "statements": [
                {
                    "statement_index": i,
                    "supported": supported,
                    "evidence": [
                        {
                            "citation_id": s.citation_ids[0],
                            "quote": next(e.text for e in sources if e.id == s.citation_ids[0]),
                        }
                    ]
                    if supported
                    else [],
                }
                for i, s in enumerate(draft.statements)
            ],
        }
    )


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "invented_quote",
        "foreign_id",
        "non_cited_source",
        "missing_statement",
        "duplicate_statement",
        "unsupported",
        "whitespace_quote",
    ],
)
def test_global_true_never_bypasses_statement_span_checks(fault):
    source = excerpt()
    other = source.model_copy(update={"id": "another-source"})
    draft = Explanation(statements=[{"text": "Claim", "citation_ids": [source.id]}])
    review = review_for(draft, [source])
    if fault == "invented_quote":
        review.statements[0].evidence[0].quote = "This sentence does not occur in the source."
    elif fault == "foreign_id":
        review.statements[0].evidence[0].citation_id = "other-tenant-id"
    elif fault == "non_cited_source":
        review.statements[0].evidence[0].citation_id = other.id
    elif fault == "missing_statement":
        review.statements = []
    elif fault == "duplicate_statement":
        review.statements *= 2
    elif fault == "unsupported":
        review.statements[0].supported = False
    elif fault == "whitespace_quote":
        review.statements[0].evidence[0].quote = " "
    assert review.verification.supported is True
    assert review_supports_statements(review, draft, [source, other]) == (fault == "none")


def test_bad_quote_goes_through_repair_and_reverification_not_publication(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    install_route(model)
    install_explainer(model)
    reviews, repairs = [], []

    def verify(q, history, sources, draft, plan, **kwargs):
        result = review_for(draft, sources)
        reviews.append(result)
        if len(reviews) == 1:
            result.statements[0].evidence[0].quote = "a fabricated support quote"
        return result

    def repair(q, history, sources, draft, plan, feedback):
        assert isinstance(feedback, AnswerReview) and feedback.verification.supported
        assert feedback.statements[0].evidence[0].quote == "a fabricated support quote"
        repairs.append(1)
        return draft

    model.review_answer, model.repair_explanation = verify, repair
    body = client.post(
        BASE + "/query", headers=token(), json={"question": QUESTION, "include_evaluation": True}
    ).json()
    assert body["status"] == "answered" and len(reviews) == 2 and len(repairs) == 1
    assert [a["accepted"] for a in body["evaluation"]["answer_attempts"]] == [False, True]


def test_repair_can_select_complementary_evidence_from_the_existing_candidate_pool(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    install_route(model)
    install_explainer(model)
    seen = []

    def verify(q, history, sources, draft, plan, **kwargs):
        seen.append([e.id for e in sources])
        return review_for(draft, sources, supported=len(seen) > 1)

    def repair(q, h, pool, draft, plan, feedback):
        new_source = next(e for e in pool if e.id not in seen[0])
        return Explanation(
            statements=[{"text": "پاسخ تکمیل‌شده با شاهد مجاز.", "citation_ids": [new_source.id]}]
        )

    model.review_answer, model.repair_explanation = verify, repair
    # Only one candidate was initially selected, but both belong to this same course.
    model.selection = lambda candidates: Selection(
        status="answered", reason_code="supported", passage_ids=[candidates[0].passage.id]
    )
    body = client.post(BASE + "/query", headers=token(), json={"question": QUESTION}).json()
    assert body["status"] == "answered" and seen[0] != seen[1]
    assert [e["id"] for e in body["excerpts"]] == seen[1]


@pytest.mark.parametrize("backend", ["hosted", "json_object", "ollama"])
def test_real_review_and_answer_repair_gateways_use_original_context_and_constrained_ids(
    settings, backend
):
    question = "اولویت چیست؟"
    source, draft = excerpt(), explanation()
    plan = QuestionPlan(search_query=question, persian_query=None, requirements=[question])
    review = review_for(draft, [source])
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        assert payload["question"] == question and payload["original_student_message"] == question
        assert payload["target_language"] == "fa" and payload["excerpts"][0]["id"] == source.id
        schema = (
            body["format"]
            if backend == "ollama"
            else (
                json.loads(body["messages"][0]["content"].split("JSON schema:\n", 1)[1])
                if backend == "json_object"
                else body["response_format"]["json_schema"]["schema"]
            )
        )
        if backend == "hosted":
            assert_strict_schema(schema)
        if not calls:
            assert schema["$defs"]["SupportQuote"]["properties"]["citation_id"]["enum"] == [
                source.id
            ]
            value = review.model_dump()
        else:
            assert (
                payload["verification"] == review.model_dump()
                and payload["proposed_answer"] == draft.model_dump()
            )
            assert schema["$defs"]["GroundedStatement"]["properties"]["citation_ids"]["items"][
                "enum"
            ] == [source.id]
            value = draft.model_dump()
        calls.append(payload)
        return (
            response(
                {"done": True, "done_reason": "stop", "message": {"content": json.dumps(value)}}
            )
            if backend == "ollama"
            else response(completion(value))
        )

    if backend == "json_object":
        settings.api_structured_output = "json_object"
    model = (
        OllamaGateway(
            settings,
            client=httpx.Client(
                base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handler)
            ),
        )
        if backend == "ollama"
        else gateway(settings, handler)
    )
    try:
        with language_scope(question):
            actual = model.review_answer(question, [], [source], draft, plan)
            repaired = model.repair_explanation(question, [], [source], draft, plan, actual)
        assert actual == review and repaired == draft and len(calls) == 2
    finally:
        model.close()
