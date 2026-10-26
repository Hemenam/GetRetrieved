"""V9 source/intent contracts. All inference uses fakes or MockTransport, never paid APIs."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
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
from test_retrieval_modes import TestRanker
from test_routing_v8 import install_route, route_group

from hrlearnium.grounding import review_supports_statements
from hrlearnium.models import OllamaGateway
from hrlearnium.request_planning import canonical_route
from hrlearnium.retrieval import Candidate
from hrlearnium.schemas import Explanation, QuestionPlan, RoutePlan, Selection, SpanAnswerReview
from hrlearnium.source_spans import source_spans


def span_review(draft, sources, **intent_changes):
    spans = source_spans(sources)
    return SpanAnswerReview.model_validate(
        {
            "verification": {
                "supported": True,
                "all_question_parts_identified": True,
                "coverage": [
                    {
                        "requirement_index": 0,
                        "covered": True,
                        "statement_indices": list(range(len(draft.statements))),
                    }
                ],
            },
            "intent": {
                "answers_original_question": True,
                "actor_action_preserved": True,
                "source_stance_preserved": True,
                "causal_claims_supported": True,
                **intent_changes,
            },
            "statements": [
                {
                    "statement_index": i,
                    "supported": True,
                    "span_ids": [
                        next(s["id"] for s in spans if s["citation_id"] in statement.citation_ids)
                    ],
                }
                for i, statement in enumerate(draft.statements)
            ],
        }
    )


@pytest.mark.parametrize(
    "text",
    [
        "روزهای سلامت روان: \nمرخصی یک‌روزه.\r\nمتن بعدی.",
        "خط اول. \\nاین یک بک‌اسلش واقعی است.\n" * 70,
        "الف" * 1901,
        "\n" * 1250 + "متن",
        "متن...\nنظر منتقدانه؛ سپس توضیح! " * 80,
    ],
    ids=["real-newlines", "literal-slashes", "long-line", "blank-lines", "punctuation"],
)
def test_server_spans_are_bounded_exact_and_lossless_for_nonblank_text(text):
    source = excerpt().model_copy(update={"text": text})
    spans = source_spans([source])
    assert spans == source_spans([source])
    assert len({s["id"] for s in spans}) == len(spans)
    assert all(s["text"] == text[s["start"] : s["end"]] and len(s["text"]) <= 600 for s in spans)
    assert "".join(s["text"] for s in spans).strip() == text.strip()
    changed = source.model_copy(update={"text": text + "new"})
    assert not {s["id"] for s in spans} & {s["id"] for s in source_spans([changed])}


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "unknown_span",
        "foreign_source",
        "duplicate_span",
        "empty_spans",
        "missing_statement",
        "duplicate_statement",
        "unsupported_statement",
        "answers_original_question",
        "actor_action_preserved",
        "source_stance_preserved",
        "causal_claims_supported",
    ],
)
def test_real_source_ids_do_not_bypass_ownership_and_meaning_checks(fault):
    source = excerpt()
    other = source.model_copy(update={"id": "another-course-source"})
    draft = explanation()
    review = span_review(draft, [source])
    if fault == "unknown_span":
        review.statements[0].span_ids = ["made-up"]
    elif fault == "foreign_source":
        review.statements[0].span_ids = [source_spans([other])[0]["id"]]
    elif fault == "duplicate_span":
        review.statements[0].span_ids *= 2
    elif fault == "empty_spans":
        review.statements[0].span_ids = []
    elif fault == "missing_statement":
        review.statements = []
    elif fault == "duplicate_statement":
        review.statements *= 2
    elif fault == "unsupported_statement":
        review.statements[0].supported = False
    elif fault != "none":
        setattr(review.intent, fault, False)
    assert review_supports_statements(review, draft, [source, other]) == (fault == "none")
    assert (
        "quote"
        not in SpanAnswerReview.model_json_schema()["$defs"]["SpanStatementSupport"]["properties"]
    )
    bad = span_review(draft, [source]).model_dump()
    bad["statements"][0]["quote"] = "Invented source quote"
    with pytest.raises(ValidationError):
        SpanAnswerReview.model_validate(bad)


@pytest.mark.parametrize("backend", ["hosted", "json_object", "ollama"])
def test_shipped_span_review_wire_schema_and_repair(settings, backend):
    source, draft = excerpt(), explanation()
    context = source.model_copy(update={"id": "context-only"})
    plan = QuestionPlan(search_query="A question", persian_query=None, requirements=["A question"])
    expected, calls = span_review(draft, [source]), []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        if len(calls) == 1:
            assert payload["candidate_context"][0]["id"] == context.id
            assert payload["support_spans"] == [
                {k: s[k] for k in ("id", "citation_id", "text")} for s in source_spans([source])
            ]
            if backend == "hosted":
                schema = body["response_format"]["json_schema"]["schema"]
                assert_strict_schema(schema)
            elif backend == "ollama":
                schema = body["format"]
            else:
                schema = json.loads(body["messages"][0]["content"].split("JSON schema:\n", 1)[1])
            assert schema["$defs"]["SpanStatementSupport"]["properties"]["span_ids"]["items"][
                "enum"
            ] == [s["id"] for s in source_spans([source])]
            value = expected.model_dump()
        else:
            assert payload["verification"] == expected.model_dump()
            value = draft.model_dump()
        return (
            response(
                {"done": True, "done_reason": "stop", "message": {"content": json.dumps(value)}}
            )
            if backend == "ollama"
            else response(completion(value))
        )

    settings.api_structured_output = "json_object" if backend == "json_object" else "json_schema"
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
        review = model.review_grounded_answer(
            "A question", [], [source], draft, plan, context_excerpts=[context]
        )
        repaired = model.repair_explanation("A question", [], [source], draft, plan, review)
        assert review == expected and repaired == draft and len(calls) == 2
    finally:
        model.close()


@pytest.mark.parametrize("mode", ["full_context", "hybrid", "hybrid_rerank"])
@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_wrong_direction_answer_is_repaired_or_withheld_never_published(
    api_factory, mode, repair_succeeds
):
    client, _, model = api_factory(retrieval_mode=mode, reranker=TestRanker())
    upload(client)
    install_route(model)
    install_explainer(model)
    reviews, repairs = [], []

    def verify(q, h, sources, draft, plan, **kwargs):
        assert kwargs["context_excerpts"]
        reviews.append(draft)
        return span_review(
            draft, sources, actor_action_preserved=repair_succeeds and len(reviews) > 1
        )

    def repair(q, h, sources, draft, plan, feedback):
        assert not feedback.intent.actor_action_preserved
        repairs.append(sources)
        return draft

    model.review_grounded_answer, model.repair_explanation = verify, repair
    result = client.post(
        BASE + "/query", headers=token(), json={"question": QUESTION, "include_evaluation": True}
    )
    assert result.status_code == (200 if repair_succeeds else 503)
    assert len(reviews) == 2 and len(repairs) == 1
    if repair_succeeds:
        assert [a["accepted"] for a in result.json()["evaluation"]["answer_attempts"]] == [
            False,
            True,
        ]
    else:
        assert "پیام، واقعیت" not in result.text


@pytest.mark.parametrize(
    "question, title",
    [
        ("آخرین فصلِ این مستند چه عنوانی دارد؟", "افق زمانی"),
        ("عنوان فصل آخر این دوره چیست؟", "افق زمانی"),
        ("What is the last chapter called?", "افق زمانی"),
        ("What is the title of the first chapter in this document?", "پیام اضطراری"),
        ("اولین فصل این دوره چه نامی دارد؟", "پیام اضطراری"),
    ],
    ids=["last-fa-original", "last-fa", "last-en", "first-en", "first-fa"],
)
@pytest.mark.parametrize("mode", ["full_context", "hybrid", "hybrid_rerank"])
def test_relative_title_uses_current_outline_without_llm(api_factory, mode, question, title):
    client, _, model = api_factory(retrieval_mode=mode, reranker=TestRanker())
    upload(client)
    model.plan = model.embed = lambda *a, **kw: pytest.fail("Metadata needs no LLM")
    body = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert body["status"] == "answered" and body["answer_kind"] == "metadata"
    assert title in body["answer"] and body["metadata"]["field"] == "chapter_title"


def test_relative_structural_guard_preserves_multipart_and_does_not_swallow_content(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    question = "آخرین فصل این مستند چه عنوانی دارد؟"
    route = canonical_route(
        RoutePlan(groups=[route_group(question), route_group(QUESTION)]),
        question + " و " + QUESTION,
    )
    assert route.groups[0].intent == "metadata" and route.groups[0].metadata_fields == [
        "last_chapter_title"
    ]
    assert route.groups[1].intent == "content"
    install_route(model, route.groups)
    install_explainer(model)
    body = client.post(
        BASE + "/query", headers=token(), json={"question": question + " و " + QUESTION}
    ).json()
    assert body["status"] == "answered" and [p["answer_kind"] for p in body["parts"]] == [
        "metadata",
        "content",
    ]
    from hrlearnium.course_metadata import metadata_request

    for text in (
        "آخرین فصل درباره همدلی چه می‌گوید؟",
        "What does the last chapter teach?",
        question + " و " + QUESTION,
    ):
        assert metadata_request(text) is None


def test_multiple_documents_have_no_invented_global_last_chapter(api_factory):
    client, _, model = api_factory()
    upload(client)
    upload(client, paragraphs=["ضمیمه", "فصل اول: دیگر", "توضیح دیگر"])
    model.plan = lambda *a, **kw: pytest.fail("No model can establish a missing global order")
    body = client.post(
        BASE + "/query", headers=token(), json={"question": "What is the last chapter called?"}
    ).json()
    assert body["status"] == "refused" and body["reason_code"] == "metadata_unavailable"


@pytest.mark.parametrize("excluded", [False, True])
def test_hybrid_repair_recovers_rank14_but_never_bypasses_relevance_gate(
    api_factory, monkeypatch, excluded
):
    from test_api import COURSE

    client, app, model = api_factory(
        retrieval_mode="hybrid", candidate_limit=2, rerank_pool_limit=16
    )
    paragraphs = ["دوره مصنوعی"]
    for i in range(1, 17):
        paragraphs += [
            f"فصل {i}: مبحث {i}",
            f"مبحث {i} توضیح مرتبطی درباره رفتار مدیر در بحران دارد.",
        ]
    upload(client, paragraphs=paragraphs)
    upload(
        client,
        headers=token(tenant="foreign"),
        paragraphs=["دوره خارجی", "فصل 1: خصوصی", "نباید وارد زمینه شود."],
    )
    sources, _ = app.state.storage.search_passages("tenant-1", COURSE)
    sources = sorted(sources, key=lambda p: p.chapter_number)
    target = sources[13]
    ranking = [
        Candidate(
            p,
            0 if excluded and p.id == target.id else 2,
            0 if excluded and p.id == target.id else 0.8,
            1 / (i + 1),
        )
        for i, p in enumerate(sources)
    ]
    monkeypatch.setattr("hrlearnium.service.fuse_query_candidates", lambda rankings: ranking)
    install_route(model)
    install_explainer(model)
    model.selection = lambda candidates: Selection(
        status="answered", reason_code="supported", passage_ids=[candidates[0].passage.id]
    )
    seen = []

    def review(q, h, own, draft, plan, **kwargs):
        context = kwargs["context_excerpts"]
        seen.append([e.id for e in context])
        assert set(seen[-1]) <= {p.id for p in sources}
        return span_review(
            draft, own, actor_action_preserved=target.id in draft.statements[0].citation_ids
        )

    def repair(q, h, context, draft, plan, feedback):
        assert not feedback.intent.actor_action_preserved
        chosen = next((e for e in context if e.id == target.id), context[0])
        return Explanation(
            statements=[{"text": "رفتار مدیر به اضطراب او مربوط است.", "citation_ids": [chosen.id]}]
        )

    model.review_grounded_answer, model.repair_explanation = review, repair
    result = client.post(
        BASE + "/query",
        headers=token(),
        json={"question": "علت رفتار مدیر در بحران چیست؟", "include_evaluation": True},
    )
    assert result.status_code == (503 if excluded else 200)
    assert len(seen) == 2 and ((target.id in seen[0]) is not excluded)
    if not excluded:
        body = result.json()
        assert len(body["evaluation"]["selector_candidates"]) == 2
        assert len(body["evaluation"]["answer_recovery_candidates"]) == 16
        assert [e["id"] for e in body["excerpts"]] == [target.id]


def test_relative_title_tracks_revisions_and_missing_headings(api_factory):
    from test_api import docx

    client, _, model = api_factory()
    document = upload(client).json()
    model.plan = lambda *a, **kw: pytest.fail("Do not infer chapter metadata")
    question = "What is the last chapter called?"
    first = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert first["metadata"]["chapter_number"] == 2
    response = client.put(
        BASE + f"/documents/{document['id']}",
        headers=token(),
        files={"file": ("new.docx", docx(["دوره جدید", "فصل 7: فصل تازه", "متن تازه"]))},
    )
    assert response.status_code == 200
    second = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert second["metadata"]["chapter_number"] == 7 and "فصل تازه" in second["answer"]
    client.delete(BASE + f"/documents/{document['id']}", headers=token())
    third = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert third["status"] == "refused" and third["reason_code"] == "metadata_unavailable"


def test_private_course_span_coverage_and_relative_metadata(api_factory):
    document = os.getenv("HRLEARNIUM_TEST_DOCX")
    database = os.getenv("HRLEARNIUM_TEST_BENCHMARK_DB")
    if not document or not database:
        pytest.skip("Set both private course paths for read-only source regression")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from audit_chapter_coverage import read_source

    _, passages = read_source(Path(database), "customer-demo", "captain-storm")
    assert len(passages) == 44
    for passage in passages:
        spans = source_spans([SimpleNamespace(id=passage["id"], text=passage["text"])])
        assert spans and len(spans) <= 450
        assert "".join(s["text"] for s in spans).strip() == passage["text"].strip()
        assert all(s["text"] == passage["text"][s["start"] : s["end"]] for s in spans)
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client, data=Path(document).read_bytes())
    model.plan = lambda *a, **kw: pytest.fail(
        "The actual course title lookup must not invoke an LLM"
    )
    result = client.post(
        BASE + "/query", headers=token(), json={"question": "آخرین فصلِ این مستند چه عنوانی دارد؟"}
    )
    body = result.json()
    assert body["status"] == "answered" and body["answer_kind"] == "metadata"
    assert body["metadata"]["field"] == "chapter_title" and body["metadata"]["chapter_number"] == 11
