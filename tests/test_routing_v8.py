"""V8 regression contracts using synthetic sources and MockTransport; never paid inference."""

import json
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from test_api import BASE, COURSE, QUESTION, token, upload
from test_api import api_factory as api_factory
from test_hosted_gateway import assert_strict_schema, completion, gateway, response
from test_hosted_gateway import settings as settings
from test_response_modes import install_explainer
from test_retrieval_modes import TestRanker

from hrlearnium import evaluation
from hrlearnium.models import ModelUnavailable, OllamaGateway
from hrlearnium.policy import conversational_reply
from hrlearnium.request_planning import canonical_route, compile_route, course_routing_context
from hrlearnium.schemas import (
    Explanation,
    QueryResponse,
    RequestPlanVerification,
    RouteGroup,
    RoutePlan,
    Selection,
)


def route_group(text, *, fields=(), chapter=None, search=None, persian=None):
    return RouteGroup(
        text=text,
        intent="metadata" if fields else "content",
        metadata_fields=list(fields),
        chapter_number=chapter,
        search_query=search or text,
        persian_query=persian,
    )


def approved(**changes):
    return RequestPlanVerification(
        **{
            "all_parts_identified": True,
            "no_added_requests": True,
            "grouping_valid": True,
            "intents_valid": True,
            **changes,
        }
    )


def install_route(model, groups=None):
    seen = []

    def route(question, history, context, *, translate):
        seen.append((question, history, context, translate))
        return RoutePlan(groups=groups or [route_group(question)])

    model.route_request = route
    model.verify_route = lambda *args: approved()
    model.repair_route = lambda *a, **kw: pytest.fail("Valid route must not be repaired")
    model.plan_request = lambda *a, **kw: pytest.fail("V7 planner must not be called")
    return seen


def test_wire_schema_cannot_invent_requirements_or_pre_search_clarification():
    schema = RoutePlan.model_json_schema()
    assert set(schema["properties"]) == {"groups"}
    assert "requirements" not in schema["$defs"]["RouteGroup"]["properties"]
    with pytest.raises(ValidationError):
        RoutePlan.model_validate(
            {"groups": [route_group(QUESTION).model_dump()], "needs_clarification": True}
        )


@pytest.mark.parametrize("mode", ["full_context", "hybrid", "hybrid_rerank"])
@pytest.mark.parametrize(
    "question",
    [
        "در نمونهٔ پیام قطع سرورها، از کارکنان چه کاری خواسته شد؟",
        "وقتی اضطراب بالاست، تفکیک کدام دو مسئله سخت است؟",
        "پنج رفتار همدلانه را به ترتیب بگو.",
    ],
)
def test_clear_course_questions_reach_evidence_with_original_requirements(
    api_factory, mode, question
):
    client, _, model = api_factory(retrieval_mode=mode, reranker=TestRanker())
    upload(client)
    seen = install_route(model)
    install_explainer(model)
    result = client.post(
        BASE + "/query", headers=token(), json={"question": question, "include_evaluation": True}
    )
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == "answered"
    assert seen[0][2]["documents"] and seen[0][2]["documents"][0]["chapters"]
    assert body["evaluation"]["question_plan"]["requirements"] == [question]
    assert model.selections[-1][0] == question
    assert body["evaluation"]["request_plan"]["needs_clarification"] is False


@pytest.mark.parametrize("failure", ["schema", "copy", "verification"])
def test_route_faults_fall_back_to_the_whole_original_request_not_learner_clarification(
    api_factory, failure
):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    install_explainer(model)
    question = QUESTION + " و مدل ۳۰-۳۰-۳۰ چیست؟"
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        if failure == "schema":
            raise ModelUnavailable("secret provider body", code="structured_response_invalid")
        if failure == "copy":
            return RoutePlan(
                groups=[route_group("fabricated first"), route_group("fabricated second")]
            )
        return RoutePlan(groups=[route_group(question)])

    model.route_request = model.repair_route = broken
    model.verify_route = lambda *a: approved(no_added_requests=False)
    response_body = client.post(
        BASE + "/query", headers=token(), json={"question": question, "include_evaluation": True}
    )
    assert response_body.status_code == 200, response_body.text
    body = response_body.json()
    assert len(calls) == 2 and len(body["evaluation"]["routing_attempts"]) == 2
    assert body["evaluation"]["routing_fallback"]["strategy"] == "whole_original_request"
    assert body["evaluation"]["question_plan"]["requirements"] == [question]
    assert model.selections[-1][0] == question and "secret" not in response_body.text


def test_provider_auth_failure_is_not_retried_or_hidden_by_route_fallback(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)

    def broken(*args, **kwargs):
        raise ModelUnavailable("secret", code="provider_http_error", provider_status=401)

    model.route_request = broken
    model.repair_route = lambda *a, **kw: pytest.fail("Do not retry permanent provider failures")
    response_body = client.post(BASE + "/query", headers=token(), json={"question": QUESTION})
    assert response_body.status_code == 503 and not model.selections
    assert "secret" not in response_body.text


def test_missing_source_and_real_ambiguity_are_decided_after_search(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    install_route(model)
    for question, status, reason in [
        ("پایتخت فرانسه کجاست؟", "refused", "outside_scope"),
        ("فرق این دو روش چیه؟", "clarification", "ambiguous"),
    ]:
        model.selection = lambda candidates, s=status, r=reason: Selection(status=s, reason_code=r)
        body = client.post(
            BASE + "/query",
            headers=token(),
            json={"question": question, "include_evaluation": True},
        ).json()
        assert body["status"] == status and body["reason_code"] == reason
        if status == "clarification":
            assert "selector_candidates" not in body["evaluation"]
        else:
            assert body["evaluation"]["selector_candidates"]
            assert model.selections[-1][0] == question
        assert body["excerpts"] == []


@pytest.mark.parametrize("fields", [["chapter_count"], ["chapter_count", "chapter_titles"]])
def test_count_and_titles_never_loses_the_listing_in_mixed_question(api_factory, fields):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    metadata = "دوره چند فصل دارد و اسم همهٔ فصل‌ها چیست؟"
    content = "پیام اضطراری چیست؟"
    question = metadata + " " + content
    install_route(model, [route_group(metadata, fields=fields), route_group(content)])
    install_explainer(model)
    body = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert body["status"] == "answered" and len(body["parts"]) == 2
    assert body["parts"][0]["metadata"]["field"] == "chapter_titles"
    assert (
        "پیام اضطراری" in body["parts"][0]["answer"] and "افق زمانی" in body["parts"][0]["answer"]
    )
    assert body["parts"][1]["explanation"]


def test_multiple_metadata_fields_and_unknown_admin_information(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    question = "عنوان و مدت دوره را بگو."
    install_route(model, [route_group(question, fields=["course_title", "duration"])])
    body = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert body["status"] == "partial"
    assert [p["status"] for p in body["parts"]] == ["answered", "refused"]
    assert body["parts"][0]["metadata"]["field"] == "course_title" and not body["excerpts"]
    question = "این دوره گواهی دانشگاهی می‌دهد؟"
    install_route(model, [route_group(question, fields=["other"])])
    body = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert body["status"] == "refused" and body["metadata"] is None and not model.selections


def test_normalization_restores_source_spelling_but_never_paraphrases_or_drops_a_clause():
    first, second = "مثبت‌اندیشی سمی چیست؟", "فصل ۱ چه عنوانی دارد؟"
    original = first + " " + second
    route = canonical_route(
        RoutePlan(
            groups=[
                route_group("مثبت اندیشی سمی چیست"),
                route_group("فصل 1 چه عنوانی دارد", fields=["chapter_title"], chapter=1),
            ]
        ),
        original,
    )
    for group in route.groups:
        assert group.text in original
    compiled = compile_route(route, "fa", translate=False)
    assert compiled.groups[0].requirements == [route.groups[0].text]
    with pytest.raises(ValueError, match="unaligned"):
        canonical_route(
            RoutePlan(groups=[route_group("invented topic"), route_group(second)]), original
        )
    # Even a shortened single group is made responsible for the COMPLETE student message.
    restored = canonical_route(RoutePlan(groups=[route_group(first)]), original)
    assert restored.groups[0].text == original


def test_routing_context_is_bounded_and_contains_only_authorized_outline(api_factory):
    client, app, _ = api_factory()
    upload(client)
    outline = app.state.storage.course_outline("tenant-1", COURSE)
    context = course_routing_context(outline, limit=1)
    assert context["truncated"] and not context["documents"]
    context = course_routing_context(outline)
    assert set(context) == {"documents", "truncated"} and len(context["documents"]) == 1
    assert "original_docx" not in json.dumps(context) and "source_sha256" not in json.dumps(context)


@pytest.mark.parametrize("mode", ["full_context", "hybrid", "hybrid_rerank"])
def test_rejected_draft_is_repaired_once_using_retrieved_not_foreign_evidence(api_factory, mode):
    client, app, model = api_factory(retrieval_mode=mode, reranker=TestRanker())
    upload(client)
    install_route(model)
    install_explainer(model)
    verifications, repairs = [], []
    original_verify = model.verify_complete

    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        verifications.append(1)
        result.supported = len(verifications) > 1
        return result

    model.verify_complete = verify

    def repair(q, history, pool, draft, plan, verification):
        repairs.append(1)
        assert plan.requirements == [q] and not verification.supported
        assert {e.id for e in pool} <= {
            p.id for p in app.state.storage.search_passages("tenant-1", COURSE)[0]
        }
        return Explanation(
            statements=[
                {
                    "text": "پیام شامل واقعیت، اقدام سازمان و گام بعدی کارکنان است.",
                    "citation_ids": [pool[0].id],
                }
            ]
        )

    model.repair_explanation = repair
    body = client.post(
        BASE + "/query", headers=token(), json={"question": QUESTION, "include_evaluation": True}
    ).json()
    assert body["status"] == "answered" and len(repairs) == 1 and len(verifications) == 2
    attempts = body["evaluation"]["answer_attempts"]
    assert [a["accepted"] for a in attempts] == [False, True]
    assert "answer_repair" in body["evaluation"]["stages_ms"]


@pytest.mark.parametrize(
    "failure", ["still_unverified", "foreign_citation", "over_budget", "language"]
)
def test_answer_repair_remains_fail_closed(api_factory, failure):
    client, app, model = api_factory(retrieval_mode="full_context")
    upload(client)
    upload(client, headers=token(tenant="foreign"))
    foreign_id = app.state.storage.search_passages("foreign", COURSE)[0][0].id
    install_route(model)
    install_explainer(model)
    model.verify_explanation = lambda *a: False
    calls = []

    def repair(q, history, pool, draft, plan, check):
        calls.append(1)
        ids = [foreign_id] if failure == "foreign_citation" else [e.id for e in pool]
        if failure == "over_budget":
            app.state.service.settings.max_excerpts = 1
        return Explanation(
            statements=[
                {
                    "text": "This is still an English answer."
                    if failure == "language"
                    else "یک پیش‌نویس غیرقابل‌تأیید.",
                    "citation_ids": ids,
                }
            ]
        )

    model.repair_explanation = repair
    response_body = client.post(BASE + "/query", headers=token(), json={"question": QUESTION})
    assert response_body.status_code == 503 and len(calls) == 1
    assert foreign_id not in response_body.text and "پیش‌نویس" not in response_body.text


def test_failed_part_retains_supported_sibling_but_never_persists_failed_topic(api_factory):
    client, app, model = api_factory(retrieval_mode="full_context")
    upload(client)
    first, second = QUESTION, "مدل ۳۰-۳۰-۳۰ چیست؟"
    install_route(model, [route_group(first), route_group(second)])
    install_explainer(model)
    model.verify_explanation = lambda q, *a: q == first
    model.repair_explanation = lambda q, h, e, d, p, v: d
    body = client.post(
        BASE + "/query",
        headers=token(),
        json={"question": first + " " + second, "include_evaluation": True},
    ).json()
    assert body["status"] == "partial" and [p["status"] for p in body["parts"]] == [
        "answered",
        "error",
    ]
    assert body["parts"][1]["excerpt_ids"] == [] and body["parts"][1]["explanation"] is None
    assert body["parts"][1]["reason_code"] == "answer_verification_failed"
    assert app.state.storage.conversation_questions(
        "tenant-1", COURSE, "learner-1", body["conversation_id"], 3600
    ) == [first]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate import response_contract_checks

    checks = response_contract_checks(QueryResponse.model_validate(body), "explained")
    assert checks["answer_canonical_render"] and checks["control_message_valid"]


def test_failed_diagnostics_preserve_usage_without_exposing_it_in_http_error(api_factory):
    client, app, model = api_factory(retrieval_mode="full_context")
    upload(client)
    install_route(model)
    install_explainer(model)
    model.verify_explanation = lambda *a: False

    def repair(*args):
        evaluation.provider_usage(
            {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}, "fake", "chat"
        )
        return args[3]

    model.repair_explanation = repair
    from hrlearnium.schemas import Principal, QueryRequest

    with pytest.raises(ModelUnavailable) as caught:
        app.state.service.query(
            Principal(sub="trusted", tenant_id="tenant-1", course_ids=[COURSE], scopes=["query"]),
            COURSE,
            QueryRequest(question=QUESTION, include_evaluation=True),
        )
    assert caught.value.diagnostic_code == "answer_verification_failed"
    assert len(caught.value.evaluation_trace["provider_calls"]) == 1
    assert len(caught.value.evaluation_trace["failed_part_trace"]["answer_attempts"]) == 2
    body = client.post(
        BASE + "/query", headers=token(), json={"question": QUESTION, "include_evaluation": True}
    )
    assert (
        body.status_code == 503
        and "evaluation_trace" not in body.text
        and "provider_calls" not in body.text
    )


@pytest.mark.parametrize("backend", ["hosted", "json_object", "ollama"])
def test_actual_gateway_v8_wire_format_context_and_repair(settings, backend):
    question = "این فایل چه عنوانی دارد؟"
    expected = RoutePlan(groups=[route_group(question, fields=["course_title"])])
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        assert payload["question"] == question and payload["course_outline"] == {"documents": []}
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
        if len(calls) == 1:
            assert (
                payload["failure"] == "route_alignment_or_operation_invalid"
                and payload["proposed_route"] == expected.model_dump()
            )
        value = approved().model_dump() if len(calls) == 2 else expected.model_dump()
        calls.append(payload)
        if backend == "ollama":
            if len(calls) < 3:
                assert body["options"]["num_predict"] == 4096
            return response(
                {"done": True, "done_reason": "stop", "message": {"content": json.dumps(value)}}
            )
        return response(completion(value))

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
        route = model.route_request(question, [], {"documents": []}, translate=False)
        repaired = model.repair_route(
            question,
            [],
            {"documents": []},
            route,
            "route_alignment_or_operation_invalid",
            translate=False,
        )
        assert model.verify_route(question, [], {"documents": []}, repaired) == approved()
        assert route == repaired == expected and len(calls) == 3
    finally:
        model.close()


def test_extended_acknowledgement_is_not_forced_through_planning():
    assert conversational_reply("ممنون، خیلی کمک کرد")[0] == "acknowledgement"
    assert conversational_reply("ممنون، مدل ۳۰-۳۰-۳۰ چیست؟") is None
