"""Exercise real structured gateway serialization/validation using MockTransport only."""

import json

import httpx
import pytest
from test_api import BASE, token, upload
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
from test_request_planning_v7 import group

from hrlearnium.language import language_scope
from hrlearnium.models import ModelUnavailable, OllamaGateway
from hrlearnium.schemas import Explanation, QuestionPlan, RequestPlan, RequestPlanVerification


@pytest.mark.parametrize("backend", ["hosted", "json_object", "ollama"])
def test_real_request_planner_and_verifier_use_typed_intents_fixed_language_and_groups(
    settings, backend
):
    question = "نام تمام فصل‌های این دوره چیست؟ و پیام اضطراری چیست؟"
    parts = ["نام تمام فصل‌های این دوره چیست؟", "پیام اضطراری چیست؟"]
    planned = RequestPlan(
        answer_language="fa",
        needs_clarification=False,
        groups=[
            group(parts[0], metadata="chapter_titles"),
            group(parts[1], search="English search hint"),
        ],
    )
    verified = RequestPlanVerification(
        all_parts_identified=True, no_added_requests=True, grouping_valid=True, intents_valid=True
    )
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        assert payload["question"] == question and question not in body["messages"][0]["content"]
        if backend == "ollama":
            schema = body["format"]
            if not calls:
                assert body["options"]["num_predict"] == 4096
        elif backend == "json_object":
            schema = json.loads(body["messages"][0]["content"].split("JSON schema:\n", 1)[1])
        else:
            schema = body["response_format"]["json_schema"]["schema"]
            assert_strict_schema(schema)
        if not calls:
            assert schema["properties"]["answer_language"]["enum"] == ["fa"]
            fields = schema["$defs"]["RequestGroup"]["properties"]
            assert fields["intent"]["enum"] == ["content", "metadata"]
            assert fields["persian_query"] == {"type": "null"}
        else:
            assert "independent" not in schema["properties"]
            assert payload["plan"] == planned.model_dump()
        value = planned.model_dump() if not calls else verified.model_dump()
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
        result = model.plan_request(question, [], translate=False, target_language="fa")
        check = model.verify_request_plan(question, [], result)
        assert result == planned and check == verified and len(calls) == 2
    finally:
        model.close()


@pytest.mark.parametrize("malformation", ["extra_root", "invalid_operation", "missing_group_field"])
def test_nonconforming_compatible_provider_cannot_supply_unbounded_operations(
    settings, malformation
):
    value = RequestPlan(
        answer_language="fa",
        needs_clarification=False,
        groups=[group("نام تمام فصل‌های دوره چیست؟", metadata="chapter_titles")],
    ).model_dump()
    if malformation == "extra_root":
        value["tenant_id"] = "other-tenant"
    elif malformation == "invalid_operation":
        value["groups"][0]["metadata_field"] = "arbitrary_execute"
    else:
        del value["groups"][0]["original_text"]
    model = gateway(settings, lambda request: response(completion(value)))
    try:
        with pytest.raises(ModelUnavailable):
            model.plan_request("question", [], translate=False, target_language="fa")
    finally:
        model.close()


@pytest.mark.parametrize("backend", ["hosted", "json_object", "ollama"])
def test_real_plan_repair_keeps_schema_original_message_and_controlled_failure(settings, backend):
    question = "مثبت‌اندیشی سمی چیست؟"
    proposed = RequestPlan(answer_language="fa", needs_clarification=True, groups=[])
    repaired = RequestPlan(
        answer_language="fa", needs_clarification=False, groups=[group(question)]
    )
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        assert payload == {
            "question": question,
            "previous_questions": [],
            "translate_to_persian": False,
            "target_language": "fa",
            "proposed_plan": proposed.model_dump(),
            "failure": "proposed_clarification_needs_review",
        }
        assert "Repair the proposed request plan once" in body["messages"][0]["content"]
        if backend == "ollama":
            schema = body["format"]
        elif backend == "json_object":
            schema = json.loads(body["messages"][0]["content"].split("JSON schema:\n", 1)[1])
        else:
            schema = body["response_format"]["json_schema"]["schema"]
            assert_strict_schema(schema)
        assert schema["properties"]["answer_language"]["enum"] == ["fa"]
        assert schema["$defs"]["RequestGroup"]["properties"]["persian_query"] == {"type": "null"}
        calls.append(payload)
        return (
            response(
                {
                    "done": True,
                    "done_reason": "stop",
                    "message": {"content": repaired.model_dump_json()},
                }
            )
            if backend == "ollama"
            else response(completion(repaired.model_dump()))
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
        result = model.repair_request_plan(
            question,
            [],
            proposed,
            "proposed_clarification_needs_review",
            translate=False,
            target_language="fa",
        )
        assert result == repaired and len(calls) == 1
    finally:
        model.close()


def test_language_repair_and_verification_receive_original_language_and_both_answers(settings):
    original = "پیام اضطراری چیست؟"
    source = excerpt()
    before = Explanation(
        statements=[
            {
                "text": "Security and initial communication are priorities.",
                "citation_ids": [source.id],
            }
        ]
    )
    repaired = explanation()
    plan = QuestionPlan(
        search_query="English rewritten search", persian_query=None, requirements=[original]
    )
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        assert payload["target_language"] == "fa"
        assert payload["original_student_message"] == original
        assert payload["question"] == original
        assert_strict_schema(body["response_format"]["json_schema"]["schema"])
        if not calls:
            assert payload["proposed_answer"] == before.model_dump()
            answer = repaired.model_dump()
        else:
            assert payload["original_explanation_to_preserve"] == before.model_dump()
            assert payload["explanation"] == repaired.model_dump()
            answer = {
                "supported": True,
                "all_question_parts_identified": True,
                "coverage": [{"requirement_index": 0, "covered": True, "statement_indices": [0]}],
            }
        calls.append(payload)
        return response(completion(answer))

    model = gateway(settings, handler)
    try:
        with language_scope(original):
            answer = model.repair_language(original, [], [source], before)
            checked = model.verify_complete(
                original, [], [source], answer, plan, language_repair_of=before
            )
        assert checked.supported and len(calls) == 2
    finally:
        model.close()


def test_http_api_executes_real_gateway_typed_plan_metadata_and_language_repair(
    settings, api_factory
):
    client, app, _ = api_factory(retrieval_mode="full_context")
    upload(client)
    metadata_question, content_question = "نام تمام فصل‌های این دوره چیست؟", "پیام اضطراری چیست؟"
    original = metadata_question + " و " + content_question
    routed = RequestPlan(
        answer_language="fa",
        needs_clarification=False,
        groups=[
            group(metadata_question, metadata="chapter_titles"),
            group(content_question, search="What are emergency message components?"),
        ],
    )
    calls = []
    source_id = None
    source_text = None

    def handler(request):
        nonlocal source_id, source_text
        body = json.loads(request.content)
        assert_strict_schema(body["response_format"]["json_schema"]["schema"])
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        if len(calls) == 1:
            value = {
                "groups": [
                    {
                        "text": g.original_text,
                        "intent": g.intent,
                        "metadata_fields": [g.metadata_field] if g.metadata_field else [],
                        "chapter_number": g.chapter_number,
                        "search_query": g.search_query,
                        "persian_query": g.persian_query,
                    }
                    for g in routed.groups
                ]
            }
        elif len(calls) == 2:
            value = {
                "all_parts_identified": True,
                "no_added_requests": True,
                "grouping_valid": True,
                "intents_valid": True,
            }
        elif len(calls) == 3:
            assert payload["question"] == content_question
            source_id = payload["passages"][0]["id"]
            source_text = payload["passages"][0]["text"]
            value = {
                "decision": {
                    "status": "answered",
                    "reason_code": "supported",
                    "source_ids": [source_id],
                    "coverage": {"r0": [0]},
                }
            }
        else:
            assert payload["target_language"] == "fa"
            assert payload["original_student_message"] == original
            assert payload["question"] == content_question
            if len(calls) == 4:
                value = {
                    "statements": [
                        {
                            "text": "The message includes a fact, action and next step.",
                            "citation_ids": [source_id],
                        }
                    ]
                }
            elif len(calls) == 5:
                assert "proposed_answer" in payload
                value = {
                    "statements": [
                        {
                            "text": "پیام اضطراری شامل واقعیت، اقدام سازمان و گام بعدی کارکنان است.",
                            "citation_ids": [source_id],
                        }
                    ]
                }
            else:
                assert len(calls) == 6 and "original_explanation_to_preserve" in payload
                value = {
                    "verification": {
                        "supported": True,
                        "all_question_parts_identified": True,
                        "coverage": [
                            {"requirement_index": 0, "covered": True, "statement_indices": [0]}
                        ],
                    },
                    "statements": [
                        {
                            "statement_index": 0,
                            "supported": True,
                            "span_ids": [payload["support_spans"][0]["id"]],
                        }
                    ],
                }
        if "support_spans" in payload:
            value["rejection_reasons"] = []
            value["intent"] = {
                "answers_original_question": True,
                "actor_action_preserved": True,
                "source_stance_preserved": True,
                "causal_claims_supported": True,
            }
            from test_answer_support_v9 import detailed_wire

            value = detailed_wire(value)
        reply = completion(value)
        reply["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return response(reply)

    model = gateway(settings, handler)
    app.state.service.gateway = model
    # Keep this legacy typed-plan adapter contract exercised independently from
    # the new indexed-unit/evidence-brief gateway path (covered by v10 tests).
    model.route_units = None
    model.prepare_evidence = None
    try:
        result = client.post(
            BASE + "/query",
            headers=token(),
            json={"question": original, "include_evaluation": True},
        )
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["status"] == "answered" and body["answer_kind"] == "multi"
        assert [p["answer_kind"] for p in body["parts"]] == ["metadata", "content"]
        assert "پیام اضطراری شامل" in body["parts"][1]["answer"]
        assert len(body["evaluation"]["provider_calls"]) == len(calls) == 6
        assert (
            sum(c["stage"] == "language_repair" for c in body["evaluation"]["provider_calls"]) == 1
        )
    finally:
        model.close()
