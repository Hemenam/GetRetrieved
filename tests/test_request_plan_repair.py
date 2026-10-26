"""Planning faults must not be presented as learner ambiguity; offline, no API spend."""

import pytest
from test_api import BASE, docx, token, upload
from test_api import api_factory as api_factory
from test_request_planning_v7 import group, install_plan
from test_response_modes import install_explainer

from hrlearnium.models import ModelUnavailable
from hrlearnium.schemas import RequestPlan, RequestPlanVerification

QUESTION = "مثبت‌اندیشی سمی چیست؟"


def plan(question=QUESTION, *, unclear=False):
    return RequestPlan(
        answer_language="fa",
        needs_clarification=unclear,
        groups=[] if unclear else [group(question)],
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


@pytest.mark.parametrize("mode", ["full_context", "hybrid", "hybrid_rerank"])
@pytest.mark.parametrize("fault", ["copy", "unnecessary_clarification", "verification", "schema"])
def test_named_definition_survives_one_planning_fault(api_factory, mode, fault):
    from test_retrieval_modes import TestRanker

    client, _, model = api_factory(retrieval_mode=mode, reranker=TestRanker())
    upload(
        client,
        data=docx(
            [
                "دوره مدیریت بحران",
                "فصل اول: مثبت‌اندیشی سمی",
                "مثبت‌اندیشی سمی یعنی نادیده گرفتن احساسات واقعی با اصرار به مثبت بودن.",
            ]
        ),
    )
    install_explainer(model)
    initial = (
        plan("مثبت اندیشی سمی چیست؟")
        if fault == "copy"
        else plan(unclear=fault == "unnecessary_clarification")
    )
    if fault == "schema":

        def broken(*args, **kwargs):
            raise ModelUnavailable("private provider payload", code="structured_response_invalid")

        model.plan_request = broken
    else:
        model.plan_request = lambda *a, **kw: initial
    verifications = []

    def verify(*args):
        verifications.append(args[-1])
        return (
            approved(grouping_valid=False)
            if fault == "verification" and len(verifications) == 1
            else approved()
        )

    model.verify_request_plan = verify
    repairs = []

    def repair(question, history, proposed, failure, *, translate, target_language):
        assert question == QUESTION and target_language == "fa" and not translate
        repairs.append((proposed, failure))
        return plan()

    model.repair_request_plan = repair
    response = client.post(
        BASE + "/query",
        headers=token(),
        json={
            "question": QUESTION,
            "include_evaluation": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered" and body["explanation"]["statements"]
    assert len(repairs) == 1 and len(verifications) == (2 if fault == "verification" else 1)
    trace = body["evaluation"]
    assert len(trace["request_plan_attempts"]) == 2
    assert trace["request_plan_attempts"][0]["failure"]
    assert trace["request_plan_attempts"][1]["failure"] is None
    assert "planning_repair" in trace["stages_ms"]
    assert "plan_repair_verification" in trace["stages_ms"]
    assert len(model.selections) == 1 and model.selections[0][0] == QUESTION


@pytest.mark.parametrize("failure", ["local", "verifier", "clarification_rejected", "schema"])
def test_failed_repair_is_bounded_redacted_and_does_not_retrieve_or_save(api_factory, failure):
    client, app, model = api_factory(retrieval_mode="full_context")
    upload(client)
    conversation = client.post(BASE + "/query", headers=token(), json={"question": "سلام"}).json()[
        "conversation_id"
    ]
    install_plan(model, [group("different invented question")])
    repairs = []

    def repair(*args, **kwargs):
        repairs.append(args)
        if failure == "schema":
            raise ModelUnavailable(
                "private secret provider body", code="structured_response_invalid"
            )
        return (
            plan("invented again")
            if failure == "local"
            else plan(unclear=failure == "clarification_rejected")
        )

    model.repair_request_plan = repair
    model.verify_request_plan = lambda *args: approved(intents_valid=False)
    model.select = lambda *args: pytest.fail("An unverified plan must not retrieve/select")
    response = client.post(
        BASE + "/query",
        headers=token(),
        json={"question": QUESTION, "conversation_id": conversation},
    )
    assert response.status_code == 503 and len(repairs) == 1
    assert "private" not in response.text and "clarification" not in response.text
    assert (
        app.state.storage.conversation_questions(
            "tenant-1", "course-1", "learner-1", conversation, 3600
        )
        == []
    )
    assert app.state.service._model_slots.acquire(blocking=False)
    app.state.service._model_slots.release()


def test_genuine_ambiguity_requires_repair_review_and_verifier_confirmation(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    question = "این موضوع را با آن روش دیگر مقایسه کن."
    unclear = plan(unclear=True)
    model.plan_request = lambda *a, **kw: unclear
    repairs, checks = [], []
    model.repair_request_plan = lambda *a, **kw: repairs.append(a) or unclear
    model.verify_request_plan = lambda *a: checks.append(a) or approved()
    model.select = lambda *a: pytest.fail("Genuine ambiguity must not execute")
    response = client.post(
        BASE + "/query", headers=token(), json={"question": question, "include_evaluation": True}
    )
    assert response.status_code == 200 and response.json()["status"] == "clarification"
    assert len(repairs) == len(checks) == 1
    assert response.json()["evaluation"]["request_plan_attempts"][-1]["failure"] is None


@pytest.mark.parametrize(
    "code", ["provider_timeout", "provider_http_error", "provider_transport_error"]
)
def test_transport_failures_do_not_trigger_semantic_repair(api_factory, code):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)

    def fail(*a, **kw):
        raise ModelUnavailable("private provider error", code=code)

    model.plan_request = fail
    model.repair_request_plan = lambda *a, **kw: pytest.fail("Do not compound transport retries")
    response = client.post(BASE + "/query", headers=token(), json={"question": QUESTION})
    assert response.status_code == 503 and "private" not in response.text


def test_accepted_plan_has_no_repair_and_metadata_routing_is_preserved(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    question = "نام تمام فصل‌های این دوره چیست؟"
    install_plan(model, [group(question, metadata="chapter_titles")])
    model.repair_request_plan = lambda *a, **kw: pytest.fail("Valid plan costs no repair call")
    model.select = lambda *a: pytest.fail("Metadata must bypass retrieval")
    response = client.post(BASE + "/query", headers=token(), json={"question": question})
    assert response.status_code == 200 and response.json()["answer_kind"] == "metadata"


@pytest.mark.parametrize("mode", ["full_context", "hybrid", "hybrid_rerank"])
@pytest.mark.parametrize("complete", [True, False])
def test_single_definition_can_have_multiple_evidence_requirements_without_repair(
    api_factory, mode, complete
):
    from test_retrieval_modes import TestRanker

    client, _, model = api_factory(retrieval_mode=mode, reranker=TestRanker())
    upload(
        client,
        data=docx(
            [
                "دوره مدیریت بحران",
                "فصل اول: مثبت‌اندیشی سمی",
                "مثبت‌اندیشی سمی یعنی نادیده گرفتن احساسات واقعی با اصرار به مثبت بودن.",
            ]
        ),
    )
    # The actual live failure: one question, several legitimate coverage points, not multipart.
    install_plan(
        model,
        [
            group(
                QUESTION,
                requirements=[
                    "مثبت‌اندیشی سمی چه معنایی دارد؟",
                    "ویژگی اصلی این مفهوم چیست؟",
                ],
            )
        ],
    )
    install_explainer(model)
    if not complete:
        original_verify = model.verify_complete

        def verify_incomplete(*args, **kwargs):
            result = original_verify(*args, **kwargs)
            result.coverage[1].covered = False
            result.coverage[1].statement_indices = []
            return result

        model.verify_complete = verify_incomplete
    model.repair_request_plan = lambda *a, **kw: pytest.fail(
        "Valid single-group plan needs no repair"
    )
    response = client.post(
        BASE + "/query",
        headers=token(),
        json={
            "question": QUESTION,
            "include_evaluation": True,
        },
    )
    if not complete:
        assert response.status_code == 503 and "not a course refusal" in response.json()["detail"]
        return
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer_kind"] == "content"
    assert not body["parts"]
    assert len(body["evaluation"]["request_plan_attempts"]) == 1
    assert len(body["evaluation"]["requirement_verification"]["coverage"]) == 2
    assert body["evaluation"]["requirement_verification"]["accepted"] is complete
