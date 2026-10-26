"""Regressions from the v6 paid run. Controlled model outputs, no network/model spend."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_api import BASE, token, upload
from test_api import api_factory as api_factory
from test_response_modes import install_explainer

from hrlearnium import evaluation
from hrlearnium.course_metadata import metadata_request
from hrlearnium.language import (
    explanation_language_matches,
    infer_language,
    language_scope,
    response_context,
    response_language,
)
from hrlearnium.request_planning import validate_request_plan
from hrlearnium.schemas import (
    Explanation,
    RequestGroup,
    RequestPlan,
    RequestPlanVerification,
    Selection,
)


def group(
    text,
    *,
    metadata=None,
    chapter=None,
    kind="single",
    requirements=None,
    search=None,
    persian=None,
):
    return RequestGroup(
        original_text=text,
        kind=kind,
        intent="metadata" if metadata else "content",
        metadata_field=metadata,
        chapter_number=chapter,
        requirements=requirements or [text],
        search_query=search or text,
        persian_query=persian,
    )


def install_plan(model, groups, *, flags=None, needs_clarification=False):
    def plan(question, history, *, translate, target_language):
        return RequestPlan(
            answer_language=target_language, needs_clarification=needs_clarification, groups=groups
        )

    model.plan_request = plan
    model.verify_request_plan = lambda *args: RequestPlanVerification(
        **{
            "all_parts_identified": True,
            "no_added_requests": True,
            "grouping_valid": True,
            "intents_valid": True,
            **(flags or {}),
        }
    )
    # New service must not fall back to legacy rewrite/global-independent decisions.
    model.plan = model.verify_plan = lambda *args, **kwargs: pytest.fail("Legacy planner used")


@pytest.mark.parametrize(
    "text,target",
    [
        ("مدل RAG چیست؟", "fa"),
        ("How can a manager handle ترس during a crisis?", "en"),
        ("مدل چیست؟ Please answer in English.", "en"),
        ("Explain the model. فارسی پاسخ بده", "fa"),
        ("مدل چیست؟ به زبان انگلیسی جواب بده", "en"),
        ("What is the model? Please respond in Farsi.", "fa"),
        ("متن می‌گوید «Please answer in English». معنای این جمله چیست؟", "fa"),
        ("What does «ترس از بحران» mean?", "en"),
    ],
)
def test_language_is_determined_from_original_and_explicit_preference(text, target):
    assert infer_language(text) == target


def test_request_language_is_immutable_and_does_not_leak_between_threads():
    def check(index):
        message = "پرسش فارسی چیست؟" if index % 2 else "What is the model?"
        expected = "fa" if index % 2 else "en"
        with language_scope(message):
            assert response_language("Completely rewritten English query") == expected
            assert response_context("rewritten")["original_student_message"] == message
        assert response_language("English again") == "en"

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, range(20)))


@pytest.mark.parametrize(
    "text,target,valid",
    [
        ("This is an English answer to the question.", "fa", False),
        ("این یک پاسخ کاملاً فارسی است.", "en", False),
        (
            "پیام باید کوتاه باشد. This additional English sentence should not slip through.",
            "fa",
            False,
        ),
        ("تیم HR باید KPI را بررسی کند.", "fa", True),
        ("The source says «چه حسی داشتیم؟» and asks about feelings.", "en", True),
        ("The source says «این نقل قول در منبع نیست».", "en", False),
        ("The two values are ١ and ٢.", "en", True),
        ("۲۰٪", "en", True),
        ("THIS IS AN ENGLISH ANSWER IN UPPERCASE.", "fa", False),
    ],
)
def test_language_guard_ignores_only_source_quotes_titles_technical_acronyms_and_numbers(
    text, target, valid
):
    source = SimpleNamespace(
        id="s", text="چه حسی داشتیم؟", citation=SimpleNamespace(chapter_title="عنوان منبع")
    )
    explanation = Explanation(statements=[{"text": text, "citation_ids": ["s"]}])
    assert explanation_language_matches(explanation, [source], target) is valid


@pytest.mark.parametrize(
    "question,expected",
    [
        ("این دوره چند فصل دارد؟ Please answer in English.", "recognized chapters"),
        ("How many chapters does this course have? فارسی پاسخ بده", "فصل شناسایی"),
        ("سلام Please reply in English.", "Hi!"),
        ("Hi. فارسی پاسخ بده", "سلام!"),
    ],
)
def test_fixed_metadata_and_greetings_honor_original_language_preference(
    api_factory, question, expected
):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    model.plan_request = lambda *a, **kw: pytest.fail("Fast path must not require model")
    response = client.post(BASE + "/query", headers=token(), json={"question": question})
    assert response.status_code == 200 and expected in response.json()["answer"]


@pytest.mark.parametrize("mode", ["full_context", "hybrid", "hybrid_rerank"])
def test_all_chapters_paraphrase_uses_typed_metadata_without_matching_rewritten_words(
    api_factory, mode
):
    from test_retrieval_modes import TestRanker

    client, _, model = api_factory(retrieval_mode=mode, reranker=TestRanker())
    upload(client)
    question = "نام تمام فصل‌های این دوره چیست؟"
    assert metadata_request(question) is None  # The actual v6 failing paraphrase.
    install_plan(
        model, [group(question, metadata="chapter_titles", search="unrelated search wording")]
    )
    model.embed = model.select = lambda *a: pytest.fail("Metadata must bypass content search")
    body = client.post(
        BASE + "/query", headers=token(), json={"question": question, "include_evaluation": True}
    ).json()
    assert body["status"] == "answered" and body["answer_kind"] == "metadata"
    assert len(body["metadata"]["outline"]["documents"][0]["chapters"]) == 2
    assert body["metadata"]["field"] == "chapter_titles"
    assert body["evaluation"]["request_plan"]["groups"][0]["question_id"] == "q1"


def test_content_question_mentioning_chapter_is_not_forced_to_metadata(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    question = "فصل اول چه چیزی درباره پیام اضطراری آموزش می‌دهد؟"
    install_plan(model, [group(question)])
    install_explainer(model)
    body = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert body["status"] == "answered" and body["answer_kind"] == "content"
    assert body["excerpts"] and body["metadata"] is None


def test_language_repair_is_once_logged_citation_preserving_and_verified(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    question = "پیام اضطراری چیست؟"
    install_plan(
        model,
        [
            group(
                question,
                search="What is an emergency message?",
                requirements=["What are the parts?"],
            )
        ],
    )
    install_explainer(model)
    seen = []

    def english(q, history, excerpts):
        seen.append((q, response_context(q)))
        return Explanation(
            statements=[
                {
                    "text": "The message includes the fact, action and next step.",
                    "citation_ids": [excerpts[0].id],
                }
            ]
        )

    model.explain = english

    def repair(q, history, excerpts, answer):
        seen.append("repair")
        evaluation.provider_usage(
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            "fake",
            "chat/completions",
        )
        return Explanation(
            statements=[
                {
                    "text": "پیام شامل واقعیت، اقدام سازمان و گام بعدی است.",
                    "citation_ids": answer.statements[0].citation_ids,
                }
            ]
        )

    model.repair_language = repair

    def verify(q, history, excerpts, answer):
        assert "پیام شامل" in answer.statements[0].text
        return True

    model.verify_explanation = verify
    body = client.post(
        BASE + "/query", headers=token(), json={"question": question, "include_evaluation": True}
    ).json()
    assert body["status"] == "answered" and "پیام شامل" in body["answer"]
    assert seen[0][0] == question and seen[0][1]["target_language"] == "fa"
    assert seen.count("repair") == 1
    trace = body["evaluation"]
    assert trace["answer_language_check"]["repair_attempts"] == 1
    assert trace["provider_calls"][0]["stage"] == "language_repair"


@pytest.mark.parametrize("failure", ["language", "citation", "count", "meaning"])
def test_failed_language_repair_is_redacted_operational_error_not_course_refusal(
    api_factory, failure
):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    question = "پیام اضطراری چیست؟"
    install_plan(model, [group(question)])
    install_explainer(model)
    model.explain = lambda q, h, e: Explanation(
        statements=[
            {"text": "The message describes the situation clearly.", "citation_ids": [e[0].id]}
        ]
    )
    calls = []

    def repair(q, h, e, a):
        calls.append(1)
        statement = {
            "text": "هنوز یک پاسخ نامعتبر وجود دارد."
            if failure != "language"
            else "This is still an English answer.",
            "citation_ids": ["foreign-course-id"]
            if failure == "citation"
            else a.statements[0].citation_ids,
        }
        return Explanation(statements=[statement] * (2 if failure == "count" else 1))

    model.repair_language = repair
    if failure == "meaning":
        model.verify_explanation = lambda *a: False
    response = client.post(BASE + "/query", headers=token(), json={"question": question})
    assert response.status_code == 503 and len(calls) == 1
    assert "foreign-course-id" not in response.text and "پاسخ نامعتبر" not in response.text


@pytest.mark.parametrize(
    "kind,question,requirements",
    [
        ("single", "پنج قدم همدلی را به ترتیب بگو.", ["پنج قدم همدلی چیست؟"]),
        (
            "single",
            "حس کنترل و عاملیت را چگونه برگردانیم؟",
            ["چگونه حس کنترل و عاملیت را بازگردانیم؟"],
        ),
        (
            "joint",
            "پیام اضطراری و مدل ۳۰-۳۰-۳۰ را مقایسه کن.",
            ["پیام اضطراری چیست؟", "تفاوت آن با مدل ۳۰-۳۰-۳۰ چیست؟"],
        ),
    ],
)
def test_lists_synonyms_and_comparisons_stay_in_one_answer_group(
    api_factory, kind, question, requirements
):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    install_plan(model, [group(question, kind=kind, requirements=requirements)])
    body = client.post(
        BASE + "/query", headers=token(), json={"question": question, "response_mode": "verbatim"}
    ).json()
    assert body["status"] == "answered" and body["answer_kind"] == "content"
    assert len(model.selections) == 1 and model.selections[0][0] == question


def test_independent_groups_keep_original_text_and_supported_siblings(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    first, second = "پیام اضطراری چیست؟", "پایتخت استرالیا چیست؟"
    question = first + " و " + second
    install_plan(model, [group(first, search="English rewrite"), group(second)])
    install_explainer(model)
    original_select = model.select

    def select(q, h, c):
        return (
            Selection(status="refused", reason_code="outside_scope")
            if q == second
            else original_select(q, h, c)
        )

    model.select = select
    body = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert body["status"] == "partial"
    assert [p["question"] for p in body["parts"]] == [first, second]
    assert [p["status"] for p in body["parts"]] == ["answered", "refused"]
    assert body["parts"][1]["excerpt_ids"] == []


@pytest.mark.parametrize(
    "flag", ["all_parts_identified", "no_added_requests", "grouping_valid", "intents_valid"]
)
def test_unrepaired_plan_verifier_rejection_is_operational_without_search(api_factory, flag):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    question = "مدیر چگونه کنترل و عاملیت کارکنان را بازگرداند؟"
    install_plan(model, [group(question)], flags={flag: False})
    model.select = lambda *a: pytest.fail("Rejected plan must not execute")
    response = client.post(BASE + "/query", headers=token(), json={"question": question})
    assert response.status_code == 503
    assert "clarification" not in response.text


def test_plan_local_contract_blocks_invented_reordered_duplicate_and_wrong_target_groups():
    first, second = "پرسش اول چیست؟", "پرسش دوم چیست؟"
    question = first + second
    valid = RequestPlan(
        answer_language="fa", needs_clarification=False, groups=[group(first), group(second)]
    )
    assert validate_request_plan(valid, question, "fa", translate=False)
    for groups in [
        [group("سوال اختراع شده چیست؟")],
        [group(second), group(first)],
        [group(first), group(first)],
        [group(first, metadata="chapter_title")],
    ]:
        invalid = valid.model_copy(update={"groups": groups})
        assert not validate_request_plan(invalid, question, "fa", translate=False)
    assert not validate_request_plan(valid, question, "en", translate=False)
    assert not validate_request_plan(
        valid.model_copy(update={"groups": []}), question, "fa", translate=False
    )
    with pytest.raises(ValidationError):
        group(first, metadata="arbitrary_operation")


def test_three_part_metadata_bundle_keeps_outline_content_and_unknown_field_separate(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    questions = ["نام تمام فصل‌های این دوره چیست؟", "پیام اضطراری چیست؟", "نام شخص مدرس را بگو؟"]
    install_plan(
        model,
        [
            group(questions[0], metadata="chapter_titles"),
            group(questions[1]),
            group(questions[2], metadata="instructor"),
        ],
    )
    install_explainer(model)
    body = client.post(
        BASE + "/query",
        headers=token(),
        json={"question": " ".join(questions), "include_evaluation": True},
    ).json()
    assert body["status"] == "partial"
    assert [p["answer_kind"] for p in body["parts"]] == ["metadata", "content", "metadata"]
    assert [p["status"] for p in body["parts"]] == ["answered", "answered", "refused"]
    assert len(body["parts"][0]["metadata"]["outline"]["documents"][0]["chapters"]) == 2
    assert len(model.selections) == 1
    assert [g["question_id"] for g in body["evaluation"]["request_plan"]["groups"]] == [
        "q1",
        "q2",
        "q3",
    ]


def test_failed_group_does_not_save_successful_sibling_as_a_completed_request(api_factory):
    from hrlearnium.models import ModelUnavailable

    client, app, model = api_factory(retrieval_mode="full_context")
    upload(client)
    conversation = client.post(BASE + "/query", headers=token(), json={"question": "سلام"}).json()[
        "conversation_id"
    ]
    first, second = "پیام اضطراری چیست؟", "مدل زمانی چیست؟"
    install_plan(model, [group(first), group(second)])
    install_explainer(model)
    original = model.explain

    def fail(q, h, e):
        if q == second:
            raise ModelUnavailable("private provider failure")
        return original(q, h, e)

    model.explain = fail
    response = client.post(
        BASE + "/query",
        headers=token(),
        json={"question": first + " و " + second, "conversation_id": conversation},
    )
    assert response.status_code == 503
    assert (
        app.state.storage.conversation_questions(
            "tenant-1", "course-1", "learner-1", conversation, 3600
        )
        == []
    )


def test_private_course_paraphrase_returns_all_eleven_titles_without_paid_calls(api_factory):
    import os
    from pathlib import Path

    path = os.environ.get("HRLEARNIUM_TEST_DOCX")
    if not path:
        pytest.skip("Set HRLEARNIUM_TEST_DOCX for the v7 private course metadata regression")
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client, data=Path(path).read_bytes())
    question = "نام تمام فصل‌های این دوره چیست؟"
    install_plan(model, [group(question, metadata="chapter_titles")])
    model.select = lambda *a: pytest.fail("No content selection for typed chapter-list intent")
    body = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert body["status"] == "answered"
    chapters = body["metadata"]["outline"]["documents"][0]["chapters"]
    assert [c["number"] for c in chapters] == list(range(1, 12))
    assert "11 فصل" in body["answer"] and "LMS" in body["answer"]
