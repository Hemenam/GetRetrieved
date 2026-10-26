"""Course-independent request/evidence boundaries; no live inference or modified gold."""

import json
from pathlib import Path

import pytest
from test_answer_support_v9 import detailed_wire, span_review
from test_api import BASE, QUESTION, token, upload
from test_api import api_factory as api_factory
from test_hosted_gateway import assert_strict_schema, completion, gateway, response
from test_hosted_gateway import settings as settings

from hrlearnium.course_metadata import metadata_request
from hrlearnium.grounding import brief_selection, supplement_general_rule
from hrlearnium.models import ModelUnavailable
from hrlearnium.policy import is_bare_followup, policy_check
from hrlearnium.request_units import compile_units, request_units
from hrlearnium.retrieval import Candidate, SearchPassage
from hrlearnium.schemas import EvidenceBrief, Explanation, UnitRoutes
from hrlearnium.source_spans import source_spans


def test_bounded_brief_wire_cannot_exceed_sources_or_reference_unselected_indices():
    from pydantic import ValidationError

    from hrlearnium.selection_contract import brief_decision_model

    model = brief_decision_model(["a", "b", "c"], 2)
    valid = {
        "decision": {
            "status": "answered",
            "reason_code": "supported",
            "source_ids": ["a"],
            "facts": [{"text": "Fact", "source_indices": [0]}],
            "requested_values": [],
        }
    }
    assert model.model_validate(valid)
    valid["decision"]["facts"][0]["source_indices"] = [1]
    with pytest.raises(ValidationError):
        model.model_validate(valid)
    valid["decision"]["source_ids"] = ["a", "b", "c"]
    with pytest.raises(ValidationError):
        model.model_validate(valid)


@pytest.mark.parametrize(
    "value,expected",
    [(None, "refused"), ("سخنگو", "refused"), ("علی احمدی", "answered"), ("نام جعلی", "refused")],
)
def test_actual_names_not_role_templates_are_required(value, expected):
    c = Candidate(
        SearchPassage("s", "سخنگو: علی احمدی. باید یک سخنگو تعیین شود.", 1, "عنوان", "explanation"),
        1,
        0.8,
        1,
    )
    brief = EvidenceBrief(
        status="answered",
        reason_code="supported",
        complete=True,
        facts=[{"text": "پاسخ", "span_ids": [s["id"] for s in source_spans([c.passage])]}],
        requested_values=[] if value is None else [{"value": value, "source_id": "s"}],
    )
    assert (
        brief_selection(brief, [c], max_excerpts=1, question="اسم سخنگو چیست؟").selection.status
        == expected
    )


def test_universal_question_adds_eligible_same_chapter_rule_but_never_crosses_budget():
    example = Candidate(
        SearchPassage("example", "یک مثال با ساعت مشخص", 1, "پیام", "example"), 1, 0.8, 1
    )
    rule = Candidate(
        SearchPassage("rule", "زمان پیام بعدی را مشخص کنید.", 1, "پیام", "explanation"), 1, 0.8, 0.8
    )
    brief = EvidenceBrief(
        status="answered",
        reason_code="supported",
        complete=True,
        facts=[{"text": "مثال", "span_ids": [s["id"] for s in source_spans([example.passage])]}],
    )
    updated = supplement_general_rule(brief, [example, rule], "آیا همیشه همین ساعت است؟", 2)
    assert len(updated.facts) == 2 and updated.facts[-1].text == rule.passage.text
    assert (
        len(supplement_general_rule(brief, [example, rule], "آیا همیشه همین ساعت است؟", 1).facts)
        == 1
    )


def test_blanket_rule_replaces_intro_note_with_actual_operative_text_not_a_new_citation_on_old_claim():
    intro = Candidate(
        SearchPassage("intro", "هدف: آموزش یک چارچوب.", 3, "عنوان", "introduction"), 1, 0.8, 1
    )
    rule = Candidate(
        SearchPassage("rule", "فقط کارهای غیرضروری قابل تعویق هستند.", 3, "عنوان", "explanation"),
        1,
        0.8,
        1,
    )
    brief = EvidenceBrief(
        status="answered",
        reason_code="supported",
        complete=True,
        facts=[
            {
                "text": "یک ادعای فراتر از مقدمه",
                "span_ids": [s["id"] for s in source_spans([intro.passage])],
            }
        ],
    )
    changed = supplement_general_rule(brief, [intro, rule], "آیا همه کارها را عقب بیندازیم؟", 1)
    assert changed.facts[0].text == rule.passage.text
    assert brief_selection(changed, [intro, rule], max_excerpts=1).selection.passage_ids == ["rule"]


def test_mixed_chapter_exercise_does_not_prevent_supplementing_examples_own_rule():
    example = Candidate(SearchPassage("ex", "مثال اولویت‌بندی.", 2, "اولویت", "example"), 1, 0.8, 1)
    rule = Candidate(
        SearchPassage("rule", "کار حیاتی اولویت دارد.", 2, "اولویت", "explanation"), 1, 0.8, 1
    )
    exercise = Candidate(
        SearchPassage("practice", "کار غیرضروری را عقب بیندازید.", 9, "تمرین", "exercise"),
        1,
        0.8,
        1,
    )
    candidates = [example, rule, exercise]
    brief = EvidenceBrief(
        status="answered",
        reason_code="supported",
        complete=True,
        facts=[
            {"text": c.passage.text, "span_ids": [s["id"] for s in source_spans([c.passage])]}
            for c in [example, exercise]
        ],
    )
    changed = supplement_general_rule(brief, candidates, "آیا هر کار را عقب بیندازیم؟", 3)
    assert changed.facts[-1].text == rule.passage.text
    assert len(changed.facts) == 3
    assert (
        len(supplement_general_rule(brief, candidates, "آیا هر کار را عقب بیندازیم؟", 2).facts) == 2
    )


@pytest.mark.parametrize(
    "question, count",
    [
        ("What does the 30-30-30 model mean, and what are the three debrief questions?", 2),
        ("این دوره چند فصل دارد و مدل ۳۰-۳۰-۳۰ چیست؟", 2),
        ("مدل 30-30-30 چیه؟ مثبت اندیشی سمی ینی چی؟ اون سؤال کمک به همکار مضطرب هم چی بود؟", 3),
        ("همهٔ عنوان‌های فصل‌ها را بگو. بعد فرق پیام اول بحران با دی‌بریف آخر روز را توضیح بده.", 2),
        ("Compare X and Y? Explain it simply.", 1),
        ("What is X?\nHow does it differ from Y?", 1),
        ("Define gravity and explain photosynthesis.", 2),
        ("What are the three steps? Answer in English.", 1),
        ("این دوره چند فصل دارد و اسم همه فصل‌ها چیست؟", 1),
        ("سؤال «الان چه کمکی کنم؟» یعنی چه؟", 1),
        ("یک پیام جدید برای شرکت ما بنویس؛ جزئیات را خودت فرض کن.", 1),
    ],
)
def test_original_units_keep_independent_requests_and_dependencies(question, count):
    units = request_units(question)
    assert len(units) == count
    assert all(u in question for u in units)


def test_all_18_saved_failures_have_their_reference_request_count_without_using_gold_at_runtime():
    root = Path(__file__).resolve().parents[1]
    cases = [
        json.loads(line)
        for line in (root / "evals/natural-student-v2/cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    ids = {
        "natural-060",
        "natural-071",
        "natural-110",
        "natural-111",
        "natural-117",
        "natural-122",
        "natural-128",
        "natural-131",
        "natural-133",
        "natural-140",
        "natural-153",
        "natural-157",
        "natural-160",
        "natural-173",
        "natural-175",
        "natural-178",
        "natural-186",
        "natural-187",
    }
    for case in cases:
        if case["id"] in ids:
            assert len(request_units(case["question"])) == len(case["parts"]), case["id"]


def test_structural_fields_survive_a_classifier_that_calls_everything_content():
    units = request_units("دوره چند فصل دارد و اسم همهٔ فصل‌ها چیست؟ سؤال جادویی چه بود؟")
    proposed = UnitRoutes(
        units=[
            {
                "unit_index": i,
                "intent": "content",
                "metadata_fields": [],
                "chapter_number": None,
                "persian_query": None,
            }
            for i in range(len(units))
        ]
    )
    plan = compile_units(units, proposed, "fa")
    assert [g.intent for g in plan.groups] == ["metadata", "content"]
    assert plan.groups[0].metadata_field == "chapter_titles"
    assert all(g.search_query == g.original_text for g in plan.groups)
    assert compile_units(units, None, "fa").groups[0].intent == "metadata"
    proposed.units[1].unit_index = 0
    with pytest.raises(ValueError, match="indices"):
        compile_units(units, proposed, "fa")


def test_course_says_question_cannot_be_routed_as_unknown_admin_metadata():
    question = "دوره می‌گوید خبر بعدی را همیشه باید ساعت مشخصی اعلام کنیم؟"
    routes = UnitRoutes(
        units=[
            {
                "unit_index": 0,
                "intent": "metadata",
                "metadata_fields": ["other"],
                "chapter_number": None,
                "persian_query": None,
            }
        ]
    )
    assert compile_units([question], routes, "fa").groups[0].intent == "content"


def test_overview_budget_is_separate_and_configurable():
    from hrlearnium.core_config import CoreSettings

    settings = CoreSettings(_env_file=None, max_excerpts=2, overview_max_excerpts=6)
    assert settings.max_excerpts == 2 and settings.overview_max_excerpts == 6


def test_overview_brief_samples_chapters_with_literal_text_not_generated_source_associations():
    from hrlearnium.grounding import overview_brief

    candidates = [
        Candidate(
            SearchPassage(str(i), f"هدف آموزشی فصل {i}.", i, f"عنوان {i}", "introduction"),
            1,
            0.8,
            1,
        )
        for i in range(1, 12)
    ]
    result = overview_brief(candidates, 8)
    assert len(result.facts) == 8
    assert result.facts[0].text == candidates[0].passage.text
    assert result.facts[-1].text == candidates[-1].passage.text
    spans = {s["id"]: s for s in source_spans([c.passage for c in candidates])}
    by_id = {c.passage.id: c.passage for c in candidates}
    assert all(f.text in by_id[spans[f.span_ids[0]]["citation_id"]].text for f in result.facts)
    assert overview_brief([], 8) is None


def test_rejected_brief_can_reselect_evidence_once_without_losing_source_ownership(api_factory):
    client, app, model = api_factory(retrieval_mode="full_context")
    upload(client)
    prepares, reviews = [], []
    model.route_units = lambda units, *a, **kw: UnitRoutes(
        units=[
            {
                "unit_index": 0,
                "intent": "content",
                "metadata_fields": [],
                "chapter_number": None,
                "persian_query": None,
            }
        ]
    )

    def prepare(q, h, candidates, plan, **kwargs):
        prepares.append(kwargs)
        source = candidates[0 if len(prepares) == 1 else -1].passage
        return EvidenceBrief(
            status="answered",
            reason_code="supported",
            complete=True,
            facts=[{"text": "پاسخ مستند.", "span_ids": [s["id"] for s in source_spans([source])]}],
        )

    def write(q, h, sources, plan, brief):
        return Explanation(statements=[{"text": "پاسخ مستند.", "citation_ids": [sources[0].id]}])

    def review(q, h, sources, draft, plan, **kwargs):
        reviews.append(sources[0].id)
        return span_review(draft, sources, answers_original_question=len(reviews) == 2)

    model.prepare_evidence, model.explain_from_brief, model.review_grounded_answer = (
        prepare,
        write,
        review,
    )
    model.repair_explanation = lambda *a, **kw: pytest.fail(
        "A revised brief must use its newly selected sources"
    )
    body = client.post(
        BASE + "/query", headers=token(), json={"question": QUESTION, "include_evaluation": True}
    ).json()
    assert body["status"] == "answered" and len(prepares) == len(reviews) == 2
    assert prepares[1]["feedback"]["review"]["intent"]["answers_original_question"] is False
    assert reviews[0] != reviews[1] and [e["id"] for e in body["excerpts"]] == [reviews[1]]
    assert len(body["evaluation"]["evidence_briefs"]) == 2


@pytest.mark.parametrize("second_review_accepts", [True, False])
def test_incidental_rejected_sentence_is_pruned_but_never_returned_without_fresh_review(
    api_factory, second_review_accepts
):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    model.route_units = lambda units, *a, **kw: UnitRoutes(
        units=[
            {
                "unit_index": 0,
                "intent": "content",
                "metadata_fields": [],
                "chapter_number": None,
                "persian_query": None,
            }
        ]
    )
    calls = {"prepare": 0, "write": 0, "review": 0}

    def prepare(q, h, candidates, plan, **kwargs):
        calls["prepare"] += 1
        assert calls["prepare"] == 1, "Incidental pruning must not reselect the same bad note"
        return EvidenceBrief(
            status="answered",
            reason_code="supported",
            complete=True,
            facts=[
                {
                    "text": "پاسخ مستند.",
                    "span_ids": [s["id"] for s in source_spans([candidates[0].passage])],
                }
            ],
        )

    def write(q, h, sources, plan, brief):
        calls["write"] += 1
        return Explanation(
            statements=[
                {"text": "پاسخ مستند.", "citation_ids": [sources[0].id]},
                {"text": "ادعای اضافی و نامعتبر.", "citation_ids": [sources[0].id]},
            ]
        )

    def review(q, h, sources, draft, plan, **kwargs):
        calls["review"] += 1
        result = span_review(draft, sources)
        if calls["review"] == 1:
            result.statements[1].supported = False
            result.statements[1].span_ids = []
            result.verification.coverage[0].statement_indices = [0]
        elif not second_review_accepts:
            result.intent.answers_original_question = False
        return result

    model.prepare_evidence, model.explain_from_brief, model.review_grounded_answer = (
        prepare,
        write,
        review,
    )
    result = client.post(BASE + "/query", headers=token(), json={"question": QUESTION})
    assert calls == {"prepare": 1, "write": 1, "review": 2}
    if second_review_accepts:
        assert result.status_code == 200 and result.json()["status"] == "answered"
        assert "ادعای اضافی" not in result.json()["answer"]
    else:
        assert result.status_code == 503


@pytest.mark.parametrize("fault", ["required", "intent", "missing_coverage", "bad_source"])
def test_pruning_cannot_drop_required_content_or_bypass_intent_and_ownership(fault):
    from test_hosted_gateway import excerpt

    from hrlearnium.grounding import prune_incidental_statements
    from hrlearnium.schemas import QuestionPlan

    source = excerpt()
    draft = Explanation(
        statements=[
            {"text": "پاسخ مستند.", "citation_ids": [source.id]},
            {"text": "ادعای اضافی.", "citation_ids": [source.id]},
        ]
    )
    review = span_review(draft, [source])
    review.statements[1].supported = False
    review.verification.coverage[0].statement_indices = [0]
    if fault == "required":
        review.verification.coverage[0].statement_indices = [0, 1]
    elif fault == "intent":
        review.intent.answers_original_question = False
    elif fault == "missing_coverage":
        review.verification.coverage = []
    else:
        review.statements[0].span_ids = ["foreign"]
    plan = QuestionPlan(search_query=QUESTION, requirements=[QUESTION], persian_query=None)
    assert prune_incidental_statements(draft, review, [source], plan) is None


@pytest.mark.parametrize("kind,scope", [("example", "مثال آموزشی"), ("exercise", "تمرین پیشنهادی")])
def test_writer_gets_server_owned_scope_before_example_or_exercise_facts(settings, kind, scope):
    from test_hosted_gateway import excerpt

    from hrlearnium.schemas import QuestionPlan

    source = excerpt()
    source.citation.section_kind = kind
    brief = EvidenceBrief(
        status="answered",
        reason_code="supported",
        complete=True,
        facts=[
            {
                "text": "یک گروه اطلاع‌رسانی بسازید.",
                "span_ids": [s["id"] for s in source_spans([source])],
            }
        ],
    )
    original = brief.model_dump()

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert scope in payload["approved_facts"]["facts"][0]["text"]
        assert "نه واقعیت تأییدشده شرکت دانشجو" in payload["approved_facts"]["facts"][0]["text"]
        return response(
            completion(
                {
                    "statements": [
                        {
                            "text": "طبق تمرین دوره یک گروه بسازید.",
                            "fact_indices": [0],
                        }
                    ]
                }
            )
        )

    model = gateway(settings, handler)
    try:
        model.explain_from_brief(
            QUESTION,
            [],
            [source],
            QuestionPlan(search_query=QUESTION, requirements=[QUESTION], persian_query=None),
            brief,
        )
    finally:
        model.close()
    assert brief.model_dump() == original


@pytest.mark.parametrize(
    "question", ["در موردش بیشتر توضیح می‌دهی؟", "فرق این دو روش چیه؟", "Explain more about it."]
)
def test_unresolved_referent_clarifies_without_guessing_sources(api_factory, question):
    client, _, model = api_factory()
    upload(client)
    assert is_bare_followup(question)
    model.plan = model.embed = lambda *a, **kw: pytest.fail(
        "No referent means no speculative retrieval"
    )
    body = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert body["status"] == "clarification" and not body["excerpts"]


def test_page_metadata_is_unavailable_not_character_offsets(api_factory):
    question = "شمارهٔ صفحهٔ وردی که مدل ۳۰-۳۰-۳۰ در آن است چند است؟"
    assert metadata_request(question) is None
    client, _, model = api_factory()
    upload(client)
    model.plan = lambda *a, **kw: pytest.fail("DOCX has no verified pagination")
    body = client.post(BASE + "/query", headers=token(), json={"question": question}).json()
    assert body["status"] == "refused" and body["answer_kind"] == "content"
    assert not body["excerpts"]


def test_new_company_announcement_is_not_a_course_example():
    question = "یک پیام جدید و مخصوص تعطیلی فردای شرکت ما بنویس؛ جزئیات اتفاق را هم خودت فرض کن."
    assert policy_check(question).reason_code == "new_advice"


def test_pagination_guard_does_not_block_page_fault_content_definitions():
    assert policy_check("What page faults mean in operating systems?") is None
    assert policy_check("صفحه چیست؟") is None


@pytest.mark.parametrize(
    "fault", ["none", "missing", "foreign_span", "too_many_sources", "refusal_with_facts"]
)
def test_pre_draft_facts_have_owned_spans_complete_coverage_and_global_budget(fault):
    candidates = [
        Candidate(SearchPassage(str(i), f"متن {i}", i + 1, "عنوان", "explanation"), 1, 0.8, 1)
        for i in range(2)
    ]
    spans = source_spans([c.passage for c in candidates])
    brief = EvidenceBrief(
        status="answered",
        reason_code="supported",
        complete=True,
        facts=[{"text": "متن", "span_ids": [spans[0]["id"]]}],
    )
    if fault == "missing":
        brief.complete = False
    if fault == "foreign_span":
        brief.facts[0].span_ids = ["foreign"]
    if fault == "too_many_sources":
        brief.facts[0].span_ids = [s["id"] for s in spans]
    if fault == "refusal_with_facts":
        brief.status, brief.reason_code, brief.complete = "refused", "insufficient_evidence", False
    if fault in {"foreign_span", "too_many_sources", "refusal_with_facts"}:
        with pytest.raises(ModelUnavailable):
            brief_selection(brief, candidates, max_excerpts=1)
    else:
        result = brief_selection(brief, candidates, max_excerpts=1)
        assert result.selection.status == ("refused" if fault == "missing" else "answered")


def test_shipped_v10_gateway_full_http_path_has_no_offsets_in_llm_context(api_factory, settings):
    client, app, _ = api_factory(retrieval_mode="full_context")
    upload(client)
    calls = []

    def handler(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        assert_strict_schema(schema)
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        assert "source_start" not in json.dumps(payload) and "paragraph_start" not in json.dumps(
            payload
        )
        if "units" in payload:
            value = {
                "units": [
                    {
                        "unit_index": i,
                        "intent": "content",
                        "metadata_fields": [],
                        "chapter_number": None,
                        "persian_query": None,
                    }
                    for i in range(len(payload["units"]))
                ]
            }
        elif "source_context" in payload:
            supported_span = next(s for s in payload["support_spans"] if "واقعیت" in s["text"])
            source = next(
                s for s in payload["source_context"] if s["id"] == supported_span["citation_id"]
            )
            assert "start" not in source and "end" not in source
            value = {
                "decision": {
                    "status": "answered",
                    "reason_code": "supported",
                    "source_ids": [source["id"]],
                    "facts": [
                        {
                            "text": "پیام شامل واقعیت، اقدام سازمان و گام بعدی کارکنان است.",
                            "span_ids": [supported_span["id"]],
                        }
                    ],
                    "requested_values": [],
                },
            }
        elif "approved_facts" in payload:
            value = {
                "statements": [
                    {
                        "text": payload["approved_facts"]["facts"][0]["text"],
                        "fact_indices": [0],
                    }
                ]
            }
        else:
            assert "explanation" in payload
            value = {
                "verification": {
                    "supported": True,
                    "all_question_parts_identified": True,
                    "coverage": [
                        {"requirement_index": 0, "covered": True, "statement_indices": [0]}
                    ],
                },
                "intent": {
                    "answers_original_question": True,
                    "actor_action_preserved": True,
                    "source_stance_preserved": True,
                    "causal_claims_supported": True,
                },
                "rejection_reasons": [],
                "statements": [
                    {
                        "statement_index": 0,
                        "supported": True,
                        "span_ids": [payload["support_spans"][0]["id"]],
                    }
                ],
            }
        if "verification" in value and "intent" in value:
            value = detailed_wire(value)
        return response(completion(value))

    model = gateway(settings, handler)
    app.state.service.gateway = model
    try:
        result = client.post(
            BASE + "/query",
            headers=token(),
            json={"question": "این دوره چند فصل دارد و " + QUESTION, "include_evaluation": True},
        )
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["status"] == "answered" and body["answer_kind"] == "multi"
        assert [p["answer_kind"] for p in body["parts"]] == ["metadata", "content"]
        assert len(calls) == 4
        assert body["evaluation"]["routing_protocol"] == "v12_indexed_modifiers"
    finally:
        model.close()
