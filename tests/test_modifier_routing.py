"""Bounded modifier integration retains every original request and per-part outcome."""

import pytest
from test_api import BASE, QUESTION, token, upload
from test_api import api_factory as api_factory
from test_response_modes import install_explainer

from hrlearnium.request_units import compile_units, request_units
from hrlearnium.schemas import UnitRoutes


def routes(count, parents=None):
    parents = parents or {}
    return UnitRoutes(
        units=[
            {
                "unit_index": i,
                "parent_unit_index": parents.get(i),
                "intent": "content",
                "metadata_fields": [],
                "chapter_number": None,
                "persian_query": None,
            }
            for i in range(count)
        ]
    )


@pytest.mark.parametrize(
    "question",
    [
        "Name the three items? Briefly explain each item.",
        "سه مورد را بگو؟ کنار هر مورد کوتاه بگو منظور چیست.",
    ],
)
def test_presentation_attachment_keeps_exact_original_question(question):
    units = request_units(question)
    assert len(units) == 2
    plan = compile_units(units, routes(2, {1: 0}), "fa", original_question=question)
    assert len(plan.groups) == 1
    assert plan.groups[0].original_text == question
    assert plan.groups[0].requirements == [question]
    assert plan.groups[0].search_query == question


def test_independent_questions_and_comparisons_keep_their_own_requirements():
    question = "Define gravity? Define photosynthesis?"
    units = request_units(question)
    plan = compile_units(units, routes(2), "en", original_question=question)
    assert [g.original_text for g in plan.groups] == units
    comparison = "Compare gravity and acceleration? Explain it simply."
    units = request_units(comparison)
    assert len(units) == 1
    assert len(compile_units(units, routes(1), "en", original_question=comparison).groups) == 1


@pytest.mark.parametrize("fault", ["metadata", "search", "forward", "chain"])
def test_invalid_relation_is_rejected_not_silently_executed(fault):
    units = ["Name the items?", "Explain each briefly.", "Keep the wording concise."]
    proposed = routes(3, {1: 0, 2: 0})
    if fault == "metadata":
        proposed.units[1].metadata_fields = ["chapter_titles"]
    elif fault == "search":
        proposed.units[1].persian_query = "جستجوی اضافه"
    elif fault == "forward":
        proposed.units[0].parent_unit_index = 1
    else:
        proposed.units[2].parent_unit_index = 1
    with pytest.raises(ValueError):
        compile_units(units, proposed, "en")


def test_http_modifier_is_one_answer_not_duplicated_subanswers(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    install_explainer(model)
    question = QUESTION + " کنار هر بخش کوتاه بگو منظور چیست."
    model.route_units = lambda units, *a, **kw: routes(len(units), {1: 0})
    result = client.post(
        BASE + "/query", headers=token(), json={"question": question, "include_evaluation": True}
    )
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == "answered" and body["answer_kind"] == "content" and not body["parts"]
    assert body["evaluation"]["request_plan"]["groups"][0]["original_text"] == question


def test_bad_parent_falls_back_to_all_original_independent_questions(api_factory):
    client, _, model = api_factory(retrieval_mode="full_context")
    upload(client)
    install_explainer(model)
    question = QUESTION + " مدل ۳۰-۳۰-۳۰ چیست؟"
    model.route_units = lambda units, *a, **kw: routes(len(units), {0: 1})
    result = client.post(BASE + "/query", headers=token(), json={"question": question})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["answer_kind"] == "multi" and len(body["parts"]) == 2
    assert [p["question"] for p in body["parts"]] == request_units(question)
