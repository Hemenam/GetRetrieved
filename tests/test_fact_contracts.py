"""Fact/span and fact/writer ownership; semantics still need independent review."""

import copy
import json

import httpx
import pytest
from pydantic import ValidationError
from test_hosted_gateway import assert_strict_schema, completion, excerpt, gateway, response
from test_hosted_gateway import settings as settings

from hrlearnium.fact_contracts import (
    bind_writer_facts,
    fact_explanation_model,
    normalize_precise_brief,
    precise_brief_model,
)
from hrlearnium.models import ModelUnavailable, OllamaGateway
from hrlearnium.retrieval import Candidate, SearchPassage
from hrlearnium.schemas import EvidenceBrief, QuestionPlan
from hrlearnium.source_spans import source_spans


def sources():
    first = excerpt().model_copy(
        update={
            "id": "one",
            "text": "Review whether a temporary 10–15% adjustment is needed.\n"
            + "Unrelated background. " * 65,
        }
    )
    second = excerpt().model_copy(
        update={
            "id": "two",
            "text": "The first sentence states facts. Later sentences state actions and the next step.",
        }
    )
    return first, second


def decision():
    first, second = sources()
    spans = source_spans([first, second])
    return {
        "status": "answered",
        "reason_code": "supported",
        "source_ids": [first.id],
        "facts": [
            {
                "text": "Review whether a temporary 10–15% adjustment is needed.",
                "span_ids": [spans[0]["id"]],
            }
        ],
        "requested_values": [],
    }, spans


def test_fact_preserves_exact_selected_spans_not_every_span_in_passage():
    value, spans = decision()
    assert len([s for s in spans if s["citation_id"] == "one"]) > 1
    model = precise_brief_model(["one", "two"], spans, 1)
    assert_strict_schema(model.model_json_schema())
    parsed = model.model_validate({"decision": value}).model_dump(mode="json")["decision"]
    brief = normalize_precise_brief(parsed, spans)
    assert brief.facts[0].span_ids == [spans[0]["id"]]
    assert brief.facts[0].text == value["facts"][0]["text"]


def test_bounded_brief_wire_limit_can_hold_schema_valid_large_facts():
    from hrlearnium.models import _content_limit

    value, spans = decision()
    model = precise_brief_model(["one", "two"], spans, 1)
    refs = [spans[0]["id"]] * 16
    # Transport size handling precedes duplicate/ownership validation. Its size
    # ceiling must not reject bounded schema-valid output before those checks.
    value["facts"] = [{"text": "A" * 1000, "span_ids": refs} for _ in range(8)]
    value["requested_values"] = [{"value": "V" * 200, "source_index": 0} for _ in range(8)]
    parsed = model.model_validate({"decision": value})
    assert len(parsed.model_dump_json()) <= _content_limit(model)


@pytest.mark.parametrize(
    "fault",
    ["unknown_span", "foreign_source", "duplicate_span", "duplicate_source", "unused_source"],
)
def test_brief_cannot_borrow_or_expand_support(fault):
    value, spans = decision()
    if fault == "unknown_span":
        value["facts"][0]["span_ids"] = ["invented"]
    elif fault == "foreign_source":
        value["facts"][0]["span_ids"] = [spans[-1]["id"]]
    elif fault == "duplicate_span":
        value["facts"][0]["span_ids"] *= 2
    elif fault == "duplicate_source":
        value["source_ids"] *= 2
    else:
        value["source_ids"].append("two")
    with pytest.raises(ModelUnavailable) as error:
        normalize_precise_brief(value, spans)
    assert error.value.diagnostic_code == "invalid_evidence_brief"


@pytest.mark.parametrize("bad", [[1], [-1], [True], [], ["0"]])
def test_writer_indices_are_bounded_typed_and_required(bad):
    with pytest.raises(ValidationError):
        fact_explanation_model(1).model_validate(
            {"statements": [{"text": "Fact.", "fact_indices": bad}]}
        )


def test_two_sided_comparison_citations_come_from_both_referenced_facts():
    first, second = sources()
    spans = source_spans([first, second])
    brief = EvidenceBrief(
        status="answered",
        reason_code="supported",
        complete=True,
        facts=[
            {"text": "First fact.", "span_ids": [spans[0]["id"]]},
            {"text": "Second fact.", "span_ids": [spans[-1]["id"]]},
        ],
    )
    wire = fact_explanation_model(2).model_validate(
        {"statements": [{"text": "A supported comparison.", "fact_indices": [0, 1]}]}
    )
    draft, mapping = bind_writer_facts(wire, brief, [first, second])
    assert draft.statements[0].citation_ids == ["one", "two"]
    assert mapping[0]["span_ids"] == [spans[0]["id"], spans[-1]["id"]]
    altered = first.model_copy(update={"text": first.text + "changed"})
    with pytest.raises(ModelUnavailable, match="changed"):
        bind_writer_facts(wire, brief, [altered, second])


def test_writer_cannot_author_citations_or_reference_duplicate_facts():
    model = fact_explanation_model(1)
    with pytest.raises(ValidationError):
        model.model_validate(
            {"statements": [{"text": "Fact.", "fact_indices": [0], "citation_ids": ["foreign"]}]}
        )
    wire = model.model_validate({"statements": [{"text": "Fact.", "fact_indices": [0]}]})
    wire.statements[0].fact_indices = [0, 0]
    value, spans = decision()
    with pytest.raises(ModelUnavailable):
        bind_writer_facts(wire, normalize_precise_brief(value, spans), list(sources()))


@pytest.mark.parametrize("backend", ["json_schema", "json_object", "ollama"])
def test_shipped_fact_brief_and_writer_protocols_keep_source_text_out_of_prose_budget(
    settings, backend
):
    first, _ = sources()
    passage = SearchPassage(first.id, first.text, 1, "Policy", "explanation")
    candidates = [Candidate(passage, 1, 0.8, 1)]
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        schema = body.get("format") or body.get("response_format", {}).get("json_schema", {}).get(
            "schema"
        )
        if schema:
            assert_strict_schema(schema)
        if len(calls) == 1:
            assert "text" not in payload["source_context"][0]
            assert len(payload["support_spans"]) > 1
            value = {
                "decision": {
                    "status": "answered",
                    "reason_code": "supported",
                    "source_ids": [first.id],
                    "facts": [
                        {
                            "text": "Review whether a temporary 10–15% adjustment is needed.",
                            "span_ids": [payload["support_spans"][0]["id"]],
                        }
                    ],
                    "requested_values": [],
                }
            }
        else:
            assert "excerpts" not in payload and len(payload["support_spans"]) == 1
            assert payload["approved_facts"]["facts"][0]["fact_index"] == 0
            value = {
                "statements": [
                    {
                        "text": "Review whether a temporary 10–15% adjustment is needed.",
                        "fact_indices": [0],
                    }
                ]
            }
        if backend == "ollama":
            return response(
                {"done": True, "done_reason": "stop", "message": {"content": json.dumps(value)}}
            )
        return response(completion(value))

    settings.api_structured_output = "json_object" if backend == "json_object" else "json_schema"
    model = (
        OllamaGateway(
            settings,
            client=httpx.Client(
                base_url="http://localhost", transport=httpx.MockTransport(handler)
            ),
        )
        if backend == "ollama"
        else gateway(settings, handler)
    )
    plan = QuestionPlan(
        search_query="What adjustment is required?",
        persian_query=None,
        requirements=["What adjustment is required?"],
    )
    try:
        brief = model.prepare_evidence(plan.search_query, [], candidates, plan)
        original = copy.deepcopy(brief.model_dump())
        draft = model.explain_from_brief(plan.search_query, [], [first], plan, brief)
        assert draft.statements[0].citation_ids == [first.id]
        assert brief.model_dump() == original and len(calls) == 2
    finally:
        model.close()
