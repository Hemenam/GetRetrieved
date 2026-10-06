"""Runner contract tests with HTTP mocks; these do not evaluate model quality."""

from __future__ import annotations

import argparse
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from hrlearnium.schemas import Excerpt, Explanation, QueryResponse, render_answer

SPEC = importlib.util.spec_from_file_location(
    "evaluation_runner", Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

EXCERPT = {
    "id": "excerpt-one",
    "text": "پیام اضطراری شامل واقعیت، اقدام و گام بعدی است.",
    "citation": {
        "document_id": "document-one",
        "version": 1,
        "document_title": "دوره بحران",
        "chapter_number": 1,
        "chapter_title": "پیام اضطراری",
        "section_kind": "explanation",
        "paragraph_start": 4,
        "paragraph_end": 4,
        "source_start": 30,
        "source_end": 78,
        "source_sha256": "a" * 64,
    },
}
CASE = {
    "id": "runner-001",
    "category": "paraphrase",
    "question": "پیام موقع بحران چه اجزایی دارد؟",
    "expected_status": "answered",
    "expected_chapters": [1],
    "required_quotes": ["واقعیت، اقدام و گام بعدی"],
    "notes": "",
}


def response(mode="verbatim"):
    explanation = (
        Explanation.model_validate(
            {
                "statements": [{"text": "پیام سه بخش دارد.", "citation_ids": [EXCERPT["id"]]}],
            }
        )
        if mode == "explained"
        else None
    )
    return {
        "request_id": "request-one",
        "status": "answered",
        "answer": render_answer([Excerpt.model_validate(EXCERPT)], explanation),
        "excerpts": [deepcopy(EXCERPT)],
        "conversation_id": str(uuid4()),
        "reason_code": "supported",
        "retrieval_mode": "hybrid",
        "response_mode": mode,
        "explanation": explanation.model_dump() if explanation else None,
    }


def arguments(mode="verbatim"):
    return argparse.Namespace(
        local_auth=False,
        document=None,
        base_url="http://127.0.0.1:8000",
        course="course-one",
        tenant=None,
        subject=None,
        timeout=1,
        delay_seconds=0,
        response_mode=mode,
    )


@pytest.fixture
def mock_api(monkeypatch):
    real_client = httpx.Client
    monkeypatch.setenv("HR_EVAL_TOKEN", "only-used-by-mock-transport")

    def install(payload, *, stored=None, status_code=200):
        calls = []

        def handle(request):
            calls.append(
                (
                    request.method,
                    request.url.path,
                    json.loads(request.content) if request.content else None,
                )
            )
            if request.method == "GET":
                return httpx.Response(200, json=stored if stored is not None else EXCERPT)
            return httpx.Response(status_code, json=payload)

        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kwargs: real_client(
                transport=httpx.MockTransport(handle),
                **kwargs,
            ),
        )
        return calls

    return install


def test_verbatim_preserves_exact_join_and_resolves_every_excerpt(mock_api):
    calls = mock_api(response())
    outcomes, metadata = runner.api_evaluation([CASE], arguments())
    outcome = outcomes[0]
    assert outcome["passed_automated_checks"]
    assert outcome["answer_exactly_joined_excerpts"]
    assert outcome["citation_resolution_exact"]
    assert not outcome["generated_explanation_human_review_required"]
    assert calls[0][2]["response_mode"] == "verbatim"
    assert calls[1][:2] == ("GET", "/v1/courses/course-one/excerpts/excerpt-one")
    assert metadata["response_mode"] == "verbatim"
    assert runner.api_summary(outcomes)["exact_answer_assembly"]["denominator"] == 1


def test_explained_followup_forwards_mode_and_requires_human_semantic_review(mock_api):
    payload = response("explained")
    calls = mock_api(payload)
    case = CASE | {"previous_question": "درباره پیام اضطراری چه می‌گوید؟"}
    outcomes, metadata = runner.api_evaluation([case], arguments("explained"))
    outcome = outcomes[0]
    assert outcome["passed_automated_checks"]
    assert not outcome["setup_failed"]
    assert outcome["answer_exactly_joined_excerpts"] is None
    assert outcome["answer_canonical_render"]
    assert outcome["explanation_citation_links_valid"]
    assert outcome["generated_explanation_human_review_required"]
    assert calls[0][2]["response_mode"] == calls[1][2]["response_mode"] == "explained"
    assert "conversation_id" not in calls[0][2]
    assert calls[1][2]["conversation_id"] == payload["conversation_id"]
    summary = runner.api_summary(outcomes)
    assert summary["exact_answer_assembly"]["denominator"] == 0
    assert summary["explained_canonical_render"]["rate"] == 1
    assert summary["generated_explanations_requiring_human_review"] == 1
    assert summary["explanation_semantic_correctness"] == "not_evaluated_by_automated_checks"
    assert metadata["response_mode"] == "explained"


@pytest.mark.parametrize("references", [["missing"], [""], ["excerpt-one", "excerpt-one"]])
def test_invalid_explanation_links_fail_without_rendering_crash(mock_api, references):
    payload = response("explained")
    payload["explanation"]["statements"][0]["citation_ids"] = references
    mock_api(payload)
    outcomes, _ = runner.api_evaluation([CASE], arguments("explained"))
    assert outcomes[0]["evaluation_status"] == "evaluated"
    assert not outcomes[0]["explanation_citation_links_valid"]
    assert not outcomes[0]["passed_automated_checks"]


@pytest.mark.parametrize(
    "requested,returned", [("verbatim", "explained"), ("explained", "verbatim")]
)
def test_response_mode_mismatch_is_failure(mock_api, requested, returned):
    mock_api(response(returned))
    outcomes, _ = runner.api_evaluation([CASE], arguments(requested))
    assert not outcomes[0]["response_mode_matches"]
    assert not outcomes[0]["passed_automated_checks"]


@pytest.mark.parametrize("mode", ["verbatim", "explained"])
def test_explanation_presence_must_follow_requested_mode(mock_api, mode):
    payload = response("explained" if mode == "verbatim" else "verbatim")
    payload["response_mode"] = mode
    mock_api(payload)
    outcomes, _ = runner.api_evaluation([CASE], arguments(mode))
    assert outcomes[0]["response_mode_matches"]
    assert not outcomes[0]["explanation_contract_valid"]
    assert not outcomes[0]["passed_automated_checks"]


def test_generated_claim_can_pass_structure_but_is_never_marked_semantically_verified(mock_api):
    payload = response("explained")
    payload["explanation"]["statements"][0]["text"] = "این ادعای ساختگی در متن منبع نیست."
    payload["answer"] = render_answer(
        [Excerpt.model_validate(EXCERPT)],
        Explanation.model_validate(payload["explanation"]),
    )
    mock_api(payload)
    outcomes, _ = runner.api_evaluation([CASE], arguments("explained"))
    assert outcomes[0]["passed_automated_checks"]
    assert outcomes[0]["generated_explanation_human_review_required"]
    assert (
        runner.api_summary(outcomes)["explanation_semantic_correctness"]
        == "not_evaluated_by_automated_checks"
    )


def test_generated_prose_cannot_satisfy_required_quote_coverage(mock_api):
    payload = response("explained")
    payload["explanation"]["statements"][0]["text"] = "این عبارت فقط در توضیح تولیدشده آمده است."
    payload["answer"] = render_answer(
        [Excerpt.model_validate(EXCERPT)],
        Explanation.model_validate(payload["explanation"]),
    )
    mock_api(payload)
    case = CASE | {"required_quotes": [payload["explanation"]["statements"][0]["text"]]}
    outcomes, _ = runner.api_evaluation([case], arguments("explained"))
    assert not outcomes[0]["required_coverage_complete"]
    assert not outcomes[0]["passed_automated_checks"]


@pytest.mark.parametrize("mode", ["verbatim", "explained"])
def test_answer_rendering_tampering_fails(mock_api, mode):
    payload = response(mode)
    payload["answer"] += "\nمتن اضافهٔ تاییدنشده"
    mock_api(payload)
    outcomes, _ = runner.api_evaluation([CASE], arguments(mode))
    assert not outcomes[0]["answer_canonical_render"]
    assert not outcomes[0]["passed_automated_checks"]


@pytest.mark.parametrize("mode", ["verbatim", "explained"])
def test_stored_excerpt_mismatch_fails_in_both_modes(mock_api, mode):
    stored = deepcopy(EXCERPT)
    stored["text"] += " متن متفاوت"
    mock_api(response(mode), stored=stored)
    outcomes, _ = runner.api_evaluation([CASE], arguments(mode))
    assert not outcomes[0]["citation_resolution_exact"]
    assert not outcomes[0]["passed_automated_checks"]


@pytest.mark.parametrize(
    "statements",
    [[], [{"text": "", "citation_ids": ["excerpt-one"]}], [{"text": "متن", "citation_ids": []}]],
)
def test_invalid_explanation_schema_is_operational_error(mock_api, statements):
    payload = response("explained")
    payload["explanation"]["statements"] = statements
    mock_api(payload)
    outcomes, _ = runner.api_evaluation([CASE], arguments("explained"))
    assert outcomes[0]["error"]["kind"] == "invalid_response"
    assert runner.api_summary(outcomes)["operational_errors"] == 1


def test_provider_503_is_an_error_not_a_successful_refusal(mock_api):
    mock_api({"detail": "unavailable"}, status_code=503)
    case = CASE | {"expected_status": "refused", "required_quotes": [], "expected_chapters": []}
    outcomes, _ = runner.api_evaluation([case], arguments("explained"))
    assert outcomes[0]["actual_status"] is None
    summary = runner.api_summary(outcomes)
    assert summary["operational_errors"] == 1
    assert summary["passed_automated_checks"]["numerator"] == 0


def test_setup_mode_mismatch_cannot_pass_a_followup():
    setup = QueryResponse.model_validate(response("verbatim"))
    checks = runner.response_contract_checks(setup, "explained")
    assert not checks["response_mode_matches"]
    assert not checks["explanation_contract_valid"]


@pytest.mark.parametrize("mode", ["verbatim", "explained"])
def test_cli_default_and_explicit_explained_mode_reach_http_requests(mock_api, tmp_path, mode):
    calls = mock_api(response(mode))
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(CASE, ensure_ascii=False) + "\n", encoding="utf-8")
    report = tmp_path / "report.json"
    argv = [
        "--mode",
        "api",
        "--course",
        "course-one",
        "--dataset",
        str(dataset),
        "--output",
        str(report),
    ]
    if mode == "explained":
        argv.extend(["--response-mode", "explained"])
    assert runner.main(argv) == 0
    assert calls[0][2]["response_mode"] == mode
    assert json.loads(report.read_text(encoding="utf-8"))["metadata"]["response_mode"] == mode
