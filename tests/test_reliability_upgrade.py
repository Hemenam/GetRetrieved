"""Bounded recovery, non-answer schemas, and cost honesty; no real provider calls."""

import json

import httpx
import pytest
from test_grounding import MULTIPART, plan
from test_hosted_gateway import candidate, completion, gateway, response
from test_hosted_gateway import settings as settings

from hrlearnium.eval_tools import runtime_summary
from hrlearnium.evaluation import collect, stage
from hrlearnium.models import ModelUnavailable


@pytest.mark.parametrize(
    "failure", [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError]
)
def test_transient_failure_recovers_once_and_every_attempt_is_visible(settings, failure, caplog):
    settings.api_retry_backoff_seconds = 0
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise failure("private-endpoint-and-source-marker", request=request)
        return response(completion())

    model = gateway(settings, handler)
    with collect(settings, True) as trace, stage("selection"):
        result = model.select("question", [], [candidate()])
    assert result.status == "answered" and len(calls) == 2
    assert len(trace["provider_calls"]) == 1 and len(trace["provider_attempts"]) == 2
    assert trace["provider_attempts"][0]["will_retry"] is True
    assert trace["provider_attempts"][1]["outcome"] == "success"
    assert calls[0].content == calls[1].content
    assert "private-endpoint-and-source-marker" not in caplog.text
    assert "will_retry=True" in caplog.text
    model.close()


def test_retry_exhaustion_and_disable_are_bounded(settings):
    settings.api_retry_backoff_seconds = 0
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadError("private", request=request)

    model = gateway(settings, handler)
    with pytest.raises(ModelUnavailable):
        model.select("question", [], [candidate()])
    assert len(calls) == 2
    calls.clear()
    settings.api_max_retries = 0
    with pytest.raises(ModelUnavailable):
        model.select("question", [], [candidate()])
    assert len(calls) == 1
    model.close()


def test_no_retry_starts_after_the_call_budget_expires(settings, monkeypatch):
    calls, clock = [], [0.0]
    monkeypatch.setattr("hrlearnium.models.time.monotonic", lambda: clock[0])

    def handler(request):
        calls.append(request)
        clock[0] = settings.model_timeout_seconds + 1
        raise httpx.ReadTimeout("private", request=request)

    model = gateway(settings, handler)
    with pytest.raises(ModelUnavailable):
        model.select("question", [], [candidate()])
    assert len(calls) == 1
    model.close()


def test_budget_is_rechecked_after_backoff_or_host_pause(settings, monkeypatch):
    calls, clock = [], [0.0]
    monkeypatch.setattr("hrlearnium.models.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "hrlearnium.models.time.sleep",
        lambda delay: clock.__setitem__(0, settings.model_timeout_seconds + 1),
    )

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private", request=request)

    model = gateway(settings, handler)
    with pytest.raises(ModelUnavailable) as error:
        model.select("question", [], [candidate()])
    assert len(calls) == 1 and error.value.diagnostic_code == "provider_timeout"
    model.close()


@pytest.mark.parametrize("value", [None, {"status": "refused", "reason_code": "outside_scope"}])
def test_non_answer_wire_branches_have_no_evidence_and_are_not_retried(settings, value):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        assert "anyOf" in schema["properties"]["decision"]
        fields = schema["$defs"]["RefusedEvidenceDecision"]["properties"]
        assert set(fields) == {"status", "reason_code"}
        calls.append(request)
        return response(
            completion(
                {"decision": value or {"status": "clarification", "reason_code": "ambiguous"}}
            )
        )

    model = gateway(settings, handler)
    selected = model.select_complete(MULTIPART, [], [candidate()], plan())
    assert selected.selection.status in {"refused", "clarification"}
    assert selected.selection.passage_ids == selected.coverage == []
    assert len(calls) == 1
    model.close()


def test_flat_ids_are_derived_from_coverage_not_independently_generated(settings):
    wire = {
        "decision": {
            "status": "answered",
            "reason_code": "supported",
            "source_ids": ["passage-1"],
            "coverage": {"r0": [0], "r1": [0]},
        }
    }
    model = gateway(settings, lambda request: response(completion(wire)))
    selected = model.select_complete(MULTIPART, [], [candidate()], plan())
    assert selected.selection.passage_ids == ["passage-1"]
    assert len(selected.coverage) == 2
    model.close()


def test_refusal_with_evidence_is_still_invalid_not_an_unverified_answer(settings):
    wire = {"decision": {"status": "refused", "reason_code": "outside_scope", "coverage": []}}
    calls = []

    def handler(request):
        calls.append(request)
        return response(completion(wire))

    model = gateway(settings, handler)
    with pytest.raises(ModelUnavailable):
        model.select_complete(MULTIPART, [], [candidate()], plan())
    assert len(calls) == 1  # No schema/semantic repair retry.
    model.close()


def test_ambiguous_failed_attempt_prevents_a_complete_dollar_estimate():
    trace = {
        "provider_calls": [
            {
                "model": "m",
                "kind": "chat/completions",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        ],
        "provider_attempts": [
            {"attempt": 1, "outcome": "provider_transport_error"},
            {"attempt": 2, "outcome": "success"},
        ],
    }
    result = runtime_summary(
        [{"evaluation_status": "evaluated", "evaluation": trace, "expected_status": "answered"}],
        {"m": {"input_per_million": 1, "output_per_million": 1}},
    )
    assert result["observed_failed_attempts"] == result["observed_retries"] == 1
    assert result["observed_provider_attempts"] == 2
    assert result["estimated_cost_usd"] is None and not result["cost_diagnostics_complete"]
