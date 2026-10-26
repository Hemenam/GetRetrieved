"""Offline reporting must distinguish observed tokens, price equivalents and truth."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report_natural_benchmark as report  # noqa: E402


def test_price_equivalent_includes_setup_once_and_never_claims_invoice():
    call = {
        "model": "chat",
        "kind": "chat",
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }
    embedding = {"model": "embed", "kind": "embeddings", "usage": {"prompt_tokens": 25}}
    outcomes = [
        {
            "setup_evaluation": {"provider_calls": [call]},
            "evaluation": {"provider_calls": [call, embedding]},
        }
    ]
    pricing = {
        "checked_date": "2026-10-23",
        "limitations": ["Not the actual bill"],
        "models": {
            "chat": {"input_per_million": 1, "output_per_million": 2},
            "embed": {"input_per_million": 0.1, "output_per_million": 0},
        },
    }
    result = report.price_equivalent(outcomes, pricing)
    assert result["by_model"]["chat"]["calls"] == 2
    assert result["by_model"]["chat"]["input_tokens"] == 200
    assert result["captured_calls_undiscounted_equivalent_usd"] == pytest.approx(0.0002425)
    assert result["actual_provider_cost_usd"] is None


def test_missing_usage_is_not_zero_cost():
    outcomes = [
        {
            "error": {"code": "provider_timeout"},
            "evaluation": {"provider_calls": [{"model": "unknown", "kind": "chat", "usage": None}]},
        }
    ]
    result = report.price_equivalent(
        outcomes, {"models": {}, "checked_date": "2026-10-23", "limitations": []}
    )
    assert result["failed_request_usage_unknown"]
    assert result["calls_missing_price_or_usage"] == 1
    assert result["actual_provider_cost_usd"] is None


def test_diagnostics_keep_errors_refusals_and_anchor_flags_distinct():
    outcomes = [
        {
            "id": "a",
            "track": "x",
            "expected_status": "answered",
            "error": {"code": "provider_timeout"},
        },
        {
            "id": "b",
            "track": "x",
            "expected_status": "answered",
            "response": {"status": "refused", "reason_code": "insufficient_evidence"},
            "checks": {"pass": False, "status_matches": False, "parts": []},
        },
        {
            "id": "c",
            "track": "x",
            "expected_status": "answered",
            "response": {"status": "answered", "reason_code": "supported"},
            "checks": {
                "pass": False,
                "status_matches": True,
                "parts": [{"reference_anchors_in_own_evidence": False}],
            },
        },
    ]
    result = report.diagnostics(outcomes)
    assert result["flagged_by_response_reason"] == {
        "operational_error": 1,
        "insufficient_evidence": 1,
        "supported": 1,
    }
    assert result["flagged_cases"][-1]["failed_checks"] == [
        "part_1.reference_anchors_in_own_evidence"
    ]


def test_final_report_rejects_running_checkpoint(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (tmp_path / "results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="collection to finish"):
        report.load_run(tmp_path)


def test_nested_generation_failure_is_reported_as_operational_not_refusal():
    outcomes = [
        {
            "id": "x",
            "track": "multipart",
            "expected_status": "answered",
            "response": {
                "status": "partial",
                "reason_code": "some_parts_unsupported",
                "parts": [{"status": "error", "reason_code": "answer_verification_failed"}],
            },
            "checks": {"pass": False, "parts": []},
        }
    ]
    result = report.diagnostics(outcomes)
    assert result["flagged_by_response_reason"] == {"part_operational_error": 1}
    assert result["flagged_cases"][0]["part_errors"] == ["answer_verification_failed"]


def test_candidate_diagnostics_do_not_conflate_search_and_verifier_failures():
    reference = {
        "parts": [
            {
                "kind": "content",
                "status": "answered",
                "facts": [
                    {"evidence": [{"quote": "required", "chapter": 1, "section": "explanation"}]}
                ],
            }
        ]
    }

    def outcome(identifier, candidates, status="refused"):
        return {
            "id": identifier,
            "reference": reference,
            "response": {"status": status},
            "evaluation": {} if candidates is None else {"selector_candidates": candidates},
        }

    rows = [
        outcome("present", [{"id": "p"}]),
        outcome("absent", []),
        outcome("presearch", None, "clarification"),
    ]
    result = report.candidate_diagnostics(
        rows,
        [
            {
                "id": "p",
                "text": "required source",
                "chapter_number": 1,
                "section_kind": "explanation",
            }
        ],
    )
    assert result["refused_despite_all_reference_anchors_in_selector"] == ["present"]
    assert result["refused_with_missing_reference_anchors_in_selector"] == ["absent"]
    assert result["no_selector_trace"] == ["presearch"]


def test_macro_rates_keep_unattempted_cases_and_correlated_families():
    cases = [
        {"id": "a", "track": "A", "question_family_id": "one"},
        {"id": "b", "track": "A", "question_family_id": "one"},
        {"id": "c", "track": "B", "question_family_id": "two"},
    ]
    outcomes = [{"id": "a", "checks": {"pass": True, "status_matches": True}}]
    result = report.diagnostic_rates(cases, outcomes)
    assert result["micro_automatic_check_rate"] == pytest.approx(1 / 3)
    assert result["track_macro_automatic_check_rate"] == 0.25
    assert result["family_macro_automatic_check_rate"] == 0.25
