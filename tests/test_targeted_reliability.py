"""Repeat arithmetic must not cherry-pick or equate automatic passes with quality."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_targeted_reliability import metrics, validate_ratings  # noqa: E402


def fixture():
    runs = [
        {
            "name": name,
            "sha256": name,
            "index": {"attempted_query_turns": 2},
            "rows": [
                {"id": "a", "response": {"status": "answered"}, "checks": {"pass": True}},
                {"id": "b", "response": {"status": "answered"}, "checks": {"pass": True}},
            ],
        }
        for name in ("r1", "r2", "r3")
    ]
    ratings = {
        name: {i: {"verdict": "pass", "note": "Supported by own source."} for i in ("a", "b")}
        for name in ("r1", "r2", "r3")
    }
    return runs, ratings


def test_every_repeat_counts_even_when_automatic_checks_pass():
    runs, ratings = fixture()
    ratings["r2"]["b"] = {"verdict": "fail", "note": "An unsupported extra claim."}
    result = metrics(runs, ratings)
    assert result["automatic_passes"] == 6 and result["acceptable_executions"] == 5
    assert result["first_repeat_acceptable"] == 2 and result["acceptable_every_repeat"] == 1
    assert result["case_executions"] == 6


def test_missing_judgment_cannot_disappear_from_denominator():
    runs, ratings = fixture()
    del ratings["r3"]["b"]
    with pytest.raises(ValueError, match="every captured response"):
        validate_ratings(runs, ratings)


def test_partial_operational_failure_cannot_be_scored_as_success():
    runs, ratings = fixture()
    runs[0]["rows"][0]["response"] = {"status": "partial", "parts": [{"status": "error"}]}
    with pytest.raises(ValueError, match="Operational errors"):
        validate_ratings(runs, ratings)
    ratings["r1"]["a"] = {"verdict": "error", "note": "One independent part was not delivered."}
    assert metrics(runs, ratings)["acceptable_executions"] == 5
