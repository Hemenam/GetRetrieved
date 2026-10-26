"""Offline report review binding never substitutes automatic checks for judgments."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report_extension_v6 as report  # noqa: E402


def test_review_requires_exact_mode_cohort_and_capture_hashes(tmp_path):
    path = tmp_path / "capture.json"
    path.write_text("original captured answer", encoding="utf-8")
    reports = {m: {"outcomes": [{"id": "case", "capture_path": str(path)}]} for m in report.MODES}
    review = {
        "reviewer": "Assistant",
        "human_approved": False,
        "judgments": [
            {
                "case_id": "case",
                "mode": m,
                "score": 5,
                "note": "Reviewed against source.",
                "capture_sha256": report.digest(path),
            }
            for m in report.MODES
        ],
    }
    assert len(report.validate_review(review, [{"id": "case"}], reports)) == 3
    review["judgments"][0]["score"] = 3
    with pytest.raises(ValueError, match="Score"):
        report.validate_review(review, [{"id": "case"}], reports)
    review["judgments"][0]["score"] = 5
    path.write_text("different answer", encoding="utf-8")
    with pytest.raises(ValueError, match="different response"):
        report.validate_review(review, [{"id": "case"}], reports)


def test_missing_judgment_is_not_invented():
    review = {"reviewer": "Assistant", "human_approved": False, "judgments": []}
    with pytest.raises(ValueError, match="Exactly 36"):
        report.validate_review(review, [{"id": "case"}], {})
