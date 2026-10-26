"""Paired reporting is offline and cannot turn a subset into a full-suite score."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_natural_failures as comparison  # noqa: E402


def test_paired_subset_excludes_old_successes_and_keeps_unattempted(tmp_path, monkeypatch):
    def outcome(identifier, passed):
        return {
            "id": identifier, "track": "content", "question": "A question?",
            "expected_status": "answered", "reference": {"id": identifier},
            "response": {"status": "answered", "answer": "An answer.", "parts": []},
            "latency_ms": 1, "error": None,
            "checks": {"pass": passed, "status_matches": True},
        }

    before = {"outcomes": [outcome("a", False), outcome("b", False), outcome("c", True)]}
    after = {"outcomes": [outcome("a", True)]}
    after["summary"] = comparison.summarize(after["outcomes"], 2)
    before_index = {"mode": "hybrid", "source": {}, "dataset_sha256": "same"}
    after_index = before_index | {
        "status": "stopped", "selection": {
            "parent_results_sha256": "parent-hash", "selected_case_ids": ["a", "b"],
            "limitation": "Not a full benchmark score.",
        },
    }
    monkeypatch.setattr(comparison, "load_run", lambda path: (
        (before_index, before, "parent-hash") if path.name == "parent"
        else (after_index, after, "rerun-hash")
    ))
    result = comparison.build(tmp_path / "parent", tmp_path)
    assert result["before"]["collected_cases"] == 1
    assert result["parent_failures_selected"] == 2 and result["cases_paired"] == 1
    assert result["before"]["automatic_checks_passed"] == 0
    assert result["after"]["planned_scored_cases"] == 2
    assert result["recovered_ids"] == ["a"] and result["unattempted_ids"] == ["b"]
    assert (tmp_path / "paired-comparison.md").exists()
    after["outcomes"][0]["reference"] = copy.deepcopy({"changed": True})
    with pytest.raises(ValueError, match="reference changed"):
        comparison.build(tmp_path / "parent", tmp_path)
