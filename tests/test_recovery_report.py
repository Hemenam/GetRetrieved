"""Recovered views retain provenance and never become a clean experiment."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recovery_report", ROOT / "scripts/report_benchmark_recovery.py"
)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def phases(tmp_path):
    original, recovery = tmp_path / "original", tmp_path / "recovery"
    original.mkdir()
    recovery.mkdir()
    entries = []
    base = {
        "id": "q1",
        "question": "Question?",
        "expected_status": "answered",
        "expected_chapters": [1],
        "gold": {},
        "actual_status": "answered",
        "evaluation_status": "evaluated",
        "passed_automated_checks": True,
        "query_latency_ms": 1000,
        "response": {"answer": "Captured answer"},
        "evaluation": {"configuration": {"backend": "synthetic"}, "source_corpus_sha256": "s"},
    }
    for mode in reporter.comparison.MODES:
        outcome = copy.deepcopy(base)
        outcome["retrieval_mode"] = mode
        if mode == "hybrid_rerank":
            outcome.update(
                actual_status=None, evaluation_status="error", passed_automated_checks=False
            )
            outcome["error"] = {"kind": "http_status", "status_code": 503, "stage": "query"}
            outcome.pop("evaluation")
        path = original / f"{mode}.json"
        put(path, {"dataset_sha256": "d", "outcomes": [outcome]})
        entries.append(
            {
                "document_id": "course",
                "repeat": 1,
                "mode": mode,
                "report": path.name,
                "report_sha256": reporter.digest(path),
            }
        )
    put(original / "index.json", {"status": "stopped", "runs": entries})
    fixed = copy.deepcopy(base)
    fixed["retrieval_mode"] = "hybrid_rerank"
    fixed["query_latency_ms"] = 2000
    path = recovery / "retry.json"
    put(path, {"dataset_sha256": "d", "outcomes": [fixed]})
    put(
        recovery / "index.json",
        {
            "kind": "explicit_operational_recovery",
            "source_index_sha256": reporter.digest(original / "index.json"),
            "dataset_sha256": "d",
            "source_corpus_sha256": "s",
            "pending_request": None,
            "captures": [
                {
                    "mode": "hybrid_rerank",
                    "case_id": "q1",
                    "report": path.name,
                    "report_sha256": reporter.digest(path),
                }
            ],
            "status": "complete",
            "attempted_requests": 1,
            "limitations": ["Separate phases"],
        },
    )
    return original, recovery


def test_recovered_view_preserves_source_files_and_labels_phase(phases):
    original, recovery = phases
    before = (original / "index.json").read_bytes()
    groups, summary = reporter.build(original, recovery)
    assert summary["clean_single_experiment"] is False
    assert summary["winner"] is summary["rank"] is None
    assert summary["semantic_review_complete"] is False
    assert summary["recovered_responses"] == summary["original_operational_errors_preserved"] == 1
    assert summary["unresolved_original_errors"] == 0
    outcomes = {m: r["outcomes"][0] for m, r in groups[("course", 1)].items()}
    assert outcomes["hybrid"]["capture_phase"] == "original"
    assert outcomes["hybrid_rerank"]["capture_phase"] == "recovery"
    assert outcomes["hybrid_rerank"]["original_error"]["status_code"] == 503
    text = reporter.comparison.render(groups, experiment_status="recovered")
    assert "RECOVERED VIEW" in text and "not a clean original experiment" in text
    assert "Original attempt retained separately" in text
    assert (original / "index.json").read_bytes() == before


def test_semantic_results_cannot_be_replaced(phases):
    original, recovery = phases
    path = recovery / "index.json"
    index = json.loads(path.read_text())
    index["captures"][0]["mode"] = "full_context"
    put(path, index)
    with pytest.raises(ValueError, match="semantic result"):
        reporter.build(original, recovery)


def test_tampered_recovery_capture_is_rejected(phases):
    original, recovery = phases
    (recovery / "retry.json").write_text("{}")
    with pytest.raises(ValueError, match="capture changed"):
        reporter.build(original, recovery)


def test_changed_runtime_cannot_be_silently_joined(phases):
    original, recovery = phases
    path = recovery / "retry.json"
    value = json.loads(path.read_text())
    value["outcomes"][0]["evaluation"]["configuration"]["backend"] = "different"
    put(path, value)
    index_path = recovery / "index.json"
    index = json.loads(index_path.read_text())
    index["captures"][0]["report_sha256"] = reporter.digest(path)
    put(index_path, index)
    with pytest.raises(ValueError, match="configuration or source"):
        reporter.build(original, recovery)


def test_unresolved_in_flight_request_cannot_be_reported_as_finished(phases):
    original, recovery = phases
    path = recovery / "index.json"
    index = json.loads(path.read_text())
    index["pending_request"] = {"case_id": "q1"}
    put(path, index)
    with pytest.raises(ValueError, match="in-flight"):
        reporter.build(original, recovery)
