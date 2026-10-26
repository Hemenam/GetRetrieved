"""Incomplete collection is reported without inventing missing responses or grades."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report_extension_interim as report  # noqa: E402


def test_partial_collection_preserves_errors_and_requires_only_actual_answer_reviews(tmp_path):
    original, recovery, pack = (tmp_path / n for n in ["original", "recovery", "pack"])
    for path in [original, recovery, pack]:
        path.mkdir()
    case = {"id": "case", "question": "Persian question", "parts": []}
    (pack / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    dataset_hash = report.digest(pack / "cases.jsonl")
    report.bench.write_json(pack / "manifest.json", {"cases_sha256": dataset_hash})
    for directory, mode, outcome in [
        (
            original,
            "hybrid",
            {"response": {"answer": "Actual answer"}, "evaluation": {"configuration": {}}},
        ),
        (recovery, "hybrid_rerank", {"error": {"kind": "timeout", "stage": "query"}}),
    ]:
        path = directory / "capture.json"
        report.bench.write_json(
            path,
            {
                "metadata": {"requested_retrieval_mode": mode},
                "outcomes": [
                    {
                        "id": "case",
                        "question": case["question"],
                        "extension_definition": case,
                        **outcome,
                    }
                ],
            },
        )
        index = {
            "status": "stopped",
            "pending_request": None,
            "dataset_sha256": dataset_hash,
            "fixed_configuration": {},
            "attempted_queries": 1,
            "captures": [
                {
                    "case_id": "case",
                    "mode": mode,
                    "report": path.name,
                    "report_sha256": report.digest(path),
                }
            ],
        }
        if directory == recovery:
            index["original_index_sha256"] = report.digest(original / "index.json")
        report.bench.write_json(directory / "index.json", index)
    review = {
        "reviewer": "Assistant",
        "human_approved": False,
        "original_index_sha256": report.digest(original / "index.json"),
        "recovery_index_sha256": report.digest(recovery / "index.json"),
        "judgments": [
            {"case_id": "case", "modes": ["hybrid"], "score": 2, "note": "Wrong response language."}
        ],
    }
    _, selected, errors, judgments, _ = report.read_captures(original, recovery, pack, review)
    assert len(selected) == len(judgments) == 1
    assert errors[0]["error"]["kind"] == "timeout"
    assert ("case", "full_context") not in selected
    review["judgments"][0]["modes"].append("full_context")
    with pytest.raises(ValueError, match="no invented scores"):
        report.read_captures(original, recovery, pack, review)
