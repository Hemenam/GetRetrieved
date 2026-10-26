"""Saved comparison reports require complete cohorts, unchanged captures and explicit reviews."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report_multipart_rerun as report  # noqa: E402


@pytest.fixture
def saved_run(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    cases = [{"id": f"case-{n}", "question": f"Q{n}", "parts": [{}, {}]} for n in range(6)]
    import json

    (inputs / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
    report.bench.write_json(
        inputs / "manifest.json", {"source": {"source_corpus_sha256": "source"}}
    )
    index = {
        "status": "complete",
        "pending_request": None,
        "planned_queries": 18,
        "attempted_queries": 18,
        "fixed_configuration": {},
        "captures": [],
        "dataset_sha256": report.digest(inputs / "cases.jsonl"),
        "manifest_sha256": report.digest(inputs / "manifest.json"),
    }
    judgments = []
    for c in cases:
        for m in report.MODES:
            path = tmp_path / f"{c['id']}--{m}.json"
            outcome = {
                "id": c["id"],
                "question": c["question"],
                "extension_definition": c,
                "evaluation": {"configuration": {}},
                "response": {"answer": "Saved"},
            }
            report.bench.write_json(
                path, {"outcomes": [outcome], "metadata": {"requested_retrieval_mode": m}}
            )
            index["captures"].append(
                {
                    "case_id": c["id"],
                    "mode": m,
                    "report": path.name,
                    "report_sha256": report.digest(path),
                }
            )
            judgments.append(
                {
                    "case_id": c["id"],
                    "mode": m,
                    "score": 2,
                    "note": "Reviewed issue.",
                    "facts_status_correct": True,
                    "language_correct": False,
                }
            )
    report.bench.write_json(tmp_path / "index.json", index)
    review = {
        "reviewer": "Assistant",
        "human_approved": False,
        "judgments": judgments,
        "index_sha256": report.digest(tmp_path / "index.json"),
    }
    return tmp_path, review


def test_complete_cohort_binds_explicit_review_to_raw_capture_hashes(saved_run):
    directory, review = saved_run
    _, cases, outcomes = report.load_run(directory)
    judgments = report.bind_review(review, directory / "index.json", outcomes)
    assert len(cases) == 6 and len(outcomes) == len(judgments) == 18
    assert all(j["score"] == 2 and j["capture_sha256"] for j in judgments.values())


def test_missing_review_is_not_inferred_from_automatic_checks(saved_run):
    directory, review = saved_run
    _, _, outcomes = report.load_run(directory)
    review["judgments"].pop()
    with pytest.raises(ValueError, match="Missing explicit review"):
        report.bind_review(review, directory / "index.json", outcomes)


def test_tampered_capture_cannot_be_reported(saved_run):
    directory, _ = saved_run
    (directory / "case-0--hybrid.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Capture hash changed"):
        report.load_run(directory)


def test_incomplete_run_cannot_be_called_complete(saved_run):
    directory, _ = saved_run
    import json

    path = directory / "index.json"
    index = json.loads(path.read_text())
    index.update(status="running", pending_request={"case_id": "case-0", "mode": "hybrid"})
    report.bench.write_json(path, index, exclusive=False)
    with pytest.raises(ValueError, match="not fully captured"):
        report.load_run(directory)
