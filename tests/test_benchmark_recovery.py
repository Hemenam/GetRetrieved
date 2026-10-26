"""Synthetic recovery safeguards; no provider calls or model grading."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_recovery", ROOT / "scripts/recover_benchmark_errors.py"
)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def capture(tmp_path):
    cases = [
        {
            "id": f"q{i}",
            "question_family_id": f"f{i}",
            "category": "direct",
            "question": f"Question {i}?",
            "expected_status": "answered",
            "expected_language": "en",
            "expected_chapters": [1],
            "required_quotes": ["Evidence"],
            "required_facts": [{"id": "fact", "text": "Meaning", "evidence_quotes": ["Evidence"]}],
            "notes": "Synthetic",
            "review_status": "approved",
            "reviewer": "Synthetic owner",
            "split": "development",
        }
        for i in (1, 2)
    ]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
    manifest = {
        "format_version": 1,
        "name": "Synthetic recovery",
        "modes": ["hybrid", "full_context", "hybrid_rerank"],
        "repetitions": 1,
        "seed": 42,
        "response_mode": "explained",
        "max_query_turns": 6,
        "documents": [
            {
                "id": "course",
                "tenant": "t",
                "course": "c",
                "dataset": "cases.jsonl",
                "split": "development",
                "source_corpus_sha256": "a" * 64,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    put(path, manifest)
    run = tmp_path / "old"
    run.mkdir()
    entries = []
    for mode in manifest["modes"]:
        outcomes = []
        for case in cases:
            failed = mode == "hybrid_rerank" or (mode == "full_context" and case["id"] == "q2")
            outcome = {
                "id": case["id"],
                "question": case["question"],
                "evaluation_status": "error" if failed else "evaluated",
                "actual_status": None
                if failed
                else "refused"
                if mode == "full_context"
                else "answered",
                "passed_automated_checks": not failed and mode != "full_context",
            }
            if failed:
                outcome["error"] = {"kind": "http_status", "status_code": 503, "stage": "query"}
            else:
                outcome["evaluation"] = {
                    "configuration": {
                        "backend": "synthetic",
                        "retrieval_mode": mode,
                        "reranker": None,
                    },
                    "source_corpus_sha256": "a" * 64,
                }
            outcomes.append(outcome)
        report_path = run / f"{mode}.json"
        put(
            report_path,
            {"dataset_sha256": recovery.digest(tmp_path / "cases.jsonl"), "outcomes": outcomes},
        )
        entries.append(
            {
                "mode": mode,
                "report": report_path.name,
                "report_sha256": recovery.digest(report_path),
            }
        )
    put(
        run / "index.json",
        {"status": "stopped", "manifest_sha256": recovery.digest(path), "runs": entries},
    )
    return run, path


def test_plan_only_retries_operational_errors_not_semantic_failures(capture):
    run, path = capture
    _, _, jobs, config = recovery.recovery_plan(run, path, 3)
    assert [(j["mode"], j["case"]["id"]) for j in jobs] == [
        ("full_context", "q2"),
        ("hybrid_rerank", "q1"),
        ("hybrid_rerank", "q2"),
    ]
    assert config == {"backend": "synthetic"}


def test_plan_enforces_hard_request_cap(capture):
    with pytest.raises(ValueError, match="request cap"):
        recovery.recovery_plan(*capture, 2)


def test_plan_rejects_modified_source_report(capture):
    run, path = capture
    (run / "hybrid.json").write_text("{}")
    with pytest.raises(ValueError, match="capture changed"):
        recovery.recovery_plan(run, path, 3)


def test_plan_rejects_changed_manifest(capture):
    run, path = capture
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest differs"):
        recovery.recovery_plan(run, path, 3)


@pytest.mark.parametrize("first_error", [True, False])
def test_execution_checkpoints_and_stops_on_first_operational_error(
    capture, tmp_path, monkeypatch, first_error
):
    run, path = capture
    manifest, doc, jobs, config = recovery.recovery_plan(run, path, 3)
    original_index = (run / "index.json").read_bytes()
    monkeypatch.setattr(
        recovery, "Settings", lambda: SimpleNamespace(database_path=tmp_path / "unused")
    )
    monkeypatch.setattr(recovery, "configuration", lambda settings: config)
    monkeypatch.setattr(
        recovery, "read_source", lambda *a: ({"source_corpus_sha256": "a" * 64}, [])
    )
    monkeypatch.setattr(recovery.bench, "preflight", lambda *a: [])
    monkeypatch.setattr(recovery, "review_template", lambda path: {"reviewer": "", "cases": []})
    monkeypatch.setattr(recovery, "runtime_summary", lambda outcomes: {})
    monkeypatch.setattr(recovery.evaluate, "api_summary", lambda outcomes: {})
    calls = []

    def fake_query(cases, args):
        calls.append((args.retrieval_mode, cases[0]["id"]))
        outcome = {
            "id": cases[0]["id"],
            "evaluation_status": "evaluated",
            "actual_status": "refused",
            "evaluation": {"configuration": config, "source_corpus_sha256": "a" * 64},
        }
        if first_error:
            outcome.update(
                evaluation_status="error",
                actual_status=None,
                error={"kind": "http_status", "status_code": 503, "stage": "query"},
            )
        return [outcome], {"evaluation_kind": "api"}

    monkeypatch.setattr(recovery.evaluate, "api_evaluation", fake_query)
    cli = argparse.Namespace(
        source_run=run,
        manifest=path,
        output=tmp_path / "new",
        max_query_turns=3,
        delay_seconds=0,
        base_url="http://127.0.0.1:8002",
        local_auth=True,
        subject="synthetic",
        timeout=30,
    )
    index = recovery.execute(cli, manifest, doc, jobs, config)
    assert (
        len(calls)
        == len(index["captures"])
        == index["attempted_requests"]
        == (1 if first_error else 3)
    )
    assert index["pending_request"] is None
    assert index["status"] == ("stopped_on_operational_error" if first_error else "complete")
    assert (run / "index.json").read_bytes() == original_index
    assert json.loads((cli.output / "index.json").read_text())["status"] == index["status"]
    for entry in index["captures"]:
        assert recovery.digest(cli.output / entry["report"]) == entry["report_sha256"]


def test_config_drift_stops_before_network_or_output_creation(capture, tmp_path, monkeypatch):
    run, path = capture
    manifest, doc, jobs, config = recovery.recovery_plan(run, path, 3)
    monkeypatch.setattr(recovery, "Settings", lambda: object())
    monkeypatch.setattr(recovery, "configuration", lambda settings: {"backend": "changed"})
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="application code changed"):
        recovery.execute(SimpleNamespace(output=output), manifest, doc, jobs, config)
    assert not output.exists()
