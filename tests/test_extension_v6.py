"""Extension benchmark safeguards and part-aware evidence checks; no paid calls."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import recover_extension_v6 as recovery  # noqa: E402
import run_extension_v6 as runner  # noqa: E402


def test_partial_reference_preserves_supported_anchors_without_claiming_owner_approval():
    case = {
        "id": "test",
        "question": "Two questions?",
        "category": "mixed",
        "expected_status": "partial",
        "expected_language": "en",
        "parts": [{"quotes": ["source quote"]}, {"quotes": []}],
    }
    adapted = runner.api_case(case)
    assert adapted["expected_status"] == "partial"
    assert adapted["required_quotes"] == ["source quote"]
    assert adapted["review_status"] == "assistant_source_audited"


def test_anchor_in_wrong_part_cannot_satisfy_coverage():
    case = {
        "parts": [
            {"status": "answered", "kind": "content", "quotes": ["first evidence"]},
            {"status": "answered", "kind": "content", "quotes": ["second evidence"]},
        ]
    }
    outcome = {
        "passed_automated_checks": True,
        "response": {
            "answer_kind": "multi",
            "parts": [
                {"status": "answered", "answer_kind": "content", "excerpt_ids": ["b"]},
                {"status": "answered", "answer_kind": "content", "excerpt_ids": ["a"]},
            ],
            "excerpts": [
                {"id": "a", "text": "first evidence"},
                {"id": "b", "text": "second evidence"},
            ],
        },
    }
    assert runner.extension_checks(case, outcome, [])["pass"] is False
    outcome["response"]["parts"][0]["excerpt_ids"] = ["a"]
    outcome["response"]["parts"][1]["excerpt_ids"] = ["b"]
    assert runner.extension_checks(case, outcome, [])["pass"] is True


@pytest.fixture
def execution(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    for name in ("manifest.json", "cases.jsonl", "catalog.json"):
        (pack / name).write_text("{}", encoding="utf-8")
    source = {"source_corpus_sha256": "source"}
    settings = SimpleNamespace(database_path=tmp_path / "unused.db")
    manifest = {
        "reference_review": "assistant_audit",
        "cases_sha256": runner.digest(pack / "cases.jsonl"),
        "catalog_sha256": runner.digest(pack / "catalog.json"),
        "source": source,
        "course": "course",
        "tenant": "tenant",
        "seed": 42,
        "modes": ["full_context", "hybrid", "hybrid_rerank"],
        "response_mode": "explained",
    }
    cases = [
        {
            "id": f"q{i}",
            "question": "Question?",
            "category": "metadata",
            "expected_status": "answered",
            "expected_language": "en",
            "parts": [{"status": "answered", "kind": "content", "quotes": [], "facts": []}],
        }
        for i in range(12)
    ]
    cli = SimpleNamespace(
        pack=pack,
        output=tmp_path / "out",
        max_query_turns=36,
        delay_seconds=0,
        base_url="http://127.0.0.1:8002",
        local_auth=True,
        timeout=1,
        subject="synthetic",
    )
    monkeypatch.setattr(runner, "Settings", lambda: settings)
    monkeypatch.setattr(runner, "configuration", lambda *args: {})
    monkeypatch.setattr(runner, "read_source", lambda *args: (source, []))
    monkeypatch.setattr(runner.bench, "preflight", lambda *args: [])
    calls, failures = [], []

    def evaluate(cases, args):
        calls.append((cases, args))
        value = {
            "expected_status": "answered",
            "evaluation_status": "evaluated",
            "actual_status": "refused",
            "response": {"status": "refused", "answer_kind": "content", "excerpts": []},
            "passed_automated_checks": False,
            "evaluation": {"configuration": {}},
        }
        if failures:
            value.update(
                error={"kind": "http_status", "status_code": 503}, evaluation_status="error"
            )
        return [value], {}

    monkeypatch.setattr(runner.evaluate, "api_evaluation", evaluate)
    return cli, manifest, cases, settings, calls, failures


def test_semantic_failure_is_not_retried_and_each_capture_is_saved(execution):
    cli, manifest, cases, settings, calls, _ = execution
    index = runner.execute(cli, manifest, cases, settings, [])
    assert len(calls) == index["attempted_queries"] == len(index["captures"]) == 36
    assert index["status"] == "complete" and index["pending_request"] is None
    assert len({(e["case_id"], e["mode"]) for e in index["captures"]}) == 36


def test_operational_failure_stops_after_one_attempt(execution):
    cli, manifest, cases, settings, calls, failures = execution
    failures.append(True)
    index = runner.execute(cli, manifest, cases, settings, [])
    assert len(calls) == 1 and index["status"] == "stopped"
    assert len(index["captures"]) == 1 and index["pending_request"] is None


def test_cap_and_existing_directory_cannot_replay_requests(execution):
    cli, manifest, cases, settings, calls, _ = execution
    cli.max_query_turns = 35
    with pytest.raises(ValueError, match="36-query"):
        runner.execute(cli, manifest, cases, settings, [])
    cli.max_query_turns = 36
    cli.output.mkdir()
    with pytest.raises(FileExistsError):
        runner.execute(cli, manifest, cases, settings, [])
    assert calls == []


def test_recovery_excludes_semantic_failures_and_preserves_original(tmp_path):
    manifest = {"cases_sha256": "dataset", "seed": 42, "modes": ["hybrid", "full_context"]}
    captures = []
    for mode, outcome in [
        ("hybrid", {"actual_status": "refused", "passed_automated_checks": False}),
        ("full_context", {"error": {"kind": "http_status", "status_code": 429, "stage": "query"}}),
    ]:
        path = tmp_path / f"{mode}.json"
        runner.bench.write_json(path, {"outcomes": [outcome]})
        captures.append(
            {
                "case_id": "case",
                "mode": mode,
                "report": path.name,
                "report_sha256": runner.digest(path),
            }
        )
    index = {
        "status": "stopped",
        "pending_request": None,
        "dataset_sha256": "dataset",
        "attempted_queries": 2,
        "captures": captures,
    }
    runner.bench.write_json(tmp_path / "index.json", index)
    before = runner.digest(tmp_path / "index.json")
    _, schedule = recovery.remaining(tmp_path, manifest, [{"id": "case"}])
    assert [(c["id"], m) for c, m in schedule] == [("case", "full_context")]
    assert runner.digest(tmp_path / "index.json") == before
    (tmp_path / "hybrid.json").write_text(json.dumps({"outcomes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        recovery.remaining(tmp_path, manifest, [{"id": "case"}])
