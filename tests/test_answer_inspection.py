"""Offline cost arithmetic and bounded diagnostic-repeat safeguards."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import inspect_chapter_answers as inspection  # noqa: E402
import rerun_benchmark_case as diagnostic  # noqa: E402


def reports_with_calls(calls):
    return {"hybrid": {"outcomes": [{"evaluation": {"provider_calls": calls}}]}}


def test_cost_separates_chat_embedding_and_output_tokens():
    rows = inspection.cost_rows(
        reports_with_calls(
            [
                {
                    "model": "gpt-4.1-mini",
                    "kind": "chat/completions",
                    "stage": "selection",
                    "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 2_000_000},
                },
                {
                    "model": "text-embedding-3-small",
                    "kind": "embeddings",
                    "stage": "embedding",
                    "usage": {"prompt_tokens": 1_000_000},
                },
            ]
        )
    )
    row = rows[0]
    assert row["llm_usd"] == pytest.approx(3.6)
    assert row["embedding_usd"] == pytest.approx(0.02)
    assert row["standard_list_usd"] == pytest.approx(3.62)
    assert row["per_1000_queries_usd"] == pytest.approx(3620)


@pytest.mark.parametrize(
    "call",
    [
        {
            "model": "unknown",
            "kind": "chat/completions",
            "stage": "selection",
            "usage": {"prompt_tokens": 1},
        },
        {"model": "gpt-4.1-mini", "kind": "chat/completions", "stage": "selection", "usage": {}},
    ],
)
def test_cost_does_not_invent_usage_or_price_for_unknown_model(call):
    with pytest.raises(ValueError):
        inspection.cost_rows(reports_with_calls([call]))


@pytest.fixture
def repeat_setup(tmp_path, monkeypatch):
    source, recovery = tmp_path / "original", tmp_path / "recovery"
    source.mkdir()
    recovery.mkdir()
    (source / "index.json").write_text('{"status":"stopped"}')
    (recovery / "index.json").write_text('{"status":"complete"}')
    output = tmp_path / "diagnostic"
    case = {"id": "coverage-07-explanation", "question": "Question?"}
    original = {"actual_status": "refused"}
    monkeypatch.setattr(
        diagnostic,
        "select_case",
        lambda *args: (
            {"modes": ["full_context"]},
            {"dataset_sha256": "d"},
            case,
            original,
            {"source_corpus_sha256": "s"},
        ),
    )
    monkeypatch.setattr(diagnostic.bench, "preflight", lambda *args: None)
    monkeypatch.setattr(diagnostic.bench, "request_args", lambda *args: object())
    monkeypatch.setattr(diagnostic.evaluate, "api_summary", lambda outcomes: {})
    calls = []
    outcome = {
        "expected_status": "answered",
        "evaluation_status": "evaluated",
        "actual_status": "answered",
        "passed_automated_checks": True,
    }

    def capture(cases, args):
        calls.append(cases)
        return [outcome], {}

    monkeypatch.setattr(diagnostic.evaluate, "api_evaluation", capture)
    argv = [
        "diagnostic",
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--source-run",
        str(source),
        "--recovery-run",
        str(recovery),
        "--case-id",
        case["id"],
        "--mode",
        "full_context",
        "--output",
        str(output),
    ]
    return argv, output, source, calls, outcome, case


def test_diagnostic_plan_is_offline(repeat_setup, monkeypatch, capsys):
    argv, output, _, calls, _, _ = repeat_setup
    monkeypatch.setattr(sys, "argv", argv)
    diagnostic.main()
    assert not calls and not output.exists()
    assert json.loads(capsys.readouterr().out)["network_requests"] == 0


def test_explicit_repeat_sends_one_case_and_keeps_original(repeat_setup, monkeypatch):
    argv, output, source, calls, _, case = repeat_setup
    before = (source / "index.json").read_bytes()
    monkeypatch.setattr(sys, "argv", argv + ["--execute"])
    diagnostic.main()
    index = json.loads((output / "index.json").read_text())
    assert calls == [[case]]
    assert index["attempted_queries"] == 1
    assert index["original_status"] == "refused"
    assert index["status"] == "complete" and index["pending_request"] is False
    assert (source / "index.json").read_bytes() == before


def test_existing_diagnostic_output_cannot_replay_paid_call(repeat_setup, monkeypatch):
    argv, output, _, calls, _, _ = repeat_setup
    output.mkdir()
    monkeypatch.setattr(sys, "argv", argv + ["--execute"])
    with pytest.raises(FileExistsError):
        diagnostic.main()
    assert calls == []


def test_operational_error_is_saved_without_retry(repeat_setup, monkeypatch):
    argv, output, _, calls, outcome, _ = repeat_setup
    outcome.update(evaluation_status="error", actual_status=None, error={"kind": "timeout"})
    monkeypatch.setattr(sys, "argv", argv + ["--execute"])
    diagnostic.main()
    index = json.loads((output / "index.json").read_text())
    assert len(calls) == 1 and index["status"] == "stopped"
    assert index["pending_request"] is False
