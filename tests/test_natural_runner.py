"""The paid runner is tested without inference or network access."""

import copy
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_natural_benchmark as runner  # noqa: E402

from hrlearnium.core_config import CoreSettings  # noqa: E402
from hrlearnium.course_metadata import metadata_response  # noqa: E402
from hrlearnium.language import language_scope  # noqa: E402
from hrlearnium.policy import conversational_reply  # noqa: E402
from hrlearnium.schemas import (  # noqa: E402
    CourseOutline,
    QueryRequest,
    QueryResponse,
    RetrievalOption,
    RetrievalOptions,
)


@pytest.fixture
def dataset():
    return runner.pack.validate_pack(runner.DEFAULT)


def test_scope_and_payload_have_no_gold(dataset):
    _, cases, _ = dataset
    assert len(cases) == 188
    assert sum(bool(c.get("previous_question")) for c in cases) == 10
    for case in cases:
        body = runner.request_payload(case["question"], "hybrid")
        assert set(body) == {"question", "response_mode", "retrieval_mode", "include_evaluation"}
        assert QueryRequest.model_validate(body).question == case["question"]


def test_failed_selection_keeps_only_flags_and_real_setup(tmp_path, dataset, monkeypatch):
    import report_natural_benchmark as reporting

    manifest, cases, _ = dataset
    parent = tmp_path / "parent"
    parent.mkdir()
    runner.write_json(parent / "index.json", {"unchanged": True})
    index = {
        "mode": "hybrid", "source": manifest["source"],
        "dataset_sha256": runner.digest(runner.DEFAULT / "cases.jsonl"),
        "manifest_sha256": runner.digest(runner.DEFAULT / "manifest.json"),
    }
    followup = next(c for c in cases if c.get("previous_question"))
    good = cases[0]
    report = {"outcomes": [
        {"id": good["id"], "reference": good, "checks": {"pass": True}},
        {"id": followup["id"], "reference": followup, "error": {"type": "failure"}},
    ]}
    monkeypatch.setattr(reporting, "load_run", lambda _: (index, report, "parent-hash"))
    selected, provenance = runner.failed_selection(parent, manifest, cases, runner.DEFAULT, "hybrid")
    assert selected == [followup] and selected[0]["previous_question"]
    assert provenance["selected_case_ids"] == [followup["id"]]
    assert provenance["parent_results_sha256"] == "parent-hash"
    report["outcomes"][1]["reference"] = {"changed": True}
    with pytest.raises(ValueError, match="reference differs"):
        runner.failed_selection(parent, manifest, cases, runner.DEFAULT, "hybrid")
    index["mode"] = "full_context"
    with pytest.raises(ValueError, match="Parent mode"):
        runner.failed_selection(parent, manifest, cases, runner.DEFAULT, "hybrid")


def test_missing_cap_never_creates_directory(tmp_path, dataset):
    manifest, cases, _ = dataset
    output = tmp_path / "new"
    cli = SimpleNamespace(output=output, max_query_turns=197)
    with pytest.raises(ValueError, match="198"):
        runner.execute(cli, manifest, cases, CoreSettings(_env_file=None))
    assert not output.exists()


def test_failure_redacts_exception_and_does_not_retry():
    calls = []

    class Broken:
        def query(self, *args):
            calls.append(args)
            raise RuntimeError("secret provider credential")

    result = runner.call(Broken(), None, "course", runner.request_payload("hi", "hybrid"))
    assert len(calls) == 1 and result["error"]["type"] == "RuntimeError"
    assert "secret" not in json.dumps(result) and result["response"] is None


def test_metadata_value_and_provenance_not_excerpt_requirement(dataset):
    manifest, cases, _ = dataset
    case = next(c for c in cases if c["id"] == "natural-089")
    # Build the exact imported outline from the reference list, without reading the LMS DB.
    titles_case = next(c for c in cases if c["parts"][0].get("metadata_field") == "chapter_titles")
    source = manifest["source"]
    outline = CourseOutline(
        course_id=source["course"],
        snapshot="test",
        warnings=[],
        documents=[
            {
                "document_id": source["document_id"],
                "version": source["version"],
                "source_sha256": source["source_sha256"],
                "title": source["title"],
                "warnings": [],
                "chapters": [
                    c | {"sections": []} for c in titles_case["parts"][0]["expected_value"]
                ],
            }
        ],
    )
    for case in [case, titles_case]:
        ref = case["parts"][0]
        with language_scope(case["question"]):
            response = metadata_response(
                outline,
                QueryRequest(question=case["question"]),
                request_id="test",
                conversation_id=str(uuid4()),
                mode="hybrid",
                operation=(ref["metadata_field"], ref.get("chapter_number")),
            )
        assert runner.reference_checks(case, response, source)["pass"]
        bad = copy.deepcopy(source)
        bad["source_sha256"] = "wrong"
        assert not runner.reference_checks(case, response, bad)["pass"]


def test_social_setup_expected_as_conversation(dataset):
    manifest, cases, _ = dataset
    case = next(c for c in cases if c["id"] == "natural-110")
    reason, answer = conversational_reply(case["previous_question"])
    response = QueryResponse(
        request_id="test",
        status="conversation",
        answer=answer,
        excerpts=[],
        conversation_id=str(uuid4()),
        reason_code=reason,
        retrieval_mode="hybrid",
    )
    setup = {
        "parts": [case["setup_reference"]],
        "expected_status": "conversation",
        "expected_answer_kind": "content",
        "expected_language": "fa",
    }
    assert runner.reference_checks(setup, response, manifest["source"])["pass"]


def test_runtime_does_not_claim_semantic_accuracy():
    result = runner.summarize([], 188)
    assert result["unattempted_cases"] == 188
    assert result["semantic_accuracy"] is None and not result["semantic_review_complete"]
    assert result["runtime"]["estimated_cost_usd"] is None


def test_partial_operational_errors_are_not_counted_as_successful_course_refusals():
    outcomes = [
        {
            "id": "test-case",
            "track": "multipart",
            "question": "Two questions?",
            "expected_status": "answered",
            "evaluation_status": "evaluated",
            "response": {
                "status": "partial",
                "parts": [
                    {"status": "answered", "reason_code": "supported"},
                    {"status": "error", "reason_code": "answer_verification_failed"},
                ],
            },
            "latency_ms": 1,
            "error": None,
        }
    ]
    summary = runner.summarize(outcomes, 1)
    assert summary["operational_errors"] == summary["cases_with_part_errors"] == 1
    assert summary["valid_response_bodies"] == 1 and summary["successful_responses"] == 0
    assert summary["request_exceptions"] == 0 and summary["failed_parts"] == 1


def test_metadata_diagnostics_are_separate_from_semantic_accuracy(dataset):
    manifest, cases, _ = dataset
    case = next(c for c in cases if c["id"] == "natural-110")
    reason, answer = conversational_reply(case["previous_question"])
    response = QueryResponse(
        request_id="test",
        status="conversation",
        answer=answer,
        excerpts=[],
        conversation_id=str(uuid4()),
        reason_code=reason,
        retrieval_mode="hybrid",
    )
    setup = {
        "parts": [case["setup_reference"]],
        "expected_status": "conversation",
        "expected_answer_kind": "content",
        "expected_language": "fa",
    }
    checks = runner.reference_checks(setup, response, manifest["source"])
    assert checks["pass"] and all(checks["diagnostic_categories"].values())
    assert checks["semantic_accuracy"] == "not_scored_requires_meaning_based_review"


def test_atomic_writes_preserve_existing_captures(tmp_path):
    path = tmp_path / "capture.json"
    runner.write_json(path, {"a": 1}, exclusive=True)
    with pytest.raises(FileExistsError):
        runner.write_json(path, {"a": 2}, exclusive=True)
    assert json.loads(path.read_text()) == {"a": 1}


def test_secret_not_serialized_in_fingerprint():
    settings = CoreSettings(_env_file=None, api_key="never expose this")
    value = runner.fingerprint(settings)
    assert len(value) == 64 and "never expose" not in value
    assert runner.fingerprint(settings.model_copy(update={"api_model": "changed"})) != value


@pytest.mark.parametrize("broken_conversation", [False, True])
def test_execution_setup_is_real_and_capture_precedes_assertions(
    tmp_path, dataset, monkeypatch, broken_conversation
):
    manifest, cases, _ = dataset
    case = next(c for c in cases if c["id"] == "natural-110")
    database = tmp_path / "source.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE untouched (value TEXT)")
    before = database.read_bytes()
    settings = CoreSettings(_env_file=None, database_path=database)
    monkeypatch.setattr(runner, "CoreSettings", lambda: settings)
    monkeypatch.setattr(runner, "read_source", lambda *args: (manifest["source"], []))
    seen = []
    conversation = str(uuid4())
    expected = runner.configuration(settings)

    class FakeService:
        closed = False

        def retrieval_options(self, *_):
            return RetrievalOptions(
                default_mode="hybrid",
                passage_count=44,
                embedded_passage_count=44,
                modes=[RetrievalOption(mode="hybrid", available=True)],
            )

        def query(self, principal, course, request):
            assert principal.scopes == ["query"] and course == manifest["source"]["course"]
            seen.append(request)
            if len(seen) == 1:
                reason, answer = conversational_reply(request.question)
                status = "conversation"
            else:
                from hrlearnium.policy import non_answer_reply

                reason, status = "incomplete_question_plan", "clarification"
                answer = non_answer_reply(request.question, status, reason)
            return QueryResponse(
                request_id="test",
                status=status,
                answer=answer,
                excerpts=[],
                conversation_id=str(uuid4())
                if broken_conversation and len(seen) == 2
                else conversation,
                reason_code=reason,
                retrieval_mode="hybrid",
                evaluation={
                    "configuration": expected | {"retrieval_mode": "hybrid"},
                    "stages_ms": {},
                    "provider_calls": [],
                },
            )

        def close(self):
            self.closed = True

    service = FakeService()

    def factory(run_settings):
        assert run_settings.database_path != database
        assert run_settings.database_path.read_bytes() == before
        return service

    cli = SimpleNamespace(
        pack=runner.DEFAULT,
        output=tmp_path / "run",
        mode="hybrid",
        max_query_turns=2,
        delay_seconds=0,
    )
    if broken_conversation:
        with pytest.raises(ValueError, match="conversation changed"):
            runner.execute(cli, manifest, [case], settings, factory=factory)
    else:
        index = runner.execute(cli, manifest, [case], settings, factory=factory)
        assert index["status"] == "complete" and index["attempted_query_turns"] == 2
        result = json.loads((cli.output / "results.json").read_text(encoding="utf-8"))
        assert result["outcomes"][0]["setup_checks"]["pass"]
        assert result["summary"]["automatic_checks_passed"] == 1
    assert len(seen) == 2 and seen[0].conversation_id is None
    assert str(seen[1].conversation_id) == conversation
    assert (cli.output / "natural-110--scored.json").exists()
    assert database.read_bytes() == before and service.closed
    with pytest.raises(FileExistsError):
        runner.execute(cli, manifest, [case], settings, factory=factory)
