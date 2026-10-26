"""Offline benchmark orchestration tests. Synthetic responses are not model scores."""

import argparse
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_runner", Path(__file__).resolve().parents[1] / "scripts/benchmark.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def put(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture_pack(tmp_path, *, approved=True, two_documents=False):
    positive = {
        "id": "a",
        "question_family_id": "family-a",
        "category": "direct",
        "question": "Question?",
        "expected_status": "answered",
        "expected_language": "en",
        "expected_chapters": [1],
        "required_quotes": ["Source fact."],
        "notes": "Synthetic, not a real instructor label.",
        "required_facts": [{"id": "f", "text": "Fact.", "evidence_quotes": ["Source fact."]}],
        "forbidden_claims": [],
        "review_status": "approved" if approved else "proposed",
        "reviewer": "test-instructor" if approved else "",
        "split": "development",
    }
    negative = positive | {
        "id": "b",
        "question_family_id": "family-b",
        "category": "unsupported",
        "expected_status": "refused",
        "required_quotes": [],
        "required_facts": [],
        "expected_chapters": [],
    }
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text("\n".join(json.dumps(c) for c in (positive, negative)), encoding="utf-8")
    documents = [
        {
            "id": "one",
            "tenant": "test",
            "course": "course-one",
            "dataset": "cases.jsonl",
            "split": "development",
        }
    ]
    if two_documents:
        documents.append(documents[0] | {"id": "two", "course": "course-two"})
    manifest = {
        "format_version": 1,
        "name": "Synthetic test",
        "modes": sorted(bench.MODES),
        "repetitions": 2,
        "seed": 42,
        "response_mode": "explained",
        "max_query_turns": 100,
        "documents": documents,
    }
    path = tmp_path / "manifest.json"
    put(path, manifest)
    return path


def cli_args(tmp_path):
    return argparse.Namespace(
        output=tmp_path / "run",
        pricing=None,
        base_url="http://127.0.0.1:8002",
        local_auth=False,
        subject="synthetic",
        timeout=1,
        delay_seconds=0,
    )


def synthetic_evaluation(cases, args):
    outcomes = []
    for case in cases:
        answered = case["expected_status"] == "answered"
        response = {
            "status": case["expected_status"],
            "answer": "Fact." if answered else "No explanation.",
            "excerpts": [{"id": "e1", "text": "Source fact."}] if answered else [],
            "explanation": {"statements": [{"text": "Fact.", "citation_ids": ["e1"]}]}
            if answered
            else None,
        }
        outcome = bench.evaluate.base_outcome(case) | {
            "evaluation_status": "evaluated",
            "actual_status": case["expected_status"],
            "status_matches": True,
            "retrieval_mode": args.retrieval_mode,
            "requested_response_mode": "explained",
            "response": response,
            "latency_ms": 12,
            "query_latency_ms": 10,
            "passed_automated_checks": True,
            "evaluation": {
                "configuration": {
                    "chat_model": "synthetic",
                    "retrieval_mode": args.retrieval_mode,
                    "reranker": "test" if args.retrieval_mode == "hybrid_rerank" else None,
                },
                "source_corpus_sha256": args.course,
                "source_passage_count": 1,
                "stages_ms": {},
                "provider_calls": [],
            },
        }
        outcome.update(
            bench.evaluate.coverage(
                case, ["Source fact."] if answered else [], [1] if answered else []
            )
        )
        outcomes.append(outcome)
    return outcomes, {
        "evaluation_kind": "api",
        "course": args.course,
        "response_mode": args.response_mode,
        "tenant_label": args.tenant,
    }


def fake_run(monkeypatch, tmp_path, **kwargs):
    path = fixture_pack(tmp_path, **kwargs)
    manifest, documents = bench.load_manifest(path)
    cli = cli_args(tmp_path)
    monkeypatch.setattr(bench, "preflight", lambda *a: [])
    monkeypatch.setattr(bench.evaluate, "api_evaluation", synthetic_evaluation)
    bench.run(cli, path, manifest, documents)
    return cli.output


def complete_reviews(directory, *, fail_language=False):
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    for entry in index["runs"]:
        path = directory / entry["review"]
        review = json.loads(path.read_text(encoding="utf-8"))
        review["reviewer"] = "synthetic-grader"
        for case in review["cases"]:
            case.update(
                reviewed=True,
                status_correct=True,
                language_correct=not fail_language,
                answer_relevant=True,
                answer_complete=True,
                example_context_preserved=True,
            )
            for fact in case["fact_checks"]:
                fact["covered"] = True
            for statement in case["statement_checks"]:
                statement.update(supported=True, citations_support_statement=True)
            for claim in case["forbidden_claim_checks"]:
                claim["present"] = False
        put(path, review)


def test_pilot_has_twenty_owner_approved_development_cases_and_198_turns():
    path = bench.ROOT / "evals/pilot/manifest.json"
    manifest, docs = bench.load_manifest(path)
    planned = bench.plan(manifest, docs)
    assert planned["scored_turns"] == 180
    assert planned["query_turns_upper_bound"] == 198
    assert planned["execution_ready"]
    assert len(docs[0]["cases"]) == 20
    approval = json.loads((path.parent / "approval.json").read_text(encoding="utf-8"))
    assert all(
        c["review_status"] == "approved"
        and c["reviewer"] == approval["reviewer"]
        and c["split"] == "development"
        for c in docs[0]["cases"]
    )
    assert approval["case_ids"] == [c["id"] for c in docs[0]["cases"]]
    assert approval["approved_dataset_sha256"] == docs[0]["dataset_sha256"]
    assert approval["source_corpus_sha256"] == docs[0]["source_corpus_sha256"]
    assert approval["reviewer_role"] == "project_owner"
    assert not approval["instructor_credentials_asserted"]
    assert not approval["expected_results_changed"]


def test_default_run_and_plan_are_network_free(monkeypatch, tmp_path):
    path = fixture_pack(tmp_path)
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("No network in a plan"))
    monkeypatch.setattr(bench, "run", lambda *a: pytest.fail("--execute was not supplied"))
    assert bench.main(["plan", "--manifest", str(path)]) == 0
    assert bench.main(["run", "--manifest", str(path)]) == 0


@pytest.mark.parametrize("mutation", ["pending", "cap"])
def test_execution_blocked_before_network_or_output(monkeypatch, tmp_path, mutation):
    path = fixture_pack(tmp_path, approved=mutation != "pending")
    if mutation == "cap":
        payload = json.loads(path.read_text()) | {"max_query_turns": 1}
        put(path, payload)
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: pytest.fail("Blocked execution cannot use network")
    )
    code = bench.main(
        ["run", "--execute", "--manifest", str(path), "--output", str(tmp_path / "run")]
    )
    assert code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "duplicate-mode", "duplicate-document", "boolean-repeats", "bad-fingerprint"],
)
def test_manifest_validation(tmp_path, mutation):
    path = fixture_pack(tmp_path)
    value = json.loads(path.read_text())
    if mutation == "unknown":
        value["unknown"] = 1
    elif mutation == "duplicate-mode":
        value["modes"].append(value["modes"][0])
    elif mutation == "duplicate-document":
        value["documents"].append(deepcopy(value["documents"][0]))
    elif mutation == "boolean-repeats":
        value["repetitions"] = True
    else:
        value["documents"][0]["source_corpus_sha256"] = "invalid"
    put(path, value)
    with pytest.raises(ValueError):
        bench.load_manifest(path)


def test_question_family_cannot_leak_between_splits(tmp_path):
    path = fixture_pack(tmp_path)
    dataset = tmp_path / "cases.jsonl"
    case = json.loads(dataset.read_text().splitlines()[0]) | {"id": "c", "split": "held_out"}
    with dataset.open("a") as handle:
        handle.write("\n" + json.dumps(case))
    with pytest.raises(ValueError, match="family"):
        bench.load_manifest(path)


def test_development_and_heldout_documents_cannot_share_a_macro_score(tmp_path):
    path = fixture_pack(tmp_path, two_documents=True)
    dataset = tmp_path / "cases.jsonl"
    cases = [json.loads(line) for line in dataset.read_text().splitlines()]
    with dataset.open("a") as handle:
        for case in cases:
            handle.write(
                "\n"
                + json.dumps(
                    case
                    | {
                        "id": case["id"] + "-held",
                        "split": "held_out",
                        "question_family_id": case["question_family_id"] + "-held",
                    }
                )
            )
    manifest = json.loads(path.read_text())
    manifest["documents"][1]["split"] = "held_out"
    put(path, manifest)
    with pytest.raises(ValueError, match="same split"):
        bench.load_manifest(path)


def test_synthetic_two_document_experiment_and_bound_human_review(monkeypatch, tmp_path):
    directory = fake_run(monkeypatch, tmp_path, two_documents=True)
    index = json.loads((directory / "index.json").read_text())
    assert index["status"] == "complete" and len(index["runs"]) == 12
    assert (directory / "inputs/one.jsonl").exists()
    pending = bench.summarize(directory)
    assert pending["comparable"] and not pending["quality_review_complete"]
    assert all(r["complete_grounded_answer_rate"] is None for r in pending["documents"])
    assert all(r["observed_rank"] is None for r in pending["macro_leaderboard"])
    complete_reviews(directory)
    graded = bench.summarize(directory)
    assert graded["quality_review_complete"]
    assert all(r["documents"] == 2 and r["observed_rank"] == 1 for r in graded["macro_leaderboard"])
    assert graded["winner"] is None
    assert all(r["estimated_cost_usd"] is None for r in graded["documents"])
    bench.export_summary(tmp_path / "summary", graded)
    assert (
        (tmp_path / "summary/leaderboard.csv")
        .read_text(encoding="utf-8-sig")
        .startswith("document_id,")
    )
    with pytest.raises(FileExistsError):
        bench.export_summary(tmp_path / "summary", graded)


def test_language_failure_changes_quality_not_source_checks(monkeypatch, tmp_path):
    directory = fake_run(monkeypatch, tmp_path)
    complete_reviews(directory, fail_language=True)
    result = bench.summarize(directory)
    assert all(r["complete_grounded_answer_rate"]["rate"] == 0 for r in result["documents"])
    assert all(r["automated_checks_passed"]["rate"] == 1 for r in result["documents"])


def test_macro_quality_weights_documents_not_question_counts(monkeypatch, tmp_path):
    path = fixture_pack(tmp_path, two_documents=True)
    manifest = json.loads(path.read_text())
    cases = [json.loads(line) for line in (tmp_path / "cases.jsonl").read_text().splitlines()]
    larger = cases + [cases[0] | {"id": f"extra-{i}"} for i in range(2)]
    (tmp_path / "larger.jsonl").write_text(
        "\n".join(json.dumps(case) for case in larger), encoding="utf-8"
    )
    manifest["documents"][1]["dataset"] = "larger.jsonl"
    put(path, manifest)
    manifest, docs = bench.load_manifest(path)
    monkeypatch.setattr(bench, "preflight", lambda *a: [])
    monkeypatch.setattr(bench.evaluate, "api_evaluation", synthetic_evaluation)
    bench.run(cli_args(tmp_path), path, manifest, docs)
    directory = tmp_path / "run"
    complete_reviews(directory)
    index = json.loads((directory / "index.json").read_text())
    for entry in index["runs"]:
        if entry["document_id"] == "one":
            review_path = directory / entry["review"]
            review = json.loads(review_path.read_text())
            review["cases"][0]["answer_complete"] = False
            put(review_path, review)
    result = bench.summarize(directory)
    assert all(r["macro_complete_grounded_answer_rate"] == 0.5 for r in result["macro_leaderboard"])


def test_refuse_existing_run_directory_without_new_requests(monkeypatch, tmp_path):
    directory = fake_run(monkeypatch, tmp_path)
    path = tmp_path / "manifest.json"
    manifest, docs = bench.load_manifest(path)
    monkeypatch.setattr(bench, "preflight", lambda *a: pytest.fail("No network on replay"))
    with pytest.raises(FileExistsError):
        bench.run(cli_args(tmp_path), path, manifest, docs)
    assert (directory / "index.json").exists()


def test_source_drift_stops_experiment_and_preserves_completed_reports(monkeypatch, tmp_path):
    path = fixture_pack(tmp_path)
    manifest, docs = bench.load_manifest(path)
    calls = []

    def changing(cases, args):
        outcomes, metadata = synthetic_evaluation(cases, args)
        for item in outcomes:
            item["evaluation"]["source_corpus_sha256"] = str(len(calls))
        calls.append(args.retrieval_mode)
        return outcomes, metadata

    monkeypatch.setattr(bench, "preflight", lambda *a: [])
    monkeypatch.setattr(bench.evaluate, "api_evaluation", changing)
    with pytest.raises(ValueError, match="source must match"):
        bench.run(cli_args(tmp_path), path, manifest, docs)
    index = json.loads((tmp_path / "run/index.json").read_text())
    assert index["status"] == "stopped" and len(index["runs"]) == 2
    assert len(calls) == 2
    assert not bench.summarize(tmp_path / "run")["comparable"]


def test_changed_report_cannot_reuse_a_review(monkeypatch, tmp_path):
    directory = fake_run(monkeypatch, tmp_path)
    index = json.loads((directory / "index.json").read_text())
    path = directory / index["runs"][0]["report"]
    payload = json.loads(path.read_text())
    payload["outcomes"][0]["response"]["answer"] = "Changed"
    put(path, payload)
    with pytest.raises(ValueError, match="report changed"):
        bench.summarize(directory)


def test_operational_error_stays_in_answerable_denominator(monkeypatch, tmp_path):
    path = fixture_pack(tmp_path)
    manifest, docs = bench.load_manifest(path)

    def with_error(cases, args):
        outcomes, metadata = synthetic_evaluation(cases, args)
        outcomes[0].update(
            evaluation_status="error",
            actual_status=None,
            status_matches=False,
            passed_automated_checks=False,
        )
        return outcomes, metadata

    monkeypatch.setattr(bench, "preflight", lambda *a: [])
    monkeypatch.setattr(bench.evaluate, "api_evaluation", with_error)
    bench.run(cli_args(tmp_path), path, manifest, docs)
    complete_reviews(tmp_path / "run")
    result = bench.summarize(tmp_path / "run")
    assert all(
        r["complete_grounded_answer_rate"] == {"numerator": 0, "denominator": 2, "rate": 0}
        for r in result["documents"]
    )
    assert all(r["operational_errors"] == 2 for r in result["documents"])
    assert all(r["observed_rank"] is None for r in result["macro_leaderboard"])


def test_prerequisites_use_authenticated_get_only(monkeypatch, tmp_path):
    path = fixture_pack(tmp_path)
    manifest, docs = bench.load_manifest(path)
    real_client = httpx.Client
    calls = []
    monkeypatch.setenv("HR_EVAL_TOKEN", "synthetic-token")

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "default_mode": "full_context",
                "passage_count": 2,
                "embedded_passage_count": 2,
                "modes": [{"mode": m, "available": True} for m in bench.MODES],
            },
        )

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw)
    )
    result = bench.preflight(cli_args(tmp_path), manifest, docs)
    assert len(calls) == 1 and calls[0].method == "GET"
    assert calls[0].headers["Authorization"] == "Bearer synthetic-token"
    assert result[0]["options"]["provider_verified"] is False
