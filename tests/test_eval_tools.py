"""Human grading and fair experiment comparison, with synthetic responses only."""

import json
from copy import deepcopy

import pytest

from hrlearnium.config import Settings
from hrlearnium.eval_tools import (
    annotation_template,
    compare_reports,
    review_template,
    runtime_summary,
    score_review,
    validate_annotations,
)
from hrlearnium.evaluation import collect, provider_usage, stage


def report(mode="full_context"):
    return {
        "dataset_sha256": "a" * 64,
        "metadata": {"evaluation_kind": "api", "course": "one", "response_mode": "explained"},
        "summary": {"cases": 1},
        "latency_ms": {"median": 10},
        "outcomes": [
            {
                "id": "case-1",
                "question": "سوال؟",
                "expected_status": "answered",
                "evaluation_status": "evaluated",
                "actual_status": "answered",
                "passed_automated_checks": True,
                "required_coverage_complete": True,
                "retrieval_mode": mode,
                "gold": {
                    "review_status": "approved",
                    "reviewer": "instructor",
                    "required_facts": [
                        {"id": "f1", "text": "مفهوم اول", "evidence_quotes": ["متن اول"]}
                    ],
                    "forbidden_claims": ["توصیه ساختگی"],
                },
                "response": {
                    "status": "answered",
                    "answer": "پاسخ اول",
                    "excerpts": [{"id": "e1", "text": "متن اول", "citation": {}}],
                    "explanation": {"statements": [{"text": "مفهوم اول", "citation_ids": ["e1"]}]},
                },
                "evaluation": {
                    "configuration": {
                        "chat_model": "same-chat",
                        "embedding_model": "same-embedding",
                        "pipeline_sha256": "code-hash",
                        "prompt_sha256": "prompt-hash",
                        "reranker": {"model": "cross-encoder"} if mode == "hybrid_rerank" else None,
                    },
                    "source_corpus_sha256": "source-hash",
                    "source_passage_count": 44,
                    "stages_ms": {"selection": 10.0},
                    "provider_calls": [],
                },
            }
        ],
    }


def write_report(tmp_path, mode="full_context", name="report.json"):
    path = tmp_path / name
    path.write_text(json.dumps(report(mode), ensure_ascii=False), encoding="utf-8")
    return path


def complete_review(path):
    review = review_template(path)
    review["reviewer"] = "instructor"
    case = review["cases"][0]
    case.update(
        reviewed=True,
        status_correct=True,
        answer_relevant=True,
        answer_complete=True,
        example_context_preserved=True,
    )
    case["fact_checks"][0]["covered"] = True
    case["statement_checks"][0].update(supported=True, citations_support_statement=True)
    case["forbidden_claim_checks"][0]["present"] = False
    return review


def test_annotations_do_not_auto_approve_a_course_or_invent_fact_meanings():
    original = {"id": "a", "expected_status": "answered", "required_quotes": ["متن"]}
    draft = annotation_template([original])[0]
    assert draft["review_status"] == "proposed"
    assert draft["required_facts"][0]["text"] == ""
    assert validate_annotations(draft)
    assert not validate_annotations(draft | {"review_status": "approved", "reviewer": "teacher"})
    assert validate_annotations(
        draft
        | {
            "review_status": "approved",
            "reviewer": "teacher",
            "required_facts": [{"id": "one", "text": "مفهوم", "evidence_quotes": ["متن"]}],
        }
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"review_status": []},
        {"split": {}},
        {"required_facts": "bad"},
        {"forbidden_claims": [False]},
    ],
)
def test_invalid_annotation_types_are_rejected(extra):
    assert not validate_annotations({"expected_status": "answered"} | extra)


def test_supported_complete_answer_is_graded_by_human_judgments(tmp_path):
    path = write_report(tmp_path)
    grade = score_review(path, complete_review(path))
    assert grade["review_coverage"]["rate"] == 1
    assert grade["required_fact_coverage"]["rate"] == 1
    assert grade["statement_support"]["rate"] == 1
    assert grade["citation_support"]["rate"] == 1
    assert grade["semantic_pass"]["rate"] == 1


def test_unsupported_claim_missing_fact_and_bad_citation_fail_independently(tmp_path):
    path = write_report(tmp_path)
    review = complete_review(path)
    case = review["cases"][0]
    case["fact_checks"][0]["covered"] = False
    case["statement_checks"][0].update(supported=False, citations_support_statement=False)
    case["forbidden_claim_checks"][0]["present"] = True
    grade = score_review(path, review)
    assert grade["required_fact_coverage"]["rate"] == 0
    assert grade["answers_with_unsupported_statements"]["rate"] == 1
    assert grade["semantic_pass"]["rate"] == 0


def test_partial_review_has_no_implied_passes(tmp_path):
    path = write_report(tmp_path)
    review = review_template(path)
    review["reviewer"] = "instructor"
    grade = score_review(path, review)
    assert grade["review_coverage"]["rate"] == 0
    assert grade["semantic_pass"]["rate"] is None


def test_false_refusal_counts_as_missing_required_facts(tmp_path):
    path = write_report(tmp_path)
    payload = report()
    payload["outcomes"][0]["actual_status"] = "refused"
    payload["outcomes"][0]["response"] = {
        "status": "refused",
        "answer": "refusal",
        "excerpts": [],
        "explanation": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    review = review_template(path)
    review["reviewer"] = "instructor"
    review["cases"][0].update(reviewed=True, status_correct=False)
    result = score_review(path, review)
    assert result["required_fact_coverage"]["denominator"] == 1
    assert result["required_fact_coverage"]["rate"] == 0
    assert result["semantic_pass"]["rate"] == 0


@pytest.mark.parametrize("change", ["response", "fact", "missing_judgment", "remove_case"])
def test_review_cannot_change_the_answer_or_omit_required_judgments(tmp_path, change):
    path = write_report(tmp_path)
    review = complete_review(path)
    case = review["cases"][0]
    if change == "response":
        case["response"]["answer"] = "edited"
    elif change == "fact":
        case["fact_checks"][0]["id"] = "another-fact"
    elif change == "missing_judgment":
        case["answer_complete"] = None
    else:
        review["cases"] = []
    with pytest.raises(ValueError):
        score_review(path, review)


def test_stale_review_cannot_grade_another_model_run(tmp_path):
    path = write_report(tmp_path)
    review = complete_review(path)
    changed = report()
    changed["outcomes"][0]["response"]["answer"] = "different"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="exact report"):
        score_review(path, review)


def test_three_retrieval_modes_compare_paired_cases_without_claiming_a_winner(tmp_path):
    paths = [
        write_report(tmp_path, mode, mode + ".json")
        for mode in ("full_context", "hybrid", "hybrid_rerank")
    ]
    comparison = compare_reports(paths)
    assert len(comparison["experiments"]) == 3
    assert comparison["status_disagreements"] == 0
    assert comparison["winner"] is None


@pytest.mark.parametrize(
    "change", ["dataset", "question", "model", "prompt", "source", "mode", "error"]
)
def test_comparison_rejects_confounded_or_incomplete_runs(tmp_path, change):
    first = write_report(tmp_path)
    second = write_report(tmp_path, "hybrid", "other.json")
    payload = json.loads(second.read_text(encoding="utf-8"))
    item = payload["outcomes"][0]
    if change == "dataset":
        payload["dataset_sha256"] = "different"
    elif change == "question":
        item["question"] = "different question"
    elif change == "model":
        item["evaluation"]["configuration"]["chat_model"] = "other-model"
    elif change == "prompt":
        item["evaluation"]["configuration"]["prompt_sha256"] = "new-prompt"
    elif change == "source":
        item["evaluation"]["source_corpus_sha256"] = "other-source"
    elif change == "mode":
        payload["metadata"]["response_mode"] = "verbatim"
    else:
        item["evaluation"] = None
    second.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        compare_reports([first, second])


def test_token_cost_requires_actual_usage_and_explicit_prices():
    item = deepcopy(report()["outcomes"][0])
    trace = item["evaluation"]
    trace["provider_calls"] = [
        {
            "kind": "chat/completions",
            "model": "chat",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
    ]
    pricing = {"chat": {"input_per_million": 2.0, "output_per_million": 5.0}}
    assert runtime_summary([item])["estimated_cost_usd"] is None
    assert runtime_summary([item], pricing)["estimated_cost_usd"] == 0.0003
    trace["provider_calls"].append({"kind": "chat/completions", "model": "chat", "usage": None})
    partial = runtime_summary([item], pricing)
    assert partial["estimated_cost_usd"] is None
    assert partial["priced_subtotal_usd"] == 0.0003


def test_diagnostics_exclude_secrets_and_reset_between_requests():
    settings = Settings(
        _env_file=None,
        jwt_secret="private-jwt-key-at-least-32-characters",
        api_key="private-provider-key",
    )
    with collect(settings, True) as first:
        with stage("selection"):
            provider_usage(
                {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}, "chat", "chat/completions"
            )
    serialized = json.dumps(first)
    assert "private-jwt-key" not in serialized
    assert "private-provider-key" not in serialized
    assert first["provider_calls"][0]["stage"] == "selection"
    provider_usage({"usage": {"prompt_tokens": 99}}, "chat", "chat/completions")
    with collect(settings, True) as second:
        assert second["provider_calls"] == []
    assert len(first["provider_calls"]) == 1
