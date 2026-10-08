"""Evaluate lexical coverage or the course API in verbatim/explained mode.

Retrieval mode never invokes a model or network service. API mode requires an
already-ingested course and creates new conversation state for every case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlparse
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrlearnium.eval_tools import (  # noqa: E402
    GOLD_FIELDS,
    latency_summary,
    load_pricing,
    runtime_summary,
    validate_annotations,
)

STATUSES = {"answered", "refused", "clarification", "conversation"}
CATEGORIES = {
    "conversation",
    "direct",
    "paraphrase",
    "typo",
    "multi_section",
    "unsupported",
    "advice",
    "injection",
    "ambiguous",
    "followup",
    "example_boundary",
    "false_premise",
    "contradiction",
    "distractor",
    "position",
    "list_completeness",
}


class EvaluationError(Exception):
    """A safe, non-secret-bearing evaluation configuration or protocol error."""


def fraction(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(numerator / denominator, 6) if denominator else None,
    }


def load_cases(path: Path, limit: int | None) -> list[dict]:
    required = {
        "id",
        "category",
        "question",
        "expected_status",
        "expected_chapters",
        "required_quotes",
        "notes",
    }
    cases, identifiers = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError:
            raise EvaluationError(f"Dataset line {number} is not valid JSON") from None
        valid = (
            isinstance(case, dict)
            and required <= case.keys() <= required | {"previous_question"} | GOLD_FIELDS
            and isinstance(case["id"], str)
            and case["id"]
            and isinstance(case["category"], str)
            and case["category"] in CATEGORIES
            and isinstance(case["expected_status"], str)
            and case["expected_status"] in STATUSES
            and isinstance(case["question"], str)
            and case["question"].strip()
            and isinstance(case["notes"], str)
            and isinstance(case["expected_chapters"], list)
            and all(type(chapter) is int and chapter > 0 for chapter in case["expected_chapters"])
            and isinstance(case["required_quotes"], list)
            and all(isinstance(anchor, str) and anchor for anchor in case["required_quotes"])
            and (
                "previous_question" not in case
                or (
                    isinstance(case["previous_question"], str) and case["previous_question"].strip()
                )
            )
        )
        if not valid:
            raise EvaluationError(f"Dataset line {number} has an invalid case schema")
        if not validate_annotations(case):
            raise EvaluationError(f"Dataset line {number} has invalid instructor annotations")
        if case["id"] in identifiers:
            raise EvaluationError(f"Dataset line {number} repeats an identifier")
        if (case["expected_status"] == "answered") != bool(case["required_quotes"]):
            raise EvaluationError(f"Dataset line {number} has inconsistent quote expectations")
        identifiers.add(case["id"])
        cases.append(case)
    if not cases:
        raise EvaluationError("Dataset is empty")
    return cases[:limit] if limit is not None else cases


def base_outcome(case: dict) -> dict:
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_status": case["expected_status"],
        "expected_chapters": case["expected_chapters"],
        "required_quote_count": len(case["required_quotes"]),
        "has_previous_question": "previous_question" in case,
        "previous_question": case.get("previous_question"),
        "gold": {
            "required_facts": case.get("required_facts", []),
            "forbidden_claims": case.get("forbidden_claims", []),
            "review_status": case.get("review_status", "proposed"),
            "reviewer": case.get("reviewer", ""),
            "split": case.get("split", "development"),
            "scope_notes": case.get("scope_notes", ""),
            "notes": case["notes"],
        },
    }


def coverage(case: dict, texts: list[str], chapters: list[int | None]) -> dict:
    missing_quotes = [
        anchor for anchor in case["required_quotes"] if not any(anchor in text for text in texts)
    ]
    missing_chapters = sorted(set(case["expected_chapters"]) - set(chapters))
    fact_anchors = [
        quote for fact in case.get("required_facts", []) for quote in fact["evidence_quotes"]
    ]
    missing_fact_anchors = [q for q in fact_anchors if not any(q in text for text in texts)]
    return {
        "matched_quote_count": len(case["required_quotes"]) - len(missing_quotes),
        "missing_quotes": missing_quotes,
        "missing_chapters": missing_chapters,
        "quote_coverage_complete": not missing_quotes,
        "chapter_coverage_complete": not missing_chapters,
        "required_coverage_complete": not missing_quotes and not missing_chapters,
        "gold_evidence_anchor_count": len(fact_anchors),
        "gold_evidence_anchor_hits": len(fact_anchors) - len(missing_fact_anchors),
        "gold_evidence_missing_anchors": missing_fact_anchors,
        "gold_evidence_coverage_complete": not missing_fact_anchors if fact_anchors else None,
    }


def coverage_summary(outcomes: list[dict]) -> dict:
    positives = [item for item in outcomes if item["expected_status"] == "answered"]
    chapter_total = sum(len(set(item["expected_chapters"])) for item in positives)
    chapter_hits = sum(
        len(set(item["expected_chapters"]))
        - len(item.get("missing_chapters", item["expected_chapters"]))
        for item in positives
    )
    return {
        "answerable_cases": len(positives),
        "chapter_recall": fraction(chapter_hits, chapter_total),
        "quote_anchor_coverage": fraction(
            sum(item.get("matched_quote_count", 0) for item in positives),
            sum(item["required_quote_count"] for item in positives),
        ),
        "complete_chapter_coverage": fraction(
            sum(item.get("chapter_coverage_complete", False) for item in positives),
            len(positives),
        ),
        "complete_quote_coverage": fraction(
            sum(item.get("quote_coverage_complete", False) for item in positives),
            len(positives),
        ),
        "complete_required_coverage": fraction(
            sum(item.get("required_coverage_complete", False) for item in positives),
            len(positives),
        ),
        "gold_evidence_anchor_coverage": fraction(
            sum(item.get("gold_evidence_anchor_hits", 0) for item in positives),
            sum(item.get("gold_evidence_anchor_count", 0) for item in positives),
        ),
    }


def retrieval_evaluation(cases: list[dict], document_path: Path) -> tuple[list[dict], dict]:
    from hrlearnium.ingestion import parse_docx
    from hrlearnium.policy import is_followup
    from hrlearnium.retrieval import SearchPassage, retrieve

    data = document_path.read_bytes()
    try:
        document = parse_docx(data, document_path.name)
    except ValueError:
        raise EvaluationError("Document could not be safely ingested") from None
    passages = [
        SearchPassage(
            id=f"p{index:03d}",
            text=passage.text,
            chapter_number=passage.chapter_number,
            chapter_title=passage.chapter_title,
            section_kind=passage.section_kind,
        )
        for index, passage in enumerate(document.passages)
    ]
    outcomes = []
    for case in cases:
        outcome = base_outcome(case)
        if case["expected_status"] != "answered":
            outcome.update(
                {
                    "evaluation_status": "not_evaluated",
                    "reason": "Candidate retrieval does not determine answerability or refusal.",
                }
            )
            outcomes.append(outcome)
            continue
        # Use the service's same retrieval enrichment rule; the offline run has no selector.
        question = case["question"]
        if case.get("previous_question") and is_followup(question):
            question = case["previous_question"] + "\n" + question
        start = time.perf_counter()
        candidates = retrieve(question, passages, limit=8)
        outcome.update(
            {
                "evaluation_status": "retrieval_only",
                "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                "candidate_ids": [item.passage.id for item in candidates],
                "candidate_chapters": [item.passage.chapter_number for item in candidates],
                "candidate_lexical_scores": [round(item.lexical_score, 6) for item in candidates],
            }
        )
        outcome.update(
            coverage(
                case,
                [item.passage.text for item in candidates],
                [item.passage.chapter_number for item in candidates],
            )
        )
        outcome["source_anchors_valid"] = all(
            anchor in document.full_text for anchor in case["required_quotes"]
        )
        outcomes.append(outcome)
    return outcomes, {
        "evaluation_kind": "retrieval_only",
        "retrieval": "lexical_bm25",
        "candidate_limit": 8,
        "source_file_sha256": hashlib.sha256(data).hexdigest(),
        "source_text_sha256": hashlib.sha256(document.full_text.encode("utf-8")).hexdigest(),
        "passage_count": len(passages),
        "paragraph_count": document.paragraph_count,
        "ingestion_warnings": document.warnings,
        "followup_context": "Use the service's is_followup rule to append supplied previous_question.",
        "limitations": [
            "Retrieval-only lexical coverage; no semantic model or answer selector was evaluated.",
            "Negative cases are not evaluated. This report measures neither refusal nor relevance correctness.",
            "An authentic retrieved quote can still be an irrelevant, incomplete, or misleading answer.",
        ],
    }


def auth_headers(args: argparse.Namespace):
    if not args.local_auth:
        token = os.environ.get("HR_EVAL_TOKEN", "").strip()
        if not token:
            raise EvaluationError("API mode requires HR_EVAL_TOKEN or explicit --local-auth")
        return lambda: {"Authorization": f"Bearer {token}"}
    if not args.tenant:
        raise EvaluationError("--local-auth requires --tenant")
    if urlparse(args.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise EvaluationError("--local-auth is restricted to a loopback API address")
    import jwt

    from hrlearnium.config import Settings
    from hrlearnium.schemas import Principal

    try:
        settings = Settings()
        principal = Principal(
            sub=args.subject or "evaluation",
            tenant_id=args.tenant,
            course_ids=[args.course],
            scopes=["query"],
        )
    except ValueError:
        # Settings validation can contain sensitive input; never print its exception.
        raise EvaluationError("Invalid local settings or evaluation principal") from None

    def headers():
        now = int(time.time())
        payload = principal.model_dump() | {
            "iat": now,
            "exp": now + min(60, settings.max_token_lifetime_seconds),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "jti": str(uuid4()),
        }
        token = jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm="HS256")
        return {"Authorization": f"Bearer {token}"}

    return headers


def response_contract_checks(result, requested_mode: str) -> dict:
    """Check rendering and links, never whether generated statements are true."""
    from hrlearnium.policy import conversational_reply, non_answer_reply
    from hrlearnium.schemas import render_answer

    answered = result.status == "answered"
    excerpt_ids = [excerpt.id for excerpt in result.excerpts]
    explanation = result.explanation
    links_valid = (
        all(
            bool(statement.citation_ids)
            and len(statement.citation_ids) == len(set(statement.citation_ids))
            and all(
                identifier.strip() and identifier in excerpt_ids
                for identifier in statement.citation_ids
            )
            for statement in explanation.statements
        )
        if explanation is not None
        else None
    )
    explanation_contract_valid = (
        explanation is not None
        if answered and requested_mode == "explained"
        else explanation is None
    )
    canonical = (
        result.answer == render_answer(result.excerpts, explanation)
        if answered and links_valid is not False
        else False
        if answered
        else None
    )
    if result.status == "conversation":
        allowed_controls = {
            reply[1]
            for question in ("سلام", "hi", "ممنون", "thanks", "کمک", "help")
            if (reply := conversational_reply(question)) and reply[0] == result.reason_code
        }
    else:
        allowed_controls = {
            non_answer_reply(question, result.status, result.reason_code)
            for question in ("سلام", "hello")
        }
    return {
        "requested_response_mode": requested_mode,
        "actual_response_mode": result.response_mode,
        "response_mode_matches": result.response_mode == requested_mode,
        "explanation_contract_valid": explanation_contract_valid,
        "explanation_citation_links_valid": links_valid,
        "explanation_statement_count": len(explanation.statements) if explanation else 0,
        "generated_explanation_human_review_required": explanation is not None,
        "answer_canonical_render": canonical,
        "answer_exactly_joined_excerpts": (
            result.answer == "\n\n".join(excerpt.text for excerpt in result.excerpts)
            if answered and requested_mode == "verbatim"
            else None
        ),
        "control_message_valid": (
            result.answer in allowed_controls and not result.excerpts and explanation is None
            if not answered
            else None
        ),
        "unique_excerpt_ids": len(set(excerpt_ids)) == len(excerpt_ids),
        "answered_has_excerpts": bool(result.excerpts) if answered else None,
    }


def api_evaluation(cases: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict]:
    import httpx
    from pydantic import ValidationError

    from hrlearnium.schemas import Excerpt, QueryResponse

    headers = auth_headers(args)
    local_source, local_hash = None, None
    if args.document:
        from hrlearnium.ingestion import parse_docx

        try:
            local_source = parse_docx(args.document.read_bytes(), args.document.name).full_text
        except ValueError:
            raise EvaluationError("Document could not be safely ingested") from None
        local_hash = hashlib.sha256(local_source.encode("utf-8")).hexdigest()
    query_path = f"/v1/courses/{quote(args.course, safe='')}/query"
    outcomes = []
    seen_conversations = set()
    candidate_cache = {}
    with httpx.Client(
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
        trust_env=False,
        follow_redirects=False,
    ) as client:

        def request(method: str, path: str, schema, **kwargs):
            response = client.request(method, path, headers=headers(), **kwargs)
            response.raise_for_status()
            return schema.model_validate(response.json())

        for index, case in enumerate(cases):
            if index and args.delay_seconds:
                time.sleep(args.delay_seconds)
            outcome = base_outcome(case)
            outcome.update({"evaluation_status": "error", "actual_status": None})
            started = time.perf_counter()
            stage = "setup" if case.get("previous_question") else "query"
            try:
                body = {
                    "question": case["question"],
                    "response_mode": args.response_mode,
                    "include_evaluation": True,
                }
                if case.get("previous_question"):
                    setup = request(
                        "POST",
                        query_path,
                        QueryResponse,
                        json={
                            "question": case["previous_question"],
                            "response_mode": args.response_mode,
                            "include_evaluation": True,
                        },
                    )
                    outcome["setup_status"] = setup.status
                    outcome["setup_evaluation"] = setup.evaluation
                    outcome["setup_response"] = setup.model_dump()
                    setup_checks = response_contract_checks(setup, args.response_mode)
                    outcome["setup_contract_checks"] = setup_checks
                    outcome["setup_failed"] = not (
                        setup.status == "answered"
                        and setup_checks["response_mode_matches"]
                        and setup_checks["explanation_contract_valid"]
                        and setup_checks["explanation_citation_links_valid"] is not False
                        and setup_checks["answer_canonical_render"]
                        and setup_checks["unique_excerpt_ids"]
                        and setup_checks["answered_has_excerpts"]
                    )
                    if setup.conversation_id in seen_conversations:
                        raise EvaluationError("Fresh case reused a previous case conversation")
                    seen_conversations.add(setup.conversation_id)
                    body["conversation_id"] = setup.conversation_id
                stage = "query"
                query_start = time.perf_counter()
                result = request("POST", query_path, QueryResponse, json=body)
                outcome["query_latency_ms"] = round((time.perf_counter() - query_start) * 1000, 3)
                if "conversation_id" in body:
                    if result.conversation_id != body["conversation_id"]:
                        raise EvaluationError(
                            "Follow-up response changed its conversation identifier"
                        )
                elif result.conversation_id in seen_conversations:
                    raise EvaluationError("Fresh case reused a previous case conversation")
                seen_conversations.add(result.conversation_id)
                outcome.update(
                    {
                        "actual_status": result.status,
                        "status_matches": result.status == case["expected_status"],
                        "retrieval_mode": result.retrieval_mode,
                        "policy_version": result.policy_version,
                        "reason_code": result.reason_code,
                        "evaluation": result.evaluation,
                        "response": result.model_dump(),
                        "excerpt_ids": [excerpt.id for excerpt in result.excerpts],
                        "returned_chapters": [
                            excerpt.citation.chapter_number for excerpt in result.excerpts
                        ],
                    }
                )
                expected_mode = getattr(args, "expect_retrieval_mode", None)
                if expected_mode and result.retrieval_mode != expected_mode:
                    raise EvaluationError(
                        "Server retrieval mode does not match --expect-retrieval-mode"
                    )
                outcome.update(response_contract_checks(result, args.response_mode))
                outcome.update(
                    coverage(
                        case,
                        [excerpt.text for excerpt in result.excerpts],
                        [excerpt.citation.chapter_number for excerpt in result.excerpts],
                    )
                )
                stage = "citation_lookup"
                citation_checks = []
                for excerpt in result.excerpts:
                    path = f"/v1/courses/{quote(args.course, safe='')}/excerpts/{quote(excerpt.id, safe='')}"
                    stored = request("GET", path, Excerpt)
                    candidate_cache[excerpt.id] = stored
                    check = {
                        "id": excerpt.id,
                        "stored_excerpt_matches": stored.model_dump() == excerpt.model_dump(),
                    }
                    if local_source is not None:
                        citation = excerpt.citation
                        check["local_source_span_matches"] = (
                            0 <= citation.source_start < citation.source_end <= len(local_source)
                            and local_source[citation.source_start : citation.source_end]
                            == excerpt.text
                            and citation.source_sha256 == local_hash
                        )
                    citation_checks.append(check)
                outcome["citation_checks"] = citation_checks
                if result.evaluation is not None:
                    stage = "candidate_lookup"
                    candidate_coverage = {}
                    for name in ("retrieved_candidates", "selector_candidates"):
                        if name not in result.evaluation:
                            continue
                        selected_candidates = []
                        for candidate in result.evaluation[name]:
                            identifier = candidate["id"]
                            if identifier not in candidate_cache:
                                path = f"/v1/courses/{quote(args.course, safe='')}/excerpts/{quote(identifier, safe='')}"
                                candidate_cache[identifier] = request("GET", path, Excerpt)
                            selected_candidates.append(candidate_cache[identifier])
                        candidate_coverage[name] = coverage(
                            case,
                            [e.text for e in selected_candidates],
                            [e.citation.chapter_number for e in selected_candidates],
                        ) | {"candidate_count": len(selected_candidates)}
                    outcome["candidate_coverage"] = candidate_coverage
                outcome["citation_resolution_exact"] = all(
                    item["stored_excerpt_matches"] for item in citation_checks
                )
                source_matches = all(
                    item.get("local_source_span_matches", True) for item in citation_checks
                )
                output_valid = (
                    outcome["answer_canonical_render"] and outcome["answered_has_excerpts"]
                    if result.status == "answered"
                    else outcome["control_message_valid"]
                )
                outcome["passed_automated_checks"] = bool(
                    outcome["status_matches"]
                    and outcome["required_coverage_complete"]
                    and outcome["gold_evidence_coverage_complete"] is not False
                    and outcome["unique_excerpt_ids"]
                    and outcome["citation_resolution_exact"]
                    and outcome["response_mode_matches"]
                    and outcome["explanation_contract_valid"]
                    and outcome["explanation_citation_links_valid"] is not False
                    and source_matches
                    and output_valid
                    and not outcome.get("setup_failed", False)
                )
                outcome["evaluation_status"] = "evaluated"
                outcome["manual_relevance_review_required"] = result.status == "answered"
            except httpx.HTTPStatusError as error:
                outcome["error"] = {
                    "kind": "http_status",
                    "status_code": error.response.status_code,
                    "stage": stage,
                }
            except httpx.TimeoutException:
                outcome["error"] = {"kind": "timeout", "stage": stage}
            except httpx.HTTPError:
                outcome["error"] = {"kind": "transport", "stage": stage}
            except (ValidationError, ValueError):
                outcome["error"] = {"kind": "invalid_response", "stage": stage}
            except EvaluationError as error:
                outcome["error"] = {"kind": "protocol", "stage": stage, "detail": str(error)}
            outcome["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            outcomes.append(outcome)
    return outcomes, {
        "evaluation_kind": "api",
        "base_url": args.base_url.rstrip("/"),
        "course": args.course,
        "response_mode": args.response_mode,
        "tenant_label": args.tenant,
        "subject_label": args.subject,
        "authentication": "local_short_lived_per_request" if args.local_auth else "HR_EVAL_TOKEN",
        "source_text_sha256": local_hash,
        "local_source_verification": local_source is not None,
        "limitations": [
            "Automated quote and citation checks do not establish answer relevance or example context.",
            "Explained-mode rendering and citation links are structural checks, not semantic verification.",
            "Generated explanations require human review against their cited excerpts.",
            "Citation GET checks compare against separately resolved authorized stored excerpts.",
            "A local DOCX verifies source spans only when --document is supplied.",
            "Instructor review and separate backend authorization tests remain required.",
            "Token expiry, HTTP 503/429, timeouts, and malformed responses count as errors, not refusals.",
        ],
    }


def api_summary(outcomes: list[dict]) -> dict:
    positives = [item for item in outcomes if item["expected_status"] == "answered"]
    negatives = [item for item in outcomes if item["expected_status"] != "answered"]
    evaluated = [item for item in outcomes if item["evaluation_status"] == "evaluated"]
    answered = [item for item in outcomes if item.get("actual_status") == "answered"]
    verbatim_answered = [
        item for item in answered if item.get("requested_response_mode") == "verbatim"
    ]
    explained_answered = [
        item for item in answered if item.get("requested_response_mode") == "explained"
    ]
    statuses = sorted(STATUSES)
    confusion = {
        expected: dict(
            Counter(
                item.get("actual_status") or "error"
                for item in outcomes
                if item["expected_status"] == expected
            )
        )
        for expected in statuses
    }
    return coverage_summary(outcomes) | {
        "cases": len(outcomes),
        "completed_cases": len(evaluated),
        "operational_errors": len(outcomes) - len(evaluated),
        "setup_failures": sum(item.get("setup_failed", False) for item in outcomes),
        "status_confusion_matrix": confusion,
        "status_accuracy": fraction(
            sum(item.get("status_matches", False) for item in outcomes), len(outcomes)
        ),
        "response_mode_accuracy": fraction(
            sum(item.get("response_mode_matches", False) for item in outcomes), len(outcomes)
        ),
        "passed_automated_checks": fraction(
            sum(item.get("passed_automated_checks", False) for item in outcomes), len(outcomes)
        ),
        "false_answer_rate_on_negative_cases": fraction(
            sum(item.get("actual_status") == "answered" for item in negatives),
            len(negatives),
        ),
        "unnecessary_refusal_rate_on_answerable_cases": fraction(
            sum(item.get("actual_status") == "refused" for item in positives),
            len(positives),
        ),
        "unnecessary_clarification_rate_on_answerable_cases": fraction(
            sum(item.get("actual_status") == "clarification" for item in positives),
            len(positives),
        ),
        "unnecessary_conversation_rate_on_answerable_cases": fraction(
            sum(item.get("actual_status") == "conversation" for item in positives),
            len(positives),
        ),
        "abstention_rate_on_answerable_cases": fraction(
            sum(
                item.get("actual_status") in {"refused", "clarification", "conversation"}
                for item in positives
            ),
            len(positives),
        ),
        "exact_answer_assembly": fraction(
            sum(item.get("answer_exactly_joined_excerpts") is True for item in verbatim_answered),
            len(verbatim_answered),
        ),
        "explained_canonical_render": fraction(
            sum(item.get("answer_canonical_render") is True for item in explained_answered),
            len(explained_answered),
        ),
        "explained_citation_links_valid": fraction(
            sum(
                item.get("explanation_citation_links_valid") is True for item in explained_answered
            ),
            len(explained_answered),
        ),
        "citation_resolution_exact": fraction(
            sum(item.get("citation_resolution_exact", False) for item in answered),
            len(answered),
        ),
        "generated_explanations_requiring_human_review": sum(
            item.get("generated_explanation_human_review_required", False) for item in outcomes
        ),
        "explanation_semantic_correctness": "not_evaluated_by_automated_checks",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrieval", "api"], required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/course_qa.jsonl")
    parser.add_argument(
        "--document",
        type=Path,
        help="Required for retrieval; optional source-span verification for API",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--response-mode",
        choices=["verbatim", "explained"],
        default="verbatim",
        help="API response mode for both setup and main turns; retrieval-only mode is unaffected",
    )
    parser.add_argument(
        "--tenant", help="Report label; required principal tenant with --local-auth"
    )
    parser.add_argument(
        "--course", help="Already-ingested authorized course; required for API mode"
    )
    parser.add_argument("--subject", help="Report label; local-auth subject defaults to evaluation")
    parser.add_argument(
        "--local-auth",
        action="store_true",
        help="Explicitly mint short-lived local tokens from HR_JWT_SECRET/.env",
    )
    parser.add_argument(
        "--output", type=Path, help="Write JSON report to this path; otherwise print full report"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=["development", "held_out"])
    parser.add_argument("--approved-only", action="store_true", help="Require reviewed gold cases")
    parser.add_argument(
        "--expect-retrieval-mode",
        choices=["full_context", "hybrid", "hybrid_rerank", "lexical"],
        help="Fail a case if the running server uses another retrieval strategy",
    )
    parser.add_argument(
        "--pricing", type=Path, help="Optional model prices in USD per million tokens"
    )
    parser.add_argument("--timeout", type=float, default=180, help="API timeout in seconds")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0,
        help="Optional pause between API cases for rate limits",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be finite and positive")
    if not 0 <= args.delay_seconds <= 60:
        parser.error("--delay-seconds must be between 0 and 60")
    if args.mode == "retrieval" and not args.document:
        parser.error("retrieval mode requires --document")
    if args.mode == "api" and not args.course:
        parser.error("api mode requires --course")
    parsed_url = urlparse(args.base_url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
        or parsed_url.fragment
    ):
        parser.error("--base-url must be HTTP(S), without credentials, query, or fragment")
    try:
        cases = load_cases(args.dataset, None)
        if args.split:
            cases = [case for case in cases if case.get("split", "development") == args.split]
        if args.approved_only:
            cases = [case for case in cases if case.get("review_status") == "approved"]
        if args.limit:
            cases = cases[: args.limit]
        if not cases:
            raise EvaluationError("No cases match the requested split/approval filter")
        try:
            pricing = load_pricing(args.pricing)
        except (ValueError, json.JSONDecodeError):
            raise EvaluationError("Invalid pricing file") from None
        if args.mode == "retrieval":
            outcomes, metadata = retrieval_evaluation(cases, args.document)
            summary_fn = coverage_summary
        else:
            outcomes, metadata = api_evaluation(cases, args)
            summary_fn = api_summary
        latencies = sorted(item["latency_ms"] for item in outcomes if "latency_ms" in item)
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_status": (
                "instructor_approved"
                if all(c.get("review_status") == "approved" for c in cases)
                else "proposed_not_instructor_approved"
            ),
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "dataset_cases_loaded": len(cases),
            "metadata": metadata,
            "runtime": runtime_summary(outcomes, pricing),
            "summary": summary_fn(outcomes),
            "by_category": {
                category: summary_fn([item for item in outcomes if item["category"] == category])
                for category in sorted({item["category"] for item in outcomes})
            },
            "latency_ms": latency_summary(latencies),
            "outcomes": outcomes,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {"output": str(args.output), "mode": args.mode, "summary": report["summary"]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        if any(item["evaluation_status"] == "error" for item in outcomes):
            return 2
        if args.mode == "retrieval":
            return int(any(item.get("required_coverage_complete") is False for item in outcomes))
        return int(any(not item.get("passed_automated_checks", False) for item in outcomes))
    except EvaluationError as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("Evaluation error: could not read input or write report", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
