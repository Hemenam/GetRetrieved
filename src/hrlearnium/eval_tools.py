"""Instructor annotations and paired comparisons for course evaluation reports.

Human judgments are kept separate from automatic provenance/format checks.
Review files are bound to the exact report bytes so an old review cannot grade a new run.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

GOLD_FIELDS = {
    "required_facts",
    "forbidden_claims",
    "review_status",
    "reviewer",
    "split",
    "scope_notes",
}


def ratio(hits: int, total: int) -> dict:
    return {"numerator": hits, "denominator": total, "rate": hits / total if total else None}


def validate_annotations(case: dict) -> bool:
    facts = case.get("required_facts", [])
    approved = case.get("review_status", "proposed") == "approved"
    if (
        not isinstance(case.get("review_status", "proposed"), str)
        or case.get("review_status", "proposed") not in {"proposed", "approved"}
        or not isinstance(case.get("split", "development"), str)
        or case.get("split", "development") not in {"development", "held_out"}
        or not isinstance(case.get("reviewer", ""), str)
        or not isinstance(case.get("scope_notes", ""), str)
        or not isinstance(facts, list)
        or not isinstance(case.get("forbidden_claims", []), list)
        or any(
            not isinstance(text, str) or not text.strip()
            for text in case.get("forbidden_claims", [])
        )
    ):
        return False
    ids = set()
    for fact in facts:
        if (
            not isinstance(fact, dict)
            or set(fact) != {"id", "text", "evidence_quotes"}
            or not isinstance(fact["id"], str)
            or not fact["id"].strip()
            or fact["id"] in ids
            or not isinstance(fact["text"], str)
            or (approved and not fact["text"].strip())
            or not isinstance(fact["evidence_quotes"], list)
            or not fact["evidence_quotes"]
            or any(not isinstance(q, str) or not q for q in fact["evidence_quotes"])
        ):
            return False
        ids.add(fact["id"])
    return not approved or (
        bool(case.get("reviewer", "").strip())
        and (case["expected_status"] != "answered" or bool(facts))
    )


def annotation_template(cases: list[dict]) -> list[dict]:
    return [
        case
        | {
            "required_facts": case.get(
                "required_facts",
                [
                    {"id": f"fact-{i}", "text": "", "evidence_quotes": [quote]}
                    for i, quote in enumerate(case["required_quotes"], 1)
                ],
            ),
            "forbidden_claims": case.get("forbidden_claims", []),
            "review_status": "proposed",
            "reviewer": "",
            "split": case.get("split", "development"),
            "scope_notes": case.get("scope_notes", ""),
        }
        for case in cases
    ]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or not isinstance(report.get("outcomes"), list)
        or not isinstance(report.get("metadata"), dict)
        or report["metadata"].get("evaluation_kind") != "api"
        or not isinstance(report.get("dataset_sha256"), str)
        or not isinstance(report.get("summary"), dict)
        or not isinstance(report.get("latency_ms"), dict)
    ):
        raise ValueError("An API evaluation report is required")
    return report


def review_template(report_path: Path) -> dict:
    report = read_report(report_path)
    cases = []
    for item in report["outcomes"]:
        if item["evaluation_status"] != "evaluated":
            continue
        response = item.get("response")
        if not isinstance(response, dict):
            raise ValueError("Rerun evaluation to capture responses before preparing a review")
        statements = (response.get("explanation") or {}).get("statements", [])
        # Verbatim answers also need relevance/meaning checks despite exact source provenance.
        if response["status"] == "answered" and not statements:
            statements = [
                {"text": e["text"], "citation_ids": [e["id"]]} for e in response["excerpts"]
            ]
        cases.append(
            {
                "id": item["id"],
                "question": item["question"],
                "previous_question": item.get("previous_question"),
                "setup_response": item.get("setup_response"),
                "gold": item.get("gold", {}),
                "response": response,
                "reviewed": False,
                "status_correct": None,
                "answer_relevant": None,
                "answer_complete": None,
                "example_context_preserved": None,
                "fact_checks": [
                    {"id": fact["id"], "covered": None}
                    for fact in item.get("gold", {}).get("required_facts", [])
                ],
                "statement_checks": [
                    statement | {"supported": None, "citations_support_statement": None}
                    for statement in statements
                ],
                "forbidden_claim_checks": [
                    {"text": text, "present": None}
                    for text in item.get("gold", {}).get("forbidden_claims", [])
                ],
                "notes": "",
            }
        )
    return {
        "format_version": 1,
        "report_sha256": digest(report_path),
        "dataset_sha256": report["dataset_sha256"],
        "reviewer": "",
        "instructions": (
            "Review against the original course and each statement's cited excerpts. "
            "Fill boolean judgments and mark reviewed=true. A blank judgment is not a pass. "
            "Required facts must describe meaning, rather than merely repeat quote anchors. "
            "For a non-answer, only status_correct is required. Do not edit captured responses."
        ),
        "cases": cases,
    }


def score_review(report_path: Path, review: dict) -> dict:
    expected = review_template(report_path)
    if (
        not isinstance(review, dict)
        or review.get("report_sha256") != expected["report_sha256"]
        or review.get("dataset_sha256") != expected["dataset_sha256"]
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not isinstance(review.get("cases"), list)
    ):
        raise ValueError("Review needs a reviewer and must match this exact report")
    originals = {item["id"]: item for item in expected["cases"]}
    seen, reviewed, decisions = set(), [], []
    for case in review["cases"]:
        if not isinstance(case, dict) or case.get("id") not in originals or case["id"] in seen:
            raise ValueError("Review contains an unknown or repeated case")
        seen.add(case["id"])
        original = originals[case["id"]]
        for field in ("question", "previous_question", "setup_response", "gold", "response"):
            if case.get(field) != original[field]:
                raise ValueError("Captured question, gold annotations or response was changed")
        if type(case.get("reviewed")) is not bool:
            raise ValueError("reviewed must be true or false")
        if not case["reviewed"]:
            continue
        answered = case["response"]["status"] == "answered"
        judgments = ["status_correct"]
        if answered:
            judgments += ["answer_relevant", "answer_complete", "example_context_preserved"]
        if any(type(case.get(key)) is not bool for key in judgments):
            raise ValueError("A reviewed case has missing boolean judgments")
        for field, keys, boolean_keys in (
            ("fact_checks", ("id",), ("covered",)),
            (
                "statement_checks",
                ("text", "citation_ids"),
                ("supported", "citations_support_statement"),
            ),
            ("forbidden_claim_checks", ("text",), ("present",)),
        ):
            checks = case.get(field)
            if not isinstance(checks, list) or len(checks) != len(original[field]):
                raise ValueError("Review check counts must match the captured template")
            for check, baseline in zip(checks, original[field], strict=True):
                if not isinstance(check, dict) or any(
                    check.get(key) != baseline[key] for key in keys
                ):
                    raise ValueError("A captured fact or statement was changed")
                if answered and any(type(check.get(key)) is not bool for key in boolean_keys):
                    raise ValueError("A reviewed answer has missing fact or statement judgments")
        passed = all(case[key] for key in judgments)
        if answered:
            passed = passed and (
                all(f["covered"] for f in case["fact_checks"])
                and all(
                    s["supported"] and s["citations_support_statement"]
                    for s in case["statement_checks"]
                )
                and not any(f["present"] for f in case["forbidden_claim_checks"])
            )
        reviewed.append(case)
        decisions.append({"id": case["id"], "semantic_pass": passed})
    if seen != set(originals):
        raise ValueError("Keep all cases in the review; leave unfinished cases reviewed=false")
    answers = [c for c in reviewed if c["response"]["status"] == "answered"]
    facts = [f for c in reviewed for f in c["fact_checks"]]
    statements = [s for c in answers for s in c["statement_checks"]]
    return {
        "report_sha256": expected["report_sha256"],
        "reviewer": review["reviewer"],
        "review_coverage": ratio(len(reviewed), len(originals)),
        "gold_approved_cases_reviewed": sum(
            c["gold"].get("review_status") == "approved" for c in reviewed
        ),
        "status_correctness": ratio(sum(c["status_correct"] for c in reviewed), len(reviewed)),
        "answer_relevance": ratio(sum(c["answer_relevant"] for c in answers), len(answers)),
        "answer_completeness": ratio(sum(c["answer_complete"] for c in answers), len(answers)),
        "required_fact_coverage": ratio(sum(f["covered"] is True for f in facts), len(facts)),
        "statement_support": ratio(sum(s["supported"] for s in statements), len(statements)),
        "citation_support": ratio(
            sum(s["citations_support_statement"] for s in statements), len(statements)
        ),
        "answers_with_unsupported_statements": ratio(
            sum(any(not s["supported"] for s in c["statement_checks"]) for c in answers),
            len(answers),
        ),
        "semantic_pass": ratio(sum(c["semantic_pass"] for c in decisions), len(reviewed)),
        "decisions": decisions,
        "limitations": [
            "These are human judgments; automatic source and citation checks remain separate.",
            "Statement support requires every claim in a statement to be supported.",
            "Pending reviews and unapproved gold cases do not establish release readiness.",
        ],
    }


def latency_summary(values: list[float]) -> dict:
    values = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
    }


def runtime_summary(outcomes: list[dict], pricing: dict | None = None) -> dict:
    traces = []
    for item in outcomes:
        for key in ("setup_evaluation", "evaluation"):
            if isinstance(item.get(key), dict):
                traces.append(item[key])
    calls = [call for trace in traces for call in trace.get("provider_calls", [])]
    diagnostics_complete = bool(outcomes) and all(
        item.get("evaluation_status") == "evaluated"
        and isinstance(item.get("evaluation"), dict)
        and (
            not item.get("has_previous_question") or isinstance(item.get("setup_evaluation"), dict)
        )
        for item in outcomes
    )
    observed = [call for call in calls if isinstance(call.get("usage"), dict)]
    priced, subtotal = 0, 0.0
    for call in calls:
        usage, price = call.get("usage"), (pricing or {}).get(call["model"])
        if not usage or not price or "prompt_tokens" not in usage:
            continue
        output = (
            usage.get("completion_tokens", 0)
            if call["kind"] == "embeddings"
            else usage.get("completion_tokens")
        )
        if output is None:
            continue
        subtotal += (
            usage["prompt_tokens"] * price["input_per_million"]
            + output * price["output_per_million"]
        ) / 1_000_000
        priced += 1
    return {
        "cases_with_diagnostics": sum(isinstance(i.get("evaluation"), dict) for i in outcomes),
        "stage_latency_ms": {
            name: latency_summary(
                [t["stages_ms"][name] for t in traces if name in t.get("stages_ms", {})]
            )
            for name in sorted({name for t in traces for name in t.get("stages_ms", {})})
        },
        "observed_provider_calls": len(calls),
        "calls_with_token_usage": len(observed),
        "reported_token_totals": {
            key: sum(c["usage"][key] for c in observed if key in c["usage"])
            if any(key in c["usage"] for c in observed)
            else None
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "pricing_currency": "USD",
        "priced_calls": priced,
        "estimated_cost_usd": (
            round(subtotal, 8) if diagnostics_complete and calls and priced == len(calls) else None
        ),
        "cost_diagnostics_complete": diagnostics_complete,
        "priced_subtotal_usd": round(subtotal, 8) if priced else None,
        "cost_scope": "Reported query/setup provider calls only; excludes indexing, hosting and local reranker compute.",
        "candidate_evidence_coverage": {
            name: ratio(
                sum(
                    item["candidate_coverage"][name]["required_coverage_complete"]
                    for item in outcomes
                    if item["expected_status"] == "answered"
                    and name in item.get("candidate_coverage", {})
                ),
                sum(
                    item["expected_status"] == "answered"
                    and name in item.get("candidate_coverage", {})
                    for item in outcomes
                ),
            )
            for name in ("retrieved_candidates", "selector_candidates")
        },
    }


def load_pricing(path: Path | None) -> dict | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("Pricing must map model IDs to input/output USD per million tokens")
    for model, prices in value.items():
        if (
            not isinstance(model, str)
            or not isinstance(prices, dict)
            or set(prices) != {"input_per_million", "output_per_million"}
        ):
            raise ValueError("Invalid pricing schema")
        if any(
            type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in prices.values()
        ):
            raise ValueError("Prices must be finite nonnegative numbers")
    return value


def compare_reports(paths: list[Path]) -> dict:
    reports = [read_report(path) for path in paths]
    if len(reports) < 2:
        raise ValueError("Comparison requires at least two reports")
    baseline = reports[0]
    base_ids = [
        (i["id"], i["question"], i["expected_status"], i.get("previous_question"))
        for i in baseline["outcomes"]
    ]
    experiments, paired = [], []
    baseline_config = None
    baseline_source = None
    for path, report in zip(paths, reports, strict=True):
        if (
            report["dataset_sha256"] != baseline["dataset_sha256"]
            or [
                (i["id"], i["question"], i["expected_status"], i.get("previous_question"))
                for i in report["outcomes"]
            ]
            != base_ids
            or report["metadata"].get("response_mode") != baseline["metadata"].get("response_mode")
            or report["metadata"].get("course") != baseline["metadata"].get("course")
        ):
            raise ValueError("Compare the same dataset, cases, course and response mode")
        traces = [
            i["evaluation"] for i in report["outcomes"] if isinstance(i.get("evaluation"), dict)
        ]
        if not traces:
            raise ValueError("Comparison requires captured runtime diagnostics")
        configs = {json.dumps(t["configuration"], sort_keys=True) for t in traces}
        sources = {
            t["source_corpus_sha256"] for t in traces if t.get("source_passage_count", 0) > 0
        }
        if len(configs) != 1 or len(sources) != 1:
            raise ValueError("Each run must use one configuration and one nonempty source corpus")
        config = json.loads(next(iter(configs)))
        # Only retrieval strategy/reranker may vary in this controlled experiment.
        common = {k: v for k, v in config.items() if k not in {"reranker", "rerank_pool_limit"}}
        source = next(iter(sources))
        if baseline_config is None:
            baseline_config, baseline_source = common, source
        elif common != baseline_config or source != baseline_source:
            raise ValueError(
                "Chat/embedding models, prompts, limits, code and source must match across runs"
            )
        modes = {i["retrieval_mode"] for i in report["outcomes"] if i.get("retrieval_mode")}
        if len(modes) != 1:
            raise ValueError("Each run must report exactly one retrieval mode")
        experiments.append(
            {
                "report": str(path),
                "report_sha256": digest(path),
                "retrieval_mode": next(iter(modes)),
                "summary": report["summary"],
                "runtime": report.get("runtime"),
                "latency_ms": report["latency_ms"],
                "configuration": config,
            }
        )
    for index, (case_id, _, _, _) in enumerate(base_ids):
        items = [report["outcomes"][index] for report in reports]
        paired.append(
            {
                "id": case_id,
                "statuses": [i.get("actual_status") for i in items],
                "automated_passes": [i.get("passed_automated_checks", False) for i in items],
                "required_evidence_coverage": [
                    i.get("required_coverage_complete", False) for i in items
                ],
                "status_disagreement": len({i.get("actual_status") for i in items}) > 1,
            }
        )
    return {
        "dataset_sha256": baseline["dataset_sha256"],
        "source_corpus_sha256": baseline_source,
        "experiments": experiments,
        "paired_cases": paired,
        "status_disagreements": sum(i["status_disagreement"] for i in paired),
        "winner": None,
        "limitations": [
            "Automatic checks alone cannot choose the best semantic answering architecture.",
            "Use instructor reviews of each report, and repeat runs to measure model variability.",
            "A reported model ID may be an alias; provider snapshots are not inferred.",
        ],
    }
