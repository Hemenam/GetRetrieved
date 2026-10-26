"""Offline, hash-bound summary of natural-student captures; never calls a model."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrlearnium.eval_tools import digest  # noqa: E402


def load_run(directory):
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    report_path = directory / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if index["status"] not in {"complete", "stopped"}:
        raise ValueError("Wait for collection to finish before generating a final report")
    if report["metadata"] != index:
        raise ValueError("Report and checkpoint differ")
    if digest(directory / "inputs/cases.jsonl") != index["dataset_sha256"]:
        raise ValueError("Reference snapshot changed")
    if digest(directory / "inputs/manifest.json") != index["manifest_sha256"]:
        raise ValueError("Manifest snapshot changed")
    if index.get("selection"):
        selection_path = directory / "inputs/selection.json"
        if (
            digest(selection_path) != index["selection_sha256"]
            or json.loads(selection_path.read_text(encoding="utf-8")) != index["selection"]
            or index["selected_case_ids"] != index["selection"]["selected_case_ids"]
        ):
            raise ValueError("Selection snapshot changed")
    for capture in index["captures"]:
        if digest(directory / capture["report"]) != capture["sha256"]:
            raise ValueError("Capture changed: " + capture["report"])
    return index, report, digest(report_path)


def price_equivalent(outcomes, pricing):
    """Undiscounted public list-price equivalent, NOT the configured provider's bill."""
    totals = {}
    missing = 0
    for outcome in outcomes:
        for key in ("setup_evaluation", "evaluation"):
            trace = outcome.get(key) or {}
            for call in trace.get("provider_calls", []):
                model = call["model"]
                value = totals.setdefault(
                    model,
                    {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "public_list_price_equivalent_usd": 0.0,
                    },
                )
                value["calls"] += 1
                usage, rate = call.get("usage"), pricing["models"].get(model)
                if not usage or not rate or "prompt_tokens" not in usage:
                    missing += 1
                    continue
                output = (
                    usage.get("completion_tokens", 0)
                    if call["kind"] == "embeddings"
                    else usage.get("completion_tokens")
                )
                if output is None:
                    missing += 1
                    continue
                value["input_tokens"] += usage["prompt_tokens"]
                value["output_tokens"] += output
                value["public_list_price_equivalent_usd"] += (
                    usage["prompt_tokens"] * rate["input_per_million"]
                    + output * rate["output_per_million"]
                ) / 1_000_000
    for value in totals.values():
        value["public_list_price_equivalent_usd"] = round(
            value["public_list_price_equivalent_usd"], 8
        )
    return {
        "actual_provider_cost_usd": None,
        "pricing_checked_date": pricing["checked_date"],
        "by_model": totals,
        "calls_missing_price_or_usage": missing,
        "captured_calls_undiscounted_equivalent_usd": round(
            sum(v["public_list_price_equivalent_usd"] for v in totals.values()), 8
        ),
        "limitations": pricing["limitations"],
        "failed_request_usage_unknown": any(o.get("error") for o in outcomes),
    }


def diagnostics(outcomes):
    groups = Counter()
    flags = []
    for outcome in outcomes:
        if outcome.get("checks", {}).get("pass"):
            continue
        body = outcome.get("response") or {}
        reason = (
            "operational_error"
            if outcome.get("error")
            else "part_operational_error"
            if any(p["status"] == "error" for p in body.get("parts", []))
            else body.get("reason_code", "no_response")
        )
        groups[reason] += 1
        checks = outcome.get("checks", {})
        failed = [k for k, v in checks.items() if v is False and k != "pass"]
        for i, part in enumerate(checks.get("parts", []), 1):
            failed += [f"part_{i}.{k}" for k, v in part.items() if v is False]
        flags.append(
            {
                "id": outcome["id"],
                "track": outcome["track"],
                "expected_status": outcome["expected_status"],
                "actual_status": body.get("status"),
                "reason": reason,
                "failed_checks": failed,
                "setup_failed_checks": outcome.get("setup_checks", {}).get("pass") is False,
                "part_errors": [
                    p["reason_code"] for p in body.get("parts", []) if p["status"] == "error"
                ],
            }
        )
    return {"flagged_by_response_reason": dict(groups), "flagged_cases": flags}


def candidate_diagnostics(outcomes, passages):
    """Exact-anchor recall for single-answer content cases, not relevance judgments."""
    by_id = {p["id"]: p for p in passages}
    rows = []
    for outcome in outcomes:
        reference = outcome["reference"]
        if len(reference["parts"]) != 1:
            continue
        part = reference["parts"][0]
        anchors = [a for f in part["facts"] for a in f["evidence"]]
        if part["kind"] != "content" or part["status"] != "answered" or not anchors:
            continue
        trace = outcome.get("evaluation") or {}
        candidates = trace.get("selector_candidates")
        if candidates is None:
            child_pools = [
                p["trace"]["selector_candidates"]
                for p in trace.get("parts", [])
                if "selector_candidates" in p.get("trace", {})
            ]
            if child_pools:
                candidates = [candidate for pool in child_pools for candidate in pool]
        present = (
            None
            if candidates is None
            else all(
                any(
                    a["quote"] in by_id.get(c["id"], {}).get("text", "")
                    and by_id.get(c["id"], {}).get("chapter_number") == a["chapter"]
                    and by_id.get(c["id"], {}).get("section_kind") == a["section"]
                    for c in candidates
                )
                for a in anchors
            )
        )
        body = outcome.get("response") or {}
        rows.append(
            {
                "id": outcome["id"],
                "actual_status": body.get("status"),
                "selector_stage_reached": candidates is not None,
                "all_reference_anchors_in_selector": present,
                "verifier_accepted": trace.get("requirement_verification", {}).get("accepted"),
            }
        )
    return {
        "scope": "Single-answer content cases with answered references and exact source anchors only. "
        "Unexpected actual groups are pooled for retrieval diagnostics, not per-part support. "
        "Not a semantic retrieval score; alternative valid evidence is possible.",
        "no_selector_trace": [r["id"] for r in rows if not r["selector_stage_reached"]],
        "refused_despite_all_reference_anchors_in_selector": [
            r["id"]
            for r in rows
            if r["actual_status"] == "refused" and r["all_reference_anchors_in_selector"] is True
        ],
        "refused_with_missing_reference_anchors_in_selector": [
            r["id"]
            for r in rows
            if r["actual_status"] == "refused" and r["all_reference_anchors_in_selector"] is False
        ],
        "cases": rows,
    }


def diagnostic_rates(cases, outcomes):
    by_id = {o["id"]: o for o in outcomes}

    def rate(selected, field):
        return (
            sum(by_id.get(c["id"], {}).get("checks", {}).get(field, False) for c in selected)
            / len(selected)
            if selected
            else None
        )

    tracks = sorted({c["track"] for c in cases})
    families = sorted({c["question_family_id"] for c in cases})
    return {
        "micro_status_match_rate": rate(cases, "status_matches"),
        "micro_automatic_check_rate": rate(cases, "pass"),
        "track_macro_automatic_check_rate": sum(
            rate([c for c in cases if c["track"] == t], "pass") for t in tracks
        )
        / len(tracks),
        "family_macro_automatic_check_rate": sum(
            rate([c for c in cases if c["question_family_id"] == f], "pass") for f in families
        )
        / len(families),
        "limitation": "Diagnostic rates, not semantic accuracy. Correlated development questions; not a sample of production LMS traffic. Unattempted cases do not receive credit.",
    }


def build(directory, pricing_path=None):
    index, report, sha256 = load_run(directory)
    summary, outcomes = report["summary"], report["outcomes"]
    selected = summary["planned_scored_cases"]
    result = {
        "source_report_sha256": sha256,
        "status": index["status"],
        "mode": index["mode"],
        "summary": summary,
        "diagnostics": diagnostics(outcomes),
        "semantic_review_complete": False,
        "semantic_accuracy": None,
    }
    cases = [
        json.loads(line)
        for line in (directory / "inputs/cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if "selected_case_ids" in index:
        ids = index["selected_case_ids"]
        if len(set(ids)) != len(ids) or not set(ids) <= {c["id"] for c in cases}:
            raise ValueError("Invalid selected case IDs")
        cases = [c for c in cases if c["id"] in set(ids)]
        if len(cases) != selected or any(o["id"] not in ids for o in outcomes):
            raise ValueError("Outcomes or planned denominator differ from selection")
    result["diagnostic_rates"] = diagnostic_rates(cases, outcomes)
    notes_path = directory / "execution-notes.json"
    if notes_path.exists():
        result["execution_notes"] = json.loads(notes_path.read_text(encoding="utf-8"))
        result["execution_notes_sha256"] = digest(notes_path)
    spot_review = directory / "spot-review.md"
    if spot_review.exists():
        result["selected_spot_review_sha256"] = digest(spot_review)
    from audit_chapter_coverage import read_source

    source, passages = read_source(
        directory / "benchmark.sqlite3", index["source"]["tenant"], index["source"]["course"]
    )
    if source != index["source"]:
        raise ValueError("Stored source no longer matches the run")
    result["candidate_diagnostics"] = candidate_diagnostics(outcomes, passages)
    if pricing_path:
        pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
        result["public_list_price_equivalent"] = price_equivalent(outcomes, pricing)
        result["pricing_reference_sha256"] = digest(pricing_path)
    lines = [
        "# Hybrid benchmark results",
        "",
        f"Collection state: **{index['status']}**. Dataset: natural-student-v2. "
        "One run, hybrid only; no assisted diagnostic pairs.",
        "",
        "Reference answers remain proposed. The following checks are diagnostics, not semantic accuracy. "
        "No full meaning-based review has been completed; no separate paid judge was used.",
        "",
        "| Measure | Result |",
        "| --- | --- |",
        f"| Collected cases | {len(outcomes)}/{selected} |",
        f"| Attempted chatbot turns (including setup) | {index['attempted_query_turns']}/{index['planned_query_turns']} |",
        f"| Successful scored responses | {summary['successful_responses']} |",
        f"| Operational errors | {summary['operational_errors']} |",
        f"| Setup operational errors | {summary['setup_errors']} |",
        f"| Expected-status matches | {summary['status_matches']}/{selected} |",
        f"| All provisional automatic checks passed | {summary['automatic_checks_passed']}/{selected} |",
        f"| Median scored-response latency | {summary['latency_ms']['median']} ms |",
        f"| p95 scored-response latency | {summary['latency_ms']['p95']} ms |",
        "",
        "Unattempted/error cases are not discarded from the planned denominator. "
        "Latency excludes setup and errors; reported token usage includes captured setup calls.",
        "",
        "## Breakdown by track",
        "",
        "| Track | Collected | Status matches | All automatic checks | Errors |",
        "| --- | --- | --- | --- | --- |",
    ]
    if index.get("selection"):
        result["selection"] = index["selection"]
        lines[2:2] = [index["selection"]["limitation"], ""]
    for name, track in summary["tracks"].items():
        lines.append(
            f"| {name} | {track['collected']} | {track['status_matches']} | "
            f"{track['automatic_checks_passed']} | {track['errors']} |"
        )
    rates = result["diagnostic_rates"]
    lines += [
        "",
        f"Automatic-check micro rate: {rates['micro_automatic_check_rate']:.1%}; "
        f"track-macro rate: {rates['track_macro_automatic_check_rate']:.1%}; "
        f"semantic-family-macro rate: {rates['family_macro_automatic_check_rate']:.1%}.",
        "",
        rates["limitation"],
        "",
    ]
    if result.get("execution_notes"):
        note = result["execution_notes"]
        lines += [
            "## Timing caveat",
            "",
            note["timing_warning"],
            "",
            f"Affected case: {note['affected_scored_case']}; raw recorded latency: "
            f"{note['raw_recorded_latency_ms']} ms. No raw timing was changed and no paid question was rerun.",
            "",
            note["interpretation"],
            "",
        ]
    if spot_review.exists():
        lines += [
            "## Selected spot-checks",
            "",
            "[Six selected failure spot-checks](spot-review.md) document observed planning, "
            "retrieval and completeness issues. This is not a full or blind semantic evaluation.",
            "",
        ]
    runtime = summary["runtime"]
    lines += [
        "",
        "## Usage and cost",
        "",
        f"Captured provider calls: {runtime['observed_provider_calls']}. "
        f"Reported tokens: {runtime['reported_token_totals']}.",
        "",
    ]
    if pricing_path:
        price = result["public_list_price_equivalent"]
        lines += [
            f"Undiscounted public OpenAI list-price equivalent for captured calls: "
            f"**${price['captured_calls_undiscounted_equivalent_usd']:.4f} USD** "
            f"(rates checked {price['pricing_checked_date']}). This is **not the actual provider bill**.",
            "",
        ]
        lines += ["- " + limitation for limitation in price["limitations"]]
        lines += [
            "",
            "Rate sources: [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), "
            "[text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small).",
            "",
        ]
    lines += [
        "## Flagged cases",
        "",
        "These require review. An evidence-anchor mismatch alone does not prove that an answer is wrong.",
        "",
        "| ID | Track | Expected → actual | Reason | Failed checks |",
        "| --- | --- | --- | --- | --- |",
    ]
    for flag in result["diagnostics"]["flagged_cases"]:
        lines.append(
            f"| {flag['id']} | {flag['track']} | {flag['expected_status']} → "
            f"{flag['actual_status']} | {flag['reason']} | {', '.join(flag['failed_checks'])} |"
        )
    lines += [
        "",
        "## Next review",
        "",
        "Read [every question and answer](answers.md), then assess directness, factual support, "
        "completeness, boundaries, grouping and language against the source-bound references. "
        "Inspect raw captures for retrieval candidates, planning repairs and per-part decisions. "
        "Do not compare this percentage directly with older 20/44-question benchmarks.",
        "",
    ]
    for name in ("summary.json", "summary.md"):
        if (directory / name).exists():
            raise FileExistsError("Refusing to overwrite existing final summary: " + name)
    (directory / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pricing-reference", type=Path)
    cli = parser.parse_args()
    result = build(cli.directory, cli.pricing_reference)
    print(json.dumps({"status": result["status"], "summary": str(cli.directory / "summary.md")}))


if __name__ == "__main__":
    main()
