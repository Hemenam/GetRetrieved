"""Offline, hash-checked 56-question catalog report; historical and v6 scores stay separate."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from statistics import median

import benchmark as bench
from export_scored_answers import (
    LABELS,
    MODES,
    cell,
    load_judgments,
    render,
    retry_appendix,
)
from inspect_chapter_answers import cost_rows
from recover_benchmark_errors import fixed_config
from report_benchmark_recovery import build

from hrlearnium.eval_tools import digest, runtime_summary


def collect(original, recovery, pack):
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    cases = [json.loads(s) for s in (pack / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    if digest(pack / "cases.jsonl") != manifest["cases_sha256"]:
        raise ValueError("Reference dataset changed")
    selected, errors, indexes = {}, [], []
    for phase, directory in [("original", original), ("operational_recovery", recovery)]:
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        indexes.append(index)
        if index["dataset_sha256"] != manifest["cases_sha256"] or index["pending_request"]:
            raise ValueError("Dataset mismatch or ambiguous pending request")
        if phase == "operational_recovery" and (
            index["status"] != "complete"
            or index["original_index_sha256"] != digest(original / "index.json")
            or index["fixed_configuration"] != indexes[0]["fixed_configuration"]
        ):
            raise ValueError("Recovery incomplete or not comparable")
        seen = set()
        for capture in index["captures"]:
            key = (capture["case_id"], capture["mode"])
            if key in seen or Path(capture["report"]).name != capture["report"]:
                raise ValueError("Duplicate or unsafe capture")
            seen.add(key)
            path = directory / capture["report"]
            if digest(path) != capture["report_sha256"]:
                raise ValueError("Report hash changed")
            report = json.loads(path.read_text(encoding="utf-8"))
            outcome = copy.deepcopy(report["outcomes"][0])
            definition = next(c for c in cases if c["id"] == key[0])
            if (
                outcome["id"] != key[0]
                or outcome["question"] != definition["question"]
                or outcome["extension_definition"] != definition
                or report["metadata"]["requested_retrieval_mode"] != key[1]
                or report["dataset_sha256"] != manifest["cases_sha256"]
                or outcome.get("configuration_mismatch")
            ):
                raise ValueError("Capture does not match pinned case/mode")
            outcome["capture_phase"] = phase
            outcome["capture_path"] = str(path)
            if outcome.get("error"):
                errors.append(outcome)
                continue
            if key in selected:
                raise ValueError("A semantic outcome was replaced")
            trace = outcome["evaluation"]
            if fixed_config(trace["configuration"]) != index["fixed_configuration"]:
                raise ValueError("Configuration drift")
            if (
                trace.get("source_passage_count", 0)
                and trace.get("source_corpus_sha256") != manifest["source"]["source_corpus_sha256"]
            ):
                raise ValueError("Source drift")
            selected[key] = outcome
        if len(seen) != index["attempted_queries"]:
            raise ValueError("Incomplete attempt ledger")
    expected = {(c["id"], m) for c in cases for m in MODES}
    if set(selected) != expected:
        raise ValueError("Not all 36 unique outcomes are available")
    reports = {m: {"outcomes": [selected[c["id"], m] for c in cases]} for m in MODES}
    return cases, reports, errors, indexes


def validate_review(review, cases, reports):
    if review.get("reviewer") != "Assistant" or review.get("human_approved") is not False:
        raise ValueError("Review must be explicitly assistant-only")
    lookup = {(r["case_id"], r["mode"]): r for r in review["judgments"]}
    if len(lookup) != len(review["judgments"]) or set(lookup) != {
        (c["id"], m) for c in cases for m in MODES
    }:
        raise ValueError("Exactly 36 judgments required")
    for mode, report in reports.items():
        for outcome in report["outcomes"]:
            judgment = lookup[outcome["id"], mode]
            if judgment["score"] not in {0, 2, 4, 5} or not judgment["note"].strip():
                raise ValueError("Score/note missing")
            if judgment["capture_sha256"] != digest(Path(outcome["capture_path"])):
                raise ValueError("Judgment attached to different response")
    return lookup


def summary_rows(cases, reports, judgments):
    costs = {r["mode"]: r for r in cost_rows(reports)}
    result = []
    for mode in MODES:
        outcomes = reports[mode]["outcomes"]
        metadata = outcomes[:6]
        bundles = outcomes[6:]
        scores = Counter(judgments[o["id"], mode]["score"] for o in outcomes)
        result.append(
            {
                "mode": mode,
                "selected_outcomes": len(outcomes),
                "automated_pass": sum(o["extension_checks"]["pass"] for o in outcomes),
                "assistant_score_counts": {str(s): scores[s] for s in [5, 4, 2, 0]},
                "status_counts": dict(Counter(o["actual_status"] for o in outcomes)),
                "metadata_median_seconds": median(o["query_latency_ms"] for o in metadata) / 1000,
                "bundle_median_seconds": median(o["query_latency_ms"] for o in bundles) / 1000,
                "bundle_max_seconds": max(o["query_latency_ms"] for o in bundles) / 1000,
                "metadata_provider_calls": sum(
                    len(o["evaluation"]["provider_calls"]) for o in metadata
                ),
                "usage_and_list_estimate": costs[mode],
                "runtime": runtime_summary(outcomes),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    cases, reports, errors, indexes = collect(cli.original, cli.recovery, cli.pack)
    review = json.loads(cli.review.read_text(encoding="utf-8"))
    judgments = validate_review(review, cases, reports)
    rows = summary_rows(cases, reports, judgments)
    historical = Path("evals/results/chapter-recovered-summary-20261019-105625")
    groups, _ = build(
        Path("evals/results/chapter-coverage-20261019-102221"),
        Path("evals/results/chapter-recovery-20261019-105625"),
    )
    old_judgments = load_judgments(historical / "assistant-review.md")
    old = render(next(iter(groups.values())), old_judgments, created_date="2026-10-20")
    old_tables = "## Chapter 1\n" + old.split("## Chapter 1\n", 1)[1]
    old_tables = "\n".join("##" + s if s.startswith("#") else s for s in old_tables.splitlines())
    lines = [
        "# Final benchmark catalog: 56 questions, three modes, answers and assistant scores",
        "",
        "Assembled 2026-10-20 (Asia/Tehran). This is a combined review document, **not a clean "
        "56-question rerun**: Part A contains 44 historical v5 cases; Part B contains 12 new v6 cases.",
        "",
        "The 168 comparison outcomes preserve their complete captured answer text, including refusals. "
        "The separate historical diagnostic retry is excluded from scores and shown in an appendix. "
        "Citations [1], [2], etc. refer to each raw response's own excerpts, not Markdown footnotes.",
        "",
        "## Scope and provenance",
        "",
        "- One Persian course, 11 recognized chapters, 44 stored passages. All modes used the same "
        "source revision. This does not demonstrate generalization to arbitrary LMS courses.",
        "- New references were source-audited by the assistant; the owner authorized inclusion/run, "
        "but has not independently approved these 12 reference answers or graded the responses.",
        "- Initial v6 collection: 18 attempts, 17 responses and one local HTTP 429. "
        "Explicit recovery: 19 attempts (the one operational failure plus 18 unattempted pairs). "
        "Total: 37 outer attempts, 36 unique selected outcomes. The 429 is retained, not scored as a refusal.",
        "- Recovery used 15-second pacing, unchanged application configuration/source, no semantic "
        "retries, no paid judge and no production fixes. Authenticated citation/outline GETs add HTTP "
        "traffic beyond the outer-query count. App rate limits also count these GETs.",
        "- Original historical capture/recovery was also interrupted; do not merge v5/v6 results into "
        "an overall accuracy, latency improvement or architecture ranking.",
        "",
        f"[New references](../../metadata-multipart-v6/references.md) · "
        f"[Original v6 ledger](../{cli.original.name}/index.json) · "
        f"[Recovery ledger](../{cli.recovery.name}/index.json) · [Assistant judgments](assistant-review.json)",
        "",
        "## Score rubric",
        "",
        "These are assistant judgments against the course, not instructor approval or automated "
        "accuracy percentages. 5 = correct, complete, supported; 4 = correct but wording/focus needs "
        "polish; 2 = material revision needed (such as contextual/citation mismatch); 0 = failed "
        "answer/false refusal. An expected, appropriate refusal can score 5. A 5 means no issue found "
        "in this review, not guaranteed truth. The raw 429 has no semantic score.",
        "",
        "## New v6 results only (12 cases per mode)",
        "",
        "Automated checks cover response contract, expected status, per-part evidence anchors, "
        "metadata completeness and provenance. They do not establish generated-answer correctness. "
        "Read the individual assistant-review notes below.",
        "",
        "| Mode | Automated checks | 5/5 | 4/5 | 2/5 | 0/5 | Metadata median | Bundles median | Bundle max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        counts = r["assistant_score_counts"]
        lines.append(
            f"| {LABELS[r['mode']]} | {r['automated_pass']}/12 | "
            f"{counts['5']} | {counts['4']} | {counts['2']} | {counts['0']} | "
            f"{r['metadata_median_seconds']:.3f}s | {r['bundle_median_seconds']:.3f}s | "
            f"{r['bundle_max_seconds']:.3f}s |"
        )
    lines += [
        "",
        "Six metadata-only cases use the same deterministic route across all three selected "
        "modes. They are not evidence that one retriever is better. The other six cases test "
        "multi-part/mixed routing. Only one repetition was measured; retain cold-load outliers.",
        "",
        "## New v6 recorded usage and normalized cost",
        "",
        "OpenAI standard uncached list rates checked 2026-10-20: "
        "[GPT-4.1-mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) "
        "input $0.40/M, output $1.60/M; "
        "[text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small) "
        "input $0.02/M. Cached input discounts were not captured/applied. These are normalized "
        "estimates, **not your configured gateway's bill**. Hosting, local reranker CPU, "
        "indexing and unreported failed/ambiguous provider usage are excluded. "
        "Flattened provider calls are counted once; nested part traces are not added again.",
        "",
        "| Mode | Chat input tokens | Chat output tokens | Embedding tokens | Recorded calls | List estimate (12 queries) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        c = r["usage_and_list_estimate"]
        lines.append(
            f"| {LABELS[r['mode']]} | {c.get('llm_input_tokens', 0):,} | "
            f"{c.get('llm_output_tokens', 0):,} | {c.get('embedding_tokens', 0):,} | "
            f"{sum(c['calls'].values())} | ${c.get('standard_list_usd', 0):.6f} |"
        )
    total = sum(r["usage_and_list_estimate"].get("standard_list_usd", 0) for r in rows)
    lines += [
        "",
        f"Selected v6 outcomes total: **${total:.6f}** at those rates. "
        "All six metadata-only cases per mode recorded zero provider calls. "
        "Do not use this metadata-heavy mix as a cost estimate for all future student traffic.",
        "",
        "## Historical v5 comparison (44 cases per mode)",
        "",
        "| Mode | 5/5 | 4/5 | 2/5 | 0/5 | Historical list estimate |",
        "|---|---:|---:|---:|---:|---:|",
        "| Full context | 38 | 2 | 3 | 1 | $0.340007 |",
        "| Hybrid | 40 | 3 | 1 | 0 | $0.166159 |",
        "| Hybrid + reranker | 37 | 7 | 0 | 0 | $0.166917 |",
        "",
        "These older results favored hybrid on cost and concise answers in this small course. "
        "Reranking did not establish a quality advantage and added CPU latency. Full context's "
        "original false refusal remains 0/5 despite its later successful diagnostic retry. "
        "This is historical evidence, not a current v6 architecture winner. "
        "[Full historical assessment/cost limitations](../chapter-recovered-summary-20261019-105625/assistant-review.md)",
        "",
        "## Findings and next steps",
        "",
        review["findings"],
        "",
        "## Part A — 44 historical v5 questions",
        "",
        old_tables,
        "",
        "## Part B — 12 new v6 questions",
        "",
    ]
    lookup = {m: {o["id"]: o for o in reports[m]["outcomes"]} for m in MODES}
    for number, case in enumerate(cases, 45):
        lines += [
            f"### Question {number} — {case['id']}",
            "",
            "| Question |",
            "|---|",
            f"| {cell(case['question'])} |",
            "",
            f"Expected overall status: `{case['expected_status']}`. Expected parts:",
            "",
        ]
        for i, p in enumerate(case["parts"], 1):
            lines.append(
                f"{i}. `{p['kind']}` / `{p['status']}`: " + "; ".join(cell(f) for f in p["facts"])
            )
        lines += [
            "",
            "| Mode | Complete captured answer | My score | Automatic checks | Query latency | Phase |",
            "|---|---|:---:|:---:|---:|---|",
        ]
        for mode in MODES:
            o, j = lookup[mode][case["id"]], judgments[case["id"], mode]
            lines.append(
                f"| {LABELS[mode]} | {cell(o['response']['answer'])} | {j['score']}/5 | "
                f"{'Pass' if o['extension_checks']['pass'] else 'Fail'} | "
                f"{o['query_latency_ms'] / 1000:.3f}s | {o['capture_phase']} |"
            )
        lines += ["", "Assistant comparison:", ""]
        for mode in MODES:
            lines.append(f"- {LABELS[mode]}: {judgments[case['id'], mode]['note']}")
        lines.append("")
    lines.append(retry_appendix(Path("evals/results/full-context-diagnostic-20261019-185627")))
    cli.output.mkdir(parents=True, exist_ok=False)
    (cli.output / "final-benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bench.write_json(cli.output / "assistant-review.json", review)
    bench.write_json(
        cli.output / "summary.json",
        {
            "kind": "historical_v5_plus_v6_extension_not_clean_rerun",
            "catalog_questions": 56,
            "comparison_outcomes": 168,
            "v6_selected_outcomes": 36,
            "v6_outer_attempts": sum(i["attempted_queries"] for i in indexes),
            "v6_operational_errors": [
                {k: o.get(k) for k in ["id", "error", "capture_phase"]} for o in errors
            ],
            "v6_modes": rows,
            "assistant_review_sha256": digest(cli.review),
            "original_v6_index_sha256": digest(cli.original / "index.json"),
            "recovery_index_sha256": digest(cli.recovery / "index.json"),
            "reference_dataset_sha256": digest(cli.pack / "cases.jsonl"),
            "human_approved": False,
            "rates_verified_date": "2026-10-20",
        },
    )
    print(json.dumps({"output": str(cli.output), "network_requests": 0, "v6_cost_estimate": total}))


if __name__ == "__main__":
    main()
