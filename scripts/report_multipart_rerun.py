"""Offline report for a separately authorized multipart repeat, with manual review binding."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

import benchmark as bench
from export_scored_answers import LABELS, MODES, cell
from inspect_chapter_answers import cost_rows
from recover_benchmark_errors import fixed_config

from hrlearnium.eval_tools import digest, runtime_summary


def load_run(directory):
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "inputs/manifest.json").read_text(encoding="utf-8"))
    cases_path = directory / "inputs/cases.jsonl"
    if (
        digest(cases_path) != index["dataset_sha256"]
        or digest(directory / "inputs/manifest.json") != index["manifest_sha256"]
    ):
        raise ValueError("Snapshot inputs changed")
    cases = [json.loads(s) for s in cases_path.read_text(encoding="utf-8").splitlines()]
    cases = [c for c in cases if len(c["parts"]) > 1]
    if len(cases) != 6 or index["planned_queries"] != 18:
        raise ValueError("Expected six multipart cases")
    expected = {(c["id"], m) for c in cases for m in MODES}
    outcomes = {}
    for capture in index["captures"]:
        key = (capture["case_id"], capture["mode"])
        if (
            key not in expected
            or key in outcomes
            or Path(capture["report"]).name != capture["report"]
        ):
            raise ValueError("Unexpected/duplicate/unsafe capture")
        path = directory / capture["report"]
        if digest(path) != capture["report_sha256"]:
            raise ValueError("Capture hash changed")
        raw = json.loads(path.read_text(encoding="utf-8"))
        o = raw["outcomes"][0]
        definition = next(c for c in cases if c["id"] == key[0])
        if (
            o["id"] != key[0]
            or o["extension_definition"] != definition
            or o["question"] != definition["question"]
            or raw["metadata"]["requested_retrieval_mode"] != key[1]
        ):
            raise ValueError("Capture case/mode mismatch")
        if o.get("configuration_mismatch"):
            raise ValueError("Configuration mismatch")
        if o.get("evaluation"):
            trace = o["evaluation"]
            if fixed_config(trace["configuration"]) != index["fixed_configuration"]:
                raise ValueError("Configuration drift")
            if (
                trace.get("source_passage_count", 0)
                and trace.get("source_corpus_sha256") != manifest["source"]["source_corpus_sha256"]
            ):
                raise ValueError("Source drift")
        o["capture_sha256"] = digest(path)
        outcomes[key] = o
    if (
        index["status"] != "complete"
        or index["pending_request"] is not None
        or set(outcomes) != expected
        or index["attempted_queries"] != 18
    ):
        raise ValueError("Rerun not fully captured; report as incomplete separately")
    return index, cases, outcomes


def bind_review(review, index_path, outcomes):
    if (
        review["reviewer"] != "Assistant"
        or review["human_approved"] is not False
        or review["index_sha256"] != digest(index_path)
    ):
        raise ValueError("Review provenance mismatch")
    judgments = {}
    for j in review["judgments"]:
        key = (j["case_id"], j["mode"])
        if key in judgments or j["score"] not in {0, 2, 4, 5} or not j["note"]:
            raise ValueError("Invalid/duplicate manual judgment")
        if key not in outcomes or outcomes[key].get("error"):
            raise ValueError("Errors have no semantic score")
        judgments[key] = {**j, "capture_sha256": outcomes[key]["capture_sha256"]}
    if set(judgments) != {k for k, o in outcomes.items() if not o.get("error")}:
        raise ValueError("Missing explicit review; grades cannot be inferred from automated checks")
    return judgments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    index, cases, outcomes = load_run(cli.run)
    review = json.loads(cli.review.read_text(encoding="utf-8"))
    judgments = bind_review(review, cli.run / "index.json", outcomes)
    reports = {
        m: {
            "outcomes": [
                outcomes[c["id"], m] for c in cases if not outcomes[c["id"], m].get("error")
            ]
        }
        for m in MODES
    }
    if any(not r["outcomes"] for r in reports.values()):
        raise ValueError("A mode has no responses; do not publish comparable cost averages")
    costs = {r["mode"]: r for r in cost_rows(reports)}
    summary = []
    for m in MODES:
        values = reports[m]["outcomes"]
        scores = Counter(judgments[c["id"], m]["score"] for c in cases if (c["id"], m) in judgments)
        summary.append(
            {
                "mode": m,
                "responses": len(values),
                "automatic_pass": sum(o["extension_checks"]["pass"] for o in values),
                "score_counts": {str(s): scores[s] for s in [5, 4, 2, 0]},
                "facts_status_correct": sum(
                    judgments[o["id"], m]["facts_status_correct"] for o in values
                ),
                "language_correct": sum(judgments[o["id"], m]["language_correct"] for o in values),
                "median_query_seconds": median(o["query_latency_ms"] for o in values) / 1000,
                "max_query_seconds": max(o["query_latency_ms"] for o in values) / 1000,
                "cost": costs[m],
                "runtime": runtime_summary(values),
            }
        )
    lines = [
        "# Multipart benchmark — explicit fresh rerun",
        "",
        "Collected 2026-10-20 (Asia/Tehran): six multipart/mixed cases × three modes, 18 outer queries. "
        "These are new attempts, not replacements for earlier failures. One repetition only; "
        "assistant source-audited references, not independent instructor approval.",
        "",
        "## Conditions",
        "",
        "- Same source, GPT-4.1-mini gateway configuration, retrieval settings, application code and prompts; no language fix applied.",
        "- Dedicated CPU worker: OMP_NUM_THREADS=4, MKL_NUM_THREADS=4 (PyTorch intra-op verified as 4 in matching launch environment; inter-op default 14). "
        "Original run used 14 intra-op threads; this is not a controlled proof of the earlier timeout's cause.",
        "- Client timeout raised from 300 to 1200 seconds; 20-second pauses between queries. "
        "Full-context/hybrid pairs interleaved first, reranker block last. Stage and cold-start latencies retained; block order can confound timing.",
        "- Offline regression tests ran concurrently during part of collection; these timings are observational, not isolated performance measurements.",
        "- No hidden paid warmups, semantic retries, outer HTTP retries or paid judge. "
        "Provider's existing bounded transport retry policy unchanged. Every response/evidence is checkpointed and hash-checked.",
        "",
        "## Results",
        "",
        review["findings"],
        "",
        "| Mode | Automatic checks | Core expected facts/status (assistant) | Expected language | 5/5 | 4/5 | 2/5 | 0/5 | Median query | Max query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        c = r["score_counts"]
        lines.append(
            f"| {LABELS[r['mode']]} | {r['automatic_pass']}/6 | {r['facts_status_correct']}/6 | "
            f"{r['language_correct']}/6 | {c['5']} | {c['4']} | {c['2']} | {c['0']} | "
            f"{r['median_query_seconds']:.3f}s | {r['max_query_seconds']:.3f}s |"
        )
    lines += [
        "",
        "Automatic checks require expected per-part shape/status and each part's own evidence anchors, "
        "not just the presence of relevant text somewhere. A joint answer may cover facts yet fail "
        "multipart routing; distinguish these findings. Core-fact coverage is not full answer quality: "
        "additional unsupported claims can lower a score even when requested facts are present. "
        "Assistant factual review includes qualification, "
        "list completeness, refusal correctness and citation meaning. Language is reviewed separately.",
        "",
        "Scores are ordinal assistant judgments: 5 correct/complete/supported; 4 correct with polish "
        "issues; 2 material language/context/citation/routing revision; 0 failed/false refusal. "
        "An appropriate expected refusal can score 5. No score establishes production reliability.",
        "",
        "## Recorded usage and normalized cost",
        "",
        "Standard uncached list-rate estimate, not the configured gateway's invoice: "
        "[GPT-4.1-mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) $0.40/M input, $1.60/M output; "
        "[text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small) $0.02/M input "
        "(rates checked 2026-10-20). Cached discounts/markup unknown; excludes hosting, indexing, local "
        "reranker CPU and unreported failed usage. Only flattened provider calls are summed once.",
        "",
        "| Mode | Chat input | Chat output | Embedding tokens | Calls | Estimate for six questions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in MODES:
        c = costs[m]
        lines.append(
            f"| {LABELS[m]} | {c.get('llm_input_tokens', 0):,} | {c.get('llm_output_tokens', 0):,} | "
            f"{c.get('embedding_tokens', 0):,} | {sum(c['calls'].values())} | ${c.get('standard_list_usd', 0):.6f} |"
        )
    total = sum(c.get("standard_list_usd", 0) for c in costs.values())
    provider_attempts = sum(r["runtime"]["observed_provider_attempts"] for r in summary)
    provider_retries = sum(r["runtime"]["observed_retries"] for r in summary)
    provider_calls = sum(sum(c["calls"].values()) for c in costs.values())
    lines += [
        "",
        f"This rerun's recorded list estimate: **${total:.6f}**. Earlier attempts are excluded, "
        "not refunded/erased. This tiny mix is not a production spend forecast.",
        "",
        f"Provider transport diagnostics: {provider_calls} successful recorded calls across "
        f"{provider_attempts} attempts, including {provider_retries} bounded transport retries. "
        "These are not outer-query replays. Unreported usage from failed attempts is not priced.",
        "",
        "## Complete questions, expected meanings and actual answers",
        "",
    ]
    for number, c in enumerate(cases, 51):
        lines += [
            f"### Question {number} — {c['id']}",
            "",
            "| Question |",
            "|---|",
            f"| {cell(c['question'])} |",
            "",
            f"Expected overall status: `{c['expected_status']}`.",
            "",
        ]
        for i, p in enumerate(c["parts"], 1):
            lines.append(
                f"{i}. `{p['kind']}` / `{p['status']}`: " + "; ".join(cell(f) for f in p["facts"])
            )
        lines += [
            "",
            "| Mode | Complete captured answer | Assistant score | Automatic checks | Query latency |",
            "|---|---|:---:|:---:|---:|",
        ]
        for m in MODES:
            o = outcomes[c["id"], m]
            if o.get("error"):
                lines.append(f"| {LABELS[m]} | {cell(o['error'])} | Not scored | Error | — |")
                continue
            j = judgments[c["id"], m]
            lines.append(
                f"| {LABELS[m]} | {cell(o['response']['answer'])} | {j['score']}/5 | "
                f"{'Pass' if o['extension_checks']['pass'] else 'Fail'} | {o['query_latency_ms'] / 1000:.3f}s |"
            )
        lines.append("")
        for m in MODES:
            if (c["id"], m) in judgments:
                lines.append(f"- {LABELS[m]}: {judgments[c['id'], m]['note']}")
        lines.append("")
    lines += [
        "## Provenance and prior failures",
        "",
        f"[Raw rerun ledger](../{cli.run.name}/index.json) · [Bound assistant judgments](assistant-review.json) · "
        "[Machine summary](summary.json) · [Source-audited references](../../metadata-multipart-v6/references.md)",
        "",
        "[Earlier incomplete run and original failures](../metadata-multipart-v6-interim-20261020/benchmark-review.md) "
        "remain unchanged. The earlier hybrid English answer and reranker timeout are not replaced "
        "in their original ledger or score.",
        "",
    ]
    cli.output.mkdir(parents=True, exist_ok=False)
    report = "\n".join(lines) + "\n"
    (cli.output / "multipart-comparison.md").write_text(report, encoding="utf-8")
    bench.write_json(
        cli.output / "assistant-review.json",
        {
            "reviewer": "Assistant",
            "human_approved": False,
            "index_sha256": digest(cli.run / "index.json"),
            "judgments": list(judgments.values()),
        },
    )
    bench.write_json(
        cli.output / "summary.json",
        {
            "kind": "fresh_multipart_repeat_not_clean_full_catalog_run",
            "queries": 18,
            "modes": summary,
            "recorded_list_estimate_usd": total,
            "index_sha256": digest(cli.run / "index.json"),
            "review_source_sha256": digest(cli.review),
            "earlier_results_preserved": True,
        },
    )
    prior = Path(
        "evals/results/metadata-multipart-v6-interim-20261020/benchmark-review.md"
    ).read_text(encoding="utf-8")
    historical = prior.split("## Part A", 1)[1].split("## Part B", 1)[0]
    metadata = (
        "### Question 45" + prior.split("### Question 45", 1)[1].split("### Question 51", 1)[0]
    )
    combined = [
        "# Complete 56-question catalog — mixed collection phases",
        "",
        "All 56 questions have actual outcomes from all three modes. This is **not** a clean "
        "56-question v6 rerun: 44 historical v5 questions, six earlier v6 metadata questions, "
        "and six freshly repeated v6 multipart questions under the conditions below. No combined "
        "architecture score is valid across these phases. Earlier failed attempts remain in their "
        "original ledgers and linked review, not silently replaced.",
        "",
        "## Current multipart rerun assessment",
        "",
        report.split("## Complete questions", 1)[0].split("## Conditions", 1)[1],
        "",
        "## Part A" + historical,
        "## Part B — six earlier v6 metadata questions",
        "",
        metadata,
        "## Part C — six fresh v6 multipart questions",
        "",
        report.split("## Complete questions, expected meanings and actual answers", 1)[1],
    ]
    (cli.output / "benchmark-56-questions.md").write_text("\n".join(combined), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(cli.output),
                "questions": 56,
                "new_queries": 18,
                "recorded_list_estimate_usd": total,
                "network_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
