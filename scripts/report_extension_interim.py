"""Preserve all 56 catalog questions with partial v6 results, never fabricate missing answers."""

import argparse
import json
from collections import Counter
from pathlib import Path

import benchmark as bench
from export_scored_answers import LABELS, MODES, cell, load_judgments, render, retry_appendix
from inspect_chapter_answers import cost_rows
from recover_benchmark_errors import fixed_config
from report_benchmark_recovery import build

from hrlearnium.eval_tools import digest


def read_captures(original, recovery, pack, review):
    if review["reviewer"] != "Assistant" or review["human_approved"] is not False:
        raise ValueError("Assistant-only review required")
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    cases = [json.loads(s) for s in (pack / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    if digest(pack / "cases.jsonl") != manifest["cases_sha256"]:
        raise ValueError("Reference dataset changed")
    selected, errors, indexes = {}, [], []
    definitions = {c["id"]: c for c in cases}
    for phase, directory in [("original", original), ("recovery", recovery)]:
        if digest(directory / "index.json") != review[phase + "_index_sha256"]:
            raise ValueError("Review refers to different collection")
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if index["pending_request"] or index["status"] != "stopped":
            raise ValueError("Expected stopped checkpoint")
        if index["dataset_sha256"] != manifest["cases_sha256"]:
            raise ValueError("Dataset mismatch")
        if indexes and index["fixed_configuration"] != indexes[0]["fixed_configuration"]:
            raise ValueError("Configuration drift")
        if indexes and index["original_index_sha256"] != digest(original / "index.json"):
            raise ValueError("Recovery provenance mismatch")
        indexes.append(index)
        seen = set()
        for capture in index["captures"]:
            key = (capture["case_id"], capture["mode"])
            if key in seen or Path(capture["report"]).name != capture["report"]:
                raise ValueError("Duplicate/unsafe capture")
            seen.add(key)
            path = directory / capture["report"]
            if digest(path) != capture["report_sha256"]:
                raise ValueError("Capture changed")
            raw = json.loads(path.read_text(encoding="utf-8"))
            o = raw["outcomes"][0]
            if (
                o["id"] != key[0]
                or o["extension_definition"] != definitions[key[0]]
                or o["question"] != definitions[key[0]]["question"]
                or raw["metadata"]["requested_retrieval_mode"] != key[1]
            ):
                raise ValueError("Case mismatch")
            o.update(capture_phase=phase, capture_sha256=digest(path), capture_path=str(path))
            if o.get("error"):
                errors.append(
                    {"case_id": key[0], "mode": key[1], "phase": phase, "error": o["error"]}
                )
                continue
            if (
                key in selected
                or fixed_config(o["evaluation"]["configuration"]) != index["fixed_configuration"]
            ):
                raise ValueError("Semantic replacement or configuration drift")
            selected[key] = o
        if len(seen) != index["attempted_queries"]:
            raise ValueError("Attempt count mismatch")
    judgments = {}
    for j in review["judgments"]:
        for mode in j["modes"]:
            key = (j["case_id"], mode)
            if key in judgments or j["score"] not in {0, 2, 4, 5} or not j["note"]:
                raise ValueError("Invalid manual judgment")
            judgments[key] = {"score": j["score"], "note": j["note"]}
    if set(judgments) != set(selected):
        raise ValueError("Every actual answer must have an explicit review; no invented scores")
    return cases, selected, errors, judgments, indexes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    review = json.loads(cli.review.read_text(encoding="utf-8"))
    cases, selected, errors, judgments, indexes = read_captures(
        cli.original, cli.recovery, cli.pack, review
    )
    reports = {
        m: {"outcomes": [selected[c["id"], m] for c in cases if (c["id"], m) in selected]}
        for m in MODES
    }
    costs = cost_rows(reports)
    groups, _ = build(
        Path("evals/results/chapter-coverage-20261019-102221"),
        Path("evals/results/chapter-recovery-20261019-105625"),
    )
    old = render(
        next(iter(groups.values())),
        load_judgments(
            Path("evals/results/chapter-recovered-summary-20261019-105625/assistant-review.md")
        ),
        created_date="2026-10-20",
    )
    old_tables = "## Chapter 1\n" + old.split("## Chapter 1\n", 1)[1]
    old_tables = "\n".join("##" + s if s.startswith("#") else s for s in old_tables.splitlines())
    lines = [
        "# Benchmark catalog — 56 questions; v6 collection INCOMPLETE",
        "",
        "Assembled 2026-10-20 (Asia/Tehran). All 56 questions and three mode columns are included, "
        "but missing responses are explicitly marked. This is **not a completed final benchmark "
        "or a clean 56-question v6 rerun**.",
        "",
        "## Current result",
        "",
        "- Historical v5: 44 cases × 3 modes = 132 captured outcomes, with original reviews preserved.",
        "- New v6: 12 cases × 3 modes planned = 36 pairs. Collection stopped after 21 total attempts "
        "across the original and recovery phases: 19 actual responses, one local query HTTP 429, "
        "and one 300-second query timeout. The failed metadata pair was recovered once; "
        "there are 16 unattempted pairs and one unresolved timed-out pair.",
        "- All 18 metadata responses are correct in this assistant review. All modes use the same "
        "deterministic metadata route; zero provider calls were recorded for those cases. "
        "This does not compare retriever quality.",
        "- The single completed content bundle (hybrid, v6-multi-07) correctly answers both facts "
        "with appropriate citations, but wrongly answers a Persian question entirely in English. "
        "It scores 2/5 for material language/UX revision, despite passing automated evidence checks.",
        "- Reranked v6-multi-07 timed out after 300 seconds. The worker was still using CPU after "
        "client timeout; its dedicated port-8002 server was stopped to prevent further background "
        "work/provider calls. CPU activity and model loading were observed, but no completed "
        "stage trace exists, so the exact blocking stage is not proven.",
        "- No semantic outcomes were retried, no production pipeline fix was made, and no paid "
        "judge was used. Timeout is an operational error, not a refusal or a zero-rated answer. "
        "Its unreported provider usage is unknown.",
        "",
        "## Assessment and next action",
        "",
        "Metadata handling works for the tested document structure. Multi-question handling is "
        "not yet validated across the three modes: only one of the 18 planned bundle responses "
        "completed. Do not claim that partial-answer handling or reranking passed this live benchmark.",
        "",
        "Before spending on the remainder, decide whether to diagnose/optimize the local reranker "
        "or deliberately allow longer test timeouts. Preserve the timeout as a failure of the "
        "current five-minute operational budget. Separately fix the planner/response-language "
        "drift, with a Persian multi-question regression test. A fix changes the pipeline version: "
        "collect a new labeled run rather than overwrite these results. Use >=15-second pacing "
        "because authenticated citation/outline GETs also count toward app rate limits.",
        "",
        "## Scoring and limits",
        "",
        "Assistant source review, not instructor approval: 5/5 = correct, complete and supported; "
        "4/5 = correct but needs wording/focus polish; 2/5 = material revision such as a language, "
        "context or citation problem; 0/5 = failed answer/false refusal. Appropriate expected "
        "refusals can score 5. Scores are ordinal, not accuracy percentages. Missing/error "
        "responses receive no semantic score. All new references are assistant-source-audited, "
        "not independently owner-approved; the owner authorized inclusion/execution.",
        "",
        "One course, one source revision, one repetition; development questions, not a held-out "
        "general LMS benchmark. Do not aggregate v5 and partial v6 scores into an architecture ranking.",
        "",
        "| New v6 mode | Actual responses | Evidence/contract checks | 5/5 | 2/5 | Pending/error pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in MODES:
        values = reports[m]["outcomes"]
        scores = Counter(judgments[o["id"], m]["score"] for o in values)
        lines.append(
            f"| {LABELS[m]} | {len(values)}/12 | "
            f"{sum(o['extension_checks']['pass'] for o in values)}/{len(values)} | "
            f"{scores[5]} | {scores[2]} | {12 - len(values)} |"
        )
    lines += [
        "",
        "## Recorded cost — incomplete, not a bill",
        "",
        "Normalized standard uncached rates checked 2026-10-20: "
        "[GPT-4.1-mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), "
        "$0.40/M input and $1.60/M output; "
        "[text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small), "
        "$0.02/M input. Cached discounts, gateway pricing/markup, local reranker CPU, "
        "indexing and hosting are excluded. The timed-out request may have incurred "
        "additional unreported charges. Flattened provider calls are counted once; "
        "nested part traces are not double-counted.",
        "",
        "| Mode | Responses with captured usage | Chat input | Chat output | Embedding tokens | List estimate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in costs:
        lines.append(
            f"| {LABELS[c['mode']]} | {c['queries']} | {c.get('llm_input_tokens', 0):,} | "
            f"{c.get('llm_output_tokens', 0):,} | {c.get('embedding_tokens', 0):,} | "
            f"${c.get('standard_list_usd', 0):.6f} |"
        )
    lines += [
        "",
        "These incomplete, metadata-heavy samples cannot rank mode costs. "
        "[Historical v5 quality/cost comparison](../chapter-recovered-summary-20261019-105625/assistant-review.md) "
        "remains separate. It favored hybrid's cost/conciseness on that small course, "
        "but does not establish a current v6 winner.",
        "",
        "## Provenance",
        "",
        f"[Source-audited new references](../../metadata-multipart-v6/references.md) · "
        f"[Original ledger](../{cli.original.name}/index.json) · "
        f"[Recovery ledger](../{cli.recovery.name}/index.json) · "
        "[Bound assistant judgments](assistant-review.json) · [Machine summary](summary.json)",
        "",
        "## Part A — historical v5: 44 questions and complete captured responses",
        "",
        old_tables,
        "",
        "## Part B — v6: 12 new questions, available answers and explicit gaps",
        "",
    ]
    for n, c in enumerate(cases, 45):
        lines += [
            f"### Question {n} — {c['id']}",
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
            "| Mode | Complete actual answer or collection status | Assistant score | Phase / query latency |",
            "|---|---|:---:|---|",
        ]
        for m in MODES:
            key = (c["id"], m)
            if key in selected:
                o, j = selected[key], judgments[key]
                lines.append(
                    f"| {LABELS[m]} | {cell(o['response']['answer'])} | {j['score']}/5 | "
                    f"{o['capture_phase']} / {o['query_latency_ms'] / 1000:.3f}s |"
                )
            else:
                error = next(
                    (e for e in reversed(errors) if (e["case_id"], e["mode"]) == key), None
                )
                label = (
                    "Operational error: " + str(error["error"])
                    if error
                    else "Not attempted — collection stopped"
                )
                lines.append(f"| {LABELS[m]} | {cell(label)} | Not scored | — |")
        lines.append("")
        for m in MODES:
            if (c["id"], m) in judgments:
                lines.append(f"- {LABELS[m]}: {judgments[c['id'], m]['note']}")
        lines.append("")
    lines.append(retry_appendix(Path("evals/results/full-context-diagnostic-20261019-185627")))
    cli.output.mkdir(parents=True, exist_ok=False)
    (cli.output / "benchmark-review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bench.write_json(
        cli.output / "assistant-review.json",
        {
            "reviewer": "Assistant",
            "human_approved": False,
            "review_source_sha256": digest(cli.review),
            "judgments": [
                {
                    "case_id": k[0],
                    "mode": k[1],
                    **v,
                    "capture_sha256": selected[k]["capture_sha256"],
                }
                for k, v in judgments.items()
            ],
        },
    )
    bench.write_json(
        cli.output / "summary.json",
        {
            "status": "incomplete",
            "catalog_questions": 56,
            "historical_v5_outcomes": 132,
            "v6_planned_pairs": 36,
            "v6_actual_responses": len(selected),
            "v6_total_attempts": sum(i["attempted_queries"] for i in indexes),
            "v6_errors": errors,
            "v6_cost_recorded_only": costs,
            "original_index_sha256": digest(cli.original / "index.json"),
            "recovery_index_sha256": digest(cli.recovery / "index.json"),
            "no_clean_overall_architecture_ranking": True,
            "dedicated_server_stopped": True,
        },
    )
    print(
        json.dumps(
            {
                "output": str(cli.output),
                "new_actual_responses": len(selected),
                "network_requests": 0,
                "status": "incomplete",
            }
        )
    )


if __name__ == "__main__":
    main()
