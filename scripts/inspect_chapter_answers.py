"""Read-only evidence/answer inspection and standard-list-price usage comparison.

No model calls, response grading or report mutation. Raw captures are hash-checked
by the recovery reporter. Prices are explicit assumptions, not an account bill.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_chapter_coverage import read_source
from report_benchmark_recovery import build


def cost_rows(reports):
    result = []
    for mode, report in reports.items():
        totals = Counter()
        stages = {}
        kinds = Counter()
        for outcome in report["outcomes"]:
            for call in outcome["evaluation"]["provider_calls"]:
                usage = call.get("usage") or {}
                model = call["model"]
                if model not in {"gpt-4.1-mini", "text-embedding-3-small"}:
                    raise ValueError("Review pricing for an unexpected model")
                if "prompt_tokens" not in usage:
                    raise ValueError("Missing recorded input usage")
                embedding = model == "text-embedding-3-small"
                input_tokens = usage["prompt_tokens"]
                output_tokens = usage.get("completion_tokens", 0)
                price = (input_tokens * (0.02 if embedding else 0.4) + output_tokens * 1.6) / 1e6
                totals["embedding_tokens" if embedding else "llm_input_tokens"] += input_tokens
                totals["llm_output_tokens"] += output_tokens
                totals["embedding_usd" if embedding else "llm_usd"] += price
                totals["standard_list_usd"] += price
                kinds[call["kind"]] += 1
                stage = stages.setdefault(call["stage"], Counter())
                stage["calls"] += 1
                stage["input_tokens"] += input_tokens
                stage["output_tokens"] += output_tokens
                stage["standard_list_usd"] += price
        count = len(report["outcomes"])
        result.append(
            {
                "mode": mode,
                "queries": count,
                "calls": dict(kinds),
                **totals,
                "per_query_usd": totals["standard_list_usd"] / count,
                "per_1000_queries_usd": totals["standard_list_usd"] / count * 1000,
                "stages": stages,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", type=int)
    parser.add_argument("--cost-output", type=Path, help="Create a new usage estimate JSON file")
    cli = parser.parse_args()
    groups, _ = build(
        Path("evals/results/chapter-coverage-20261019-102221"),
        Path("evals/results/chapter-recovery-20261019-105625"),
    )
    reports = next(iter(groups.values()))
    if cli.chapter is None:
        result = {
            "kind": "offline_standard_list_price_estimate",
            "not_an_account_bill": True,
            "collection": "Original semantic outcomes plus operational recovery; retry excluded",
            "rates_verified_date": "2026-10-19",
            "currency": "USD",
            "per_million_tokens": {
                "gpt-4.1-mini": {"input": 0.4, "output": 1.6, "cached_input_not_applied": 0.1},
                "text-embedding-3-small": {"input": 0.02, "output": 0},
            },
            "sources": [
                "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
                "https://developers.openai.com/api/docs/models/text-embedding-3-small",
            ],
            "limitations": [
                "Uses standard uncached list prices, not configured gateway/account prices.",
                "Cached-token breakdown was not captured; actual cache discounts are unknown.",
                "Excludes unreported failed/ambiguous attempts and the separately requested diagnostic retry.",
                "Excludes ingestion/indexing, database, hosting and local reranker compute.",
                "Per-1000 values are linear projections of these 44 questions, not measured scale costs.",
            ],
            "modes": cost_rows(reports),
        }
        if cli.cost_output:
            cli.cost_output.parent.mkdir(parents=True, exist_ok=True)
            with cli.cost_output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            print(json.dumps({"output": str(cli.cost_output), "network_requests": 0}))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if cli.cost_output:
        parser.error("--cost-output cannot be combined with --chapter")
    _, passages = read_source(Path("data/hrlearnium.sqlite3"), "customer-demo", "captain-storm")
    source = [p for p in passages if p["chapter_number"] == cli.chapter]
    print(json.dumps({"source_chapter": cli.chapter, "sections": source}, ensure_ascii=False))
    for original in reports["full_context"]["outcomes"]:
        if original["expected_chapters"] != [cli.chapter]:
            continue
        print(
            json.dumps(
                {
                    "id": original["id"],
                    "question": original["question"],
                    "gold": original["gold"]["required_facts"],
                },
                ensure_ascii=False,
            )
        )
        for mode, report in reports.items():
            outcome = next(o for o in report["outcomes"] if o["id"] == original["id"])
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "status": outcome["actual_status"],
                        "answer": outcome["response"]["answer"],
                        "evidence_ids": [e["id"] for e in outcome["response"].get("excerpts", [])],
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
