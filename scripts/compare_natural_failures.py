"""Offline paired comparison of a failure-only rerun. Never invokes a model."""

import argparse
import json
from pathlib import Path

from report_natural_benchmark import load_run, price_equivalent
from run_natural_benchmark import summarize, write_json


def build(parent, rerun, pricing_path=None):
    before_index, before, before_hash = load_run(parent)
    after_index, after, after_hash = load_run(rerun)
    selection = after_index["selection"]
    if selection["parent_results_sha256"] != before_hash:
        raise ValueError("Parent does not match rerun provenance")
    if any(before_index[k] != after_index[k] for k in ("mode", "source", "dataset_sha256")):
        raise ValueError("Mode, source or references changed")
    old = {o["id"]: o for o in before["outcomes"]}
    ids = selection["selected_case_ids"]
    if set(ids) != {o["id"] for o in before["outcomes"] if not o.get("checks", {}).get("pass")}:
        raise ValueError("Selection differs from parent failures")
    rows = []
    for outcome in after["outcomes"]:
        previous = old[outcome["id"]]
        if outcome["id"] not in ids or outcome["reference"] != previous["reference"]:
            raise ValueError("Case/reference changed")
        rows.append({"id": outcome["id"], "before": previous, "after": outcome})
    selected_before = [old[row["id"]] for row in rows]
    result = {
        "scope": selection["limitation"],
        "parent_results_sha256": before_hash,
        "rerun_results_sha256": after_hash,
        "rerun_status": after_index["status"],
        "parent_failures_selected": len(ids),
        "cases_paired": len(rows),
        "before": summarize(selected_before, len(ids)),
        "after": after["summary"],
        "recovered_ids": [r["id"] for r in rows if r["after"].get("checks", {}).get("pass")],
        "remaining_flagged_ids": [r["id"] for r in rows if not r["after"].get("checks", {}).get("pass")],
        "unattempted_ids": [identifier for identifier in ids if identifier not in {r["id"] for r in rows}],
        "pairs": rows,
    }
    if pricing_path:
        pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
        result["price_equivalent_before"] = price_equivalent(selected_before, pricing)
        result["price_equivalent_after"] = price_equivalent(after["outcomes"], pricing)
    lines = [
        "# Failed-question rerun: paired hybrid comparison", "",
        result["scope"], "",
        "Unchanged proposed references; strict automatic checks are diagnostics, not semantic quality scores.",
        "No separate paid judge or full semantic review. The other 125 prior successes were not rerun.", "",
        "Before metrics below include only the same collected case IDs as after; "
        "unattempted cases remain in the planned denominator.", "",
        "| Measure (same collected questions) | Before | After |",
        "| --- | --- | --- |",
    ]
    for key in ("collected_cases", "automatic_checks_passed", "status_matches", "operational_errors",
                "request_exceptions", "cases_with_part_errors", "setup_errors"):
        lines.append(f"| {key} | {result['before'][key]} | {result['after'][key]} |")
    if pricing_path:
        lines += ["", "## Usage and indicative cost", ""]
        for label in ("before", "after"):
            cost = result[f"price_equivalent_{label}"]
            lines.append(
                f"{label.title()}: ${cost['captured_calls_undiscounted_equivalent_usd']:.4f} "
                f"public-list equivalent (rates dated {cost['pricing_checked_date']}); "
                "not the configured proxy provider's bill; setup included. "
                "Unknown failed-call usage and cache discounts limit comparability."
            )
        lines.append("")
    lines += ["## Every paired question and answer", "",
              "| ID | Question | Expected | Before | After | New strict checks |",
              "| --- | --- | --- | --- | --- | --- |"]

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    def answer(outcome):
        body = outcome.get("response") or {}
        return f"{body.get('status', 'ERROR')}: " + body.get("answer", json.dumps(outcome.get("error")))

    for row in rows:
        current = row["after"]
        values = [row["id"], current["question"], current["expected_status"],
                  answer(row["before"]), answer(current),
                  "pass" if current.get("checks", {}).get("pass") else "flag"]
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    markdown, data = rerun / "paired-comparison.md", rerun / "paired-comparison.json"
    if markdown.exists() or data.exists():
        raise FileExistsError("Refusing to overwrite a paired comparison")
    write_json(data, result, exclusive=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("--pricing-reference", type=Path)
    cli = parser.parse_args()
    result = build(cli.parent, cli.rerun, cli.pricing_reference)
    print(json.dumps({"recovered": len(result["recovered_ids"]),
                      "still_flagged": len(result["remaining_flagged_ids"]),
                      "unattempted": len(result["unattempted_ids"])}))
