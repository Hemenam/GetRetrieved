"""Offline same-version repeat report, using explicit semantic judgments, never auto-as-gold."""

import argparse
import json
from collections import Counter
from pathlib import Path

from report_natural_benchmark import load_run

ACCEPTABLE = {"pass", "minor"}
VERDICTS = ACCEPTABLE | {"partial", "fail", "error"}


def collect(directories):
    runs, identity = [], None
    for directory in directories:
        index, result, checksum = load_run(directory)
        if index["status"] != "complete":
            raise ValueError("Repeat must be complete; do not silently exclude unfinished attempts")
        current = (
            index["source"],
            index["configuration"],
            index["dataset_sha256"],
            index["manifest_sha256"],
            index["selected_case_ids"],
            index["mode"],
        )
        if identity is not None and current != identity:
            raise ValueError("Repeats must have identical code/config, source and references")
        identity = current
        rows = result["outcomes"]
        if [r["id"] for r in rows] != index["selected_case_ids"]:
            raise ValueError("Missing or reordered repeat outcomes")
        runs.append({"name": directory.name, "sha256": checksum, "index": index, "rows": rows})
    if len(runs) < 2 or len({r["name"] for r in runs}) != len(runs):
        raise ValueError("At least two distinct runs are required")
    return runs


def validate_ratings(runs, ratings):
    if set(ratings) != {r["name"] for r in runs}:
        raise ValueError("Rate every repeat explicitly")
    for run in runs:
        labels = ratings[run["name"]]
        if set(labels) != {row["id"] for row in run["rows"]}:
            raise ValueError("Rate every captured response explicitly, including errors")
        for row in run["rows"]:
            label = labels[row["id"]]
            if label.get("verdict") not in VERDICTS or not label.get("note", "").strip():
                raise ValueError("Every rating needs a verdict and source-grounded review note")
            body = row.get("response") or {}
            operational = (
                bool(row.get("error"))
                or body.get("status") == "error"
                or any(p["status"] == "error" for p in body.get("parts", []))
            )
            if operational and label["verdict"] != "error":
                raise ValueError(
                    "Operational errors cannot be rated as delivered acceptable answers"
                )


def metrics(runs, ratings):
    validate_ratings(runs, ratings)
    ids = [row["id"] for row in runs[0]["rows"]]
    counts = Counter(v["verdict"] for labels in ratings.values() for v in labels.values())
    stable = sum(all(ratings[run["name"]][i]["verdict"] in ACCEPTABLE for run in runs) for i in ids)
    return {
        "cases": len(ids),
        "repeats": len(runs),
        "case_executions": len(ids) * len(runs),
        "semantic_verdicts": dict(counts),
        "acceptable_executions": sum(counts[v] for v in ACCEPTABLE),
        "acceptable_every_repeat": stable,
        "first_repeat_acceptable": sum(
            ratings[runs[0]["name"]][i]["verdict"] in ACCEPTABLE for i in ids
        ),
        "automatic_passes": sum(
            bool(row.get("checks", {}).get("pass")) for run in runs for row in run["rows"]
        ),
        "query_turns": sum(run["index"]["attempted_query_turns"] for run in runs),
        "results_sha256": {run["name"]: run["sha256"] for run in runs},
        "limitation": "Nonblind assistant semantic review of a targeted development subset; not full-suite or unseen-course accuracy. Concurrent collection is not a single-user latency benchmark.",
    }


def render(runs, ratings):
    summary = metrics(runs, ratings)
    lines = [
        "# V11 targeted repeat evaluation",
        "",
        summary["limitation"],
        "",
        "References, course, code/config and original questions are identical across repeats. No best-answer selection.",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "Verdicts: pass = clean; minor = acceptable with presentation issues; partial = missing required meaning; fail = unsupported/altered content or citation support; error = delivery failure.",
        "",
        "Setup quality is discussed in the notes for follow-up cases; main-answer metrics must not be mistaken for guaranteed conversation quality.",
        "",
    ]
    for n, row in enumerate(runs[0]["rows"]):
        lines.extend(["## " + row["id"], "", row["question"], ""])
        for run in runs:
            current = run["rows"][n]
            label = ratings[run["name"]][row["id"]]
            lines.extend(
                ["### " + run["name"], "", "**" + label["verdict"] + "** — " + label["note"], ""]
            )
            if current.get("setup_response"):
                lines.extend(["Setup answer:", "", current["setup_response"]["answer"], ""])
            lines.extend(
                [
                    "Actual answer:",
                    "",
                    (current.get("response") or {}).get("answer", "No response delivered."),
                    "",
                ]
            )
            for i, source in enumerate((current.get("response") or {}).get("excerpts", []), 1):
                citation = source["citation"]
                lines.append(
                    f"- [{i}] Chapter {citation['chapter_number']} / {citation['section_kind']} / `{source['id']}`"
                )
            lines.append("")
    return summary, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = collect(args.directories)
    ratings = json.loads(args.ratings.read_text(encoding="utf-8"))
    summary, markdown = render(runs, ratings)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "answers-and-review.md").write_text(markdown, encoding="utf-8")
    (args.output / "ratings.json").write_bytes(args.ratings.read_bytes())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
