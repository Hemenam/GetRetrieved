"""Offline preparation/reporting for explicit assistant-authored semantic ratings.

Never calls a model, assigns semantic ratings, updates gold, or changes captures.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_chapter_coverage import read_source  # noqa: E402
from report_natural_benchmark import load_run  # noqa: E402

from hrlearnium.eval_tools import digest  # noqa: E402

RUNS = [
    "natural-student-v2-hybrid-20261023-01",
    "natural-student-v2-hybrid-v9-failures-20261024-01",
    "natural-student-v2-hybrid-v10-failures-20261024-08",
]
DIMENSIONS = [
    "intent_status",
    "directness",
    "factual_support",
    "completeness",
    "boundaries",
    "context_grouping",
    "language_presentation",
    "evidence_provenance",
]
VERDICTS = {"pass", "minor", "partial", "fail", "reference_review"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(output):
    if output.exists():
        raise FileExistsError("Choose a new review directory")
    selected, provenance, first = {}, [], None
    for name in RUNS:
        path = ROOT / "evals/results" / name
        index, result, checksum = load_run(path)
        identity = (index["dataset_sha256"], index["source"], index["mode"])
        first = first or identity
        if identity != first or index["status"] != "complete":
            raise ValueError("Mixed source/reference/mode or incomplete run")
        provenance.append(
            {"run": name, "results_sha256": checksum, "index_sha256": digest(path / "index.json")}
        )
        for outcome in result["outcomes"]:
            row = {
                k: outcome.get(k)
                for k in (
                    "id",
                    "question",
                    "track",
                    "reference",
                    "response",
                    "checks",
                    "error",
                    "setup_response",
                    "setup_checks",
                    "setup_error",
                    "latency_ms",
                )
            }
            row.update(
                run=name,
                policy=(outcome.get("evaluation") or {})
                .get("configuration", {})
                .get("policy_version"),
            )
            selected[outcome["id"]] = row
    source, passages = read_source(
        ROOT / "evals/results" / RUNS[-1] / "benchmark.sqlite3",
        first[1]["tenant"],
        first[1]["course"],
    )
    if source != first[1] or len(selected) != 188:
        raise ValueError("Source or case count differs")
    by_id = {p["id"]: p for p in passages}
    for row in selected.values():
        for body in (row["response"], row["setup_response"]):
            if not body:
                continue
            for excerpt in body["excerpts"]:
                p, c = by_id.get(excerpt["id"]), excerpt["citation"]
                if (
                    p is None
                    or p["text"] != excerpt["text"]
                    or any(
                        c[k] != p[k]
                        for k in ("source_start", "source_end", "chapter_number", "section_kind")
                    )
                    or c["source_sha256"] != source["source_sha256"]
                ):
                    raise ValueError("Captured source differs from canonical source")
    output.mkdir(parents=True)
    write(output / "responses.json", [selected[k] for k in sorted(selected)])
    write(output / "source-passages.json", passages)
    write(
        output / "manifest.json",
        {
            "kind": "mixed_version_saved_response_review",
            "selection": "latest capture wins, regardless of pass",
            "runs": provenance,
            "source": source,
            "cases": len(selected),
            "by_policy": dict(Counter(r["policy"] for r in selected.values())),
            "responses_sha256": digest(output / "responses.json"),
            "source_passages_sha256": digest(output / "source-passages.json"),
            "new_provider_calls": 0,
            "reference_approval": "proposed_not_owner_approved",
        },
    )


def verify_inputs(directory):
    manifest = read(directory / "manifest.json")
    for name, key in (
        ("responses.json", "responses_sha256"),
        ("source-passages.json", "source_passages_sha256"),
    ):
        if digest(directory / name) != manifest[key]:
            raise ValueError("Review input changed: " + name)
    for run in manifest["runs"]:
        _, _, checksum = load_run(ROOT / "evals/results" / run["run"])
        if checksum != run["results_sha256"]:
            raise ValueError("Original run changed")
    return manifest


def show(directory, start, end):
    for row in read(directory / "responses.json"):
        number = int(row["id"].split("-")[1])
        if not start <= number <= end:
            continue
        ref, body = row["reference"], row["response"] or {}
        print(
            f"\n{number:03d} [{row['policy']}] expected={ref['expected_status']} actual={body.get('status')} Q: {row['question']}"
        )
        if ref.get("previous_question"):
            print(
                "SETUP Q:",
                ref["previous_question"],
                "\nSETUP A:",
                (row["setup_response"] or {}).get("answer"),
            )
        print("A:", body.get("answer", row.get("error")))
        print(
            "S:",
            " ".join(
                f"[{i}]={e['citation']['chapter_number']}/{e['citation']['section_kind']}"
                for i, e in enumerate(body.get("excerpts", []), 1)
            ),
        )
        if len(ref["parts"]) > 1 or row["track"] in {"answer_presentation", "evidence_boundaries"}:
            print(
                "EXPECTED:",
                " | ".join(
                    f"{i}:{p['status']} " + "; ".join(f["text"] for f in p["facts"])
                    for i, p in enumerate(ref["parts"], 1)
                ),
            )
        if body.get("parts"):
            print(
                "PARTS:",
                [
                    (
                        p["question"],
                        p["status"],
                        [
                            e
                            for s in (p.get("explanation") or {}).get("statements", [])
                            for e in s["citation_ids"]
                        ],
                    )
                    for p in body["parts"]
                ],
            )


def validate_rating(rating, source_keys):
    if len(rating["s"]) != 8 or any(
        type(v) is not int or v not in (0, 1, 2) for v in rating["s"] if v is not None
    ):
        raise ValueError("Invalid dimension scores")
    if rating["verdict"] not in VERDICTS or not rating["note"].strip():
        raise ValueError("Missing manual verdict/rationale")
    if not set(rating["e"]) <= source_keys | {"outline", "policy", "conversation"}:
        raise ValueError("Unknown source key")
    if rating["verdict"] in {"pass", "minor"} and any(
        rating["s"][i] not in (None, 2) for i in (0, 2, 3, 4, 5, 7)
    ):
        raise ValueError("Accepted verdict conflicts with a core-quality deficit")
    if rating["verdict"] == "pass" and 1 in rating["s"]:
        raise ValueError("Clean pass has a partial dimension")


def accepted(rating):
    return rating["verdict"] in {"pass", "minor"}


def additional_metrics(rows):
    """Arithmetic only; every semantic judgment is supplied by the reviewer."""
    multi = [r for r in rows if r["manual_review"].get("parts")]
    followups = [r for r in rows if r["reference"].get("previous_question")]
    eligible = [r for r in followups if accepted(r["manual_review"]["setup"])]
    parts = [p for r in multi for p in r["manual_review"]["parts"]]
    expected_answered = [
        (r, i, p)
        for r in multi
        for i, p in enumerate(r["reference"]["parts"])
        if p["status"] == "answered"
    ]
    return {
        "all_parts_acceptable": {
            "numerator": sum(all(accepted(p) for p in r["manual_review"]["parts"]) for r in multi),
            "denominator": len(multi),
        },
        "part_level": {"counts": dict(Counter(p["verdict"] for p in parts)), "total": len(parts)},
        "supported_answer_part_quality": {
            "numerator": sum(
                accepted(r["manual_review"]["parts"][i]) for r, i, _ in expected_answered
            ),
            "denominator": len(expected_answered),
            "interpretation": "Expected answerable parts delivered acceptably; also reports delivery below. Not a source-recall metric.",
        },
        "supported_answer_part_delivery": {
            "numerator": sum(
                i < len((r["response"] or {}).get("parts", []))
                and r["response"]["parts"][i]["status"] == "answered"
                for r, i, _ in expected_answered
            ),
            "denominator": len(expected_answered),
        },
        "setup_quality": {
            "counts": dict(Counter(r["manual_review"]["setup"]["verdict"] for r in followups)),
            "eligible": len(eligible),
            "total": len(followups),
            "failed_setup_ids": [
                r["id"] for r in followups if not accepted(r["manual_review"]["setup"])
            ],
        },
        "eligible_followup_quality": {
            "numerator": sum(accepted(r["manual_review"]) for r in eligible),
            "denominator": len(eligible),
        },
        "end_to_end_acceptable_including_setup": sum(
            accepted(r["manual_review"])
            and (
                not r["reference"].get("previous_question") or accepted(r["manual_review"]["setup"])
            )
            for r in rows
        ),
    }


def render(directory, prefix=""):
    if prefix and not prefix.replace("-", "").isalnum():
        raise ValueError("Report prefix must contain only letters, digits or hyphens")
    manifest = verify_inputs(directory)
    rows, passages = read(directory / "responses.json"), read(directory / "source-passages.json")
    annotations = [
        json.loads(s)
        for s in (directory / "ratings.jsonl").read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    ratings = {r["id"]: r for r in annotations}
    if (
        len(ratings) != 188
        or len(annotations) != 188
        or set(ratings) != {int(r["id"].split("-")[1]) for r in rows}
    ):
        raise ValueError("Every case needs exactly one explicit manual rating")
    source_keys = {f"{p['chapter_number']}/{p['section_kind']}" for p in passages}
    source_by_key = {f"{p['chapter_number']}/{p['section_kind']}": p for p in passages}
    breakdowns = {k: defaultdict(Counter) for k in ("policy", "track", "family")}
    dimension_totals = {k: Counter() for k in DIMENSIONS}
    multipart, setups, failures, manual = [], [], [], []
    for row in rows:
        n = int(row["id"].split("-")[1])
        r, ref = ratings[n], row["reference"]
        validate_rating(r, source_keys)
        for i, name in enumerate(DIMENSIONS):
            dimension_totals[name]["na" if r["s"][i] is None else str(r["s"][i])] += 1
        for key, val in (
            ("policy", row["policy"]),
            ("track", row["track"]),
            ("family", ref["question_family_id"]),
        ):
            breakdowns[key][val][r["verdict"]] += 1
        if len(ref["parts"]) > 1:
            if len(r.get("parts", [])) != len(ref["parts"]):
                raise ValueError(f"Missing per-part ratings: {n}")
            for p in r["parts"]:
                validate_rating(p, source_keys)
            if accepted(r) and not all(accepted(p) for p in r["parts"]):
                raise ValueError(f"Accepted whole request contains a failed part: {n}")
            multipart.append({"id": row["id"], "verdict": r["verdict"], "parts": r["parts"]})
        if ref.get("previous_question"):
            validate_rating(r["setup"], source_keys)
            setups.append({"id": row["id"], "rating": r["setup"]})
        if r["verdict"] not in {"pass", "minor"}:
            failures.append({"id": row["id"], "verdict": r["verdict"], "reason": r["note"]})
        manual.append(row | {"manual_review": r})
    counts = Counter(r["verdict"] for r in ratings.values())

    def macro(groups):
        return sum((c["pass"] + c["minor"]) / sum(c.values()) for c in groups.values()) / len(
            groups
        )

    summary = {
        "kind": manifest["kind"],
        "scope": "Assistant review of saved mixed-version development responses, not current-model accuracy or independent evaluation",
        "reference_approval": manifest["reference_approval"],
        "reviewer": "implementing_assistant_nonblind",
        "reviewed": len(rows),
        "verdicts": dict(counts),
        "acceptable_including_minor": counts["pass"] + counts["minor"],
        "acceptable_micro_rate": (counts["pass"] + counts["minor"]) / len(rows),
        "track_macro_rate": macro(breakdowns["track"]),
        "family_macro_rate": macro(breakdowns["family"]),
        "by_policy": dict(breakdowns["policy"]),
        "by_track": dict(breakdowns["track"]),
        "by_family": dict(breakdowns["family"]),
        "dimension_counts": dimension_totals,
        "multipart": multipart,
        "setups": setups,
        "flagged": failures,
        "automatic_pass_but_manually_unacceptable": [
            r["id"]
            for r in manual
            if r["checks"].get("pass") and r["manual_review"]["verdict"] not in {"pass", "minor"}
        ],
        "new_provider_calls": 0,
        "ratings_sha256": digest(directory / "ratings.jsonl"),
        "responses_sha256": manifest["responses_sha256"],
        "main_automatic_passes": sum(bool(r["checks"].get("pass")) for r in rows),
        "automatic_flag_but_manually_acceptable": [
            r["id"] for r in manual if not r["checks"].get("pass") and accepted(r["manual_review"])
        ],
        **additional_metrics(manual),
    }
    write(directory / (prefix + "summary.json"), summary)
    write(directory / (prefix + "reviewed-responses.json"), manual)

    def cell(value):
        return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")

    lines = [
        "# Source-grounded review of 188 saved responses",
        "",
        summary["scope"]
        + ". No new API calls. All original captures and proposed references are unchanged.",
        "",
        "Scores: 0 = wrong/missing; 1 = partial; 2 = satisfactory; — = not applicable. Dimension order: I intent/status; D directness; F factual support; C completeness; B boundaries; G context/grouping; L language/presentation; E evidence/provenance.",
        "",
        "Acceptable = pass or minor (presentation/directness issues only). Partial does not count as fully acceptable. Any observed unsupported substantive claim fails, even if its core answer is correct. Missing exact reference anchors alone is not a semantic failure.",
        "",
        f"Verdicts: {dict(counts)}. Acceptable: {summary['acceptable_including_minor']}/188. Track macro: {summary['track_macro_rate']:.1%}; semantic-family macro: {summary['family_macro_rate']:.1%}. Not an estimate of production LMS accuracy.",
        "",
        f"Including successful setup quality, end-to-end acceptable recorded cases: {summary['end_to_end_acceptable_including_setup']}/188. Main-answer and setup quality are reported separately, not silently substituted.",
        "",
        "## Version composition",
        "",
        "Version rows describe disjoint question subsets with different selection histories, not an apples-to-apples architecture ranking.",
        "",
        "| Code policy | Cases | Verdict counts |",
        "| --- | --- | --- |",
    ]
    for policy, cs in breakdowns["policy"].items():
        lines.append(f"| {policy} | {sum(cs.values())} | {dict(cs)} |")
    lines += [
        "",
        "## Per-track results",
        "",
        "| Track | Cases | Acceptable | Verdict counts |",
        "| --- | --- | --- | --- |",
    ]
    for track, cs in breakdowns["track"].items():
        lines.append(f"| {track} | {sum(cs.values())} | {cs['pass'] + cs['minor']} | {dict(cs)} |")
    lines += [
        "",
        "## Dimension score distribution",
        "",
        "| Dimension | 0 | 1 | 2 | N/A | Mean where applicable / 2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, totals in dimension_totals.items():
        applicable = sum(totals[str(i)] for i in range(3))
        mean = sum(i * totals[str(i)] for i in range(3)) / applicable if applicable else 0
        lines.append(
            f"| {name} | {totals['0']} | {totals['1']} | {totals['2']} | {totals['na']} | {mean:.3f} |"
        )
    lines += [
        "",
        "## Every question and answer",
        "",
        "| ID / version | Student question | Captured answer | Expected meaning/status | I D F C B G L E | Verdict and rationale | Review evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in manual:
        r, ref = row["manual_review"], row["reference"]
        expected = " / ".join(
            p["status"] + ": " + "; ".join(f["text"] for f in p["facts"]) for p in ref["parts"]
        )
        source_doc = prefix + "source-evidence.md"
        evidence = "Reviewer: " + ", ".join(
            f"[{k}]({source_doc}#{k.replace('/', '-')})" for k in r["e"]
        )
        actual_map = []
        for i, excerpt in enumerate((row["response"] or {}).get("excerpts", []), 1):
            c = excerpt["citation"]
            key = f"{c['chapter_number']}/{c['section_kind']}"
            actual_map.append(f"[{i}] → [{key}]({source_doc}#{key.replace('/', '-')})")
        evidence = (
            (
                "Bot citations: " + "; ".join(actual_map)
                if actual_map
                else "Bot: no content excerpts"
            )
            + "<br>"
            + evidence
        )
        setup = (
            ("<br>Previous question: " + ref["previous_question"])
            if ref.get("previous_question")
            else ""
        )
        values = [
            row["id"] + " / " + row["policy"],
            row["question"] + setup,
            (row["response"] or {}).get("answer", str(row["error"])),
            expected,
            " ".join("—" if s is None else str(s) for s in r["s"]),
            r["verdict"] + ": " + r["note"],
            evidence,
        ]
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    lines += ["", "## Multipart part-level reviews", ""]
    for item in multipart:
        lines.append("### " + item["id"] + "\n")
        for i, part in enumerate(item["parts"], 1):
            lines.append(
                f"- Part {i}: {part['verdict']}; scores {part['s']}; {part['note']} Sources: {part['e']}."
            )
        lines.append("")
    lines += ["## Follow-up setups", ""]
    for item in setups:
        lines.append(f"- {item['id']}: {item['rating']['verdict']}; {item['rating']['note']}")
    lines += [
        "",
        "## Limits",
        "",
        "This review is nonblind and performed by the implementing assistant. All questions are development cases, correlated paraphrases are not independent samples, and references remain proposed. Accepting a recorded old-version answer does not establish that today's code returns it. Scores reflect the supplied course's claims, not independent medical/scientific validation of those claims. Operational delivery failures remain in the denominator. No current full-188 or three-mode benchmark was run.",
        "",
    ]
    with (directory / (prefix + "evaluation.md")).open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    evidence_lines = [
        "# Canonical course evidence",
        "",
        "Offsets below are canonical character offsets, NOT page numbers. Read-only source snapshot; this file is private course material.",
        "",
    ]
    for k, p in source_by_key.items():
        evidence_lines += [
            f'<a id="{k.replace("/", "-")}"></a>',
            f"## {k}",
            "",
            f"Passage `{p['id']}`, source characters {p['source_start']}–{p['source_end']}; chapter: {p['chapter_title']}.",
            "",
            p["text"],
            "",
        ]
    evidence_lines += [
        '<a id="outline"></a>',
        "## outline",
        "",
        "Document title: " + manifest["source"]["title"],
        "",
    ]
    chapters = sorted({(p["chapter_number"], p["chapter_title"]) for p in passages})
    evidence_lines += [f"{n}. {title}" for n, title in chapters]
    evidence_lines += [
        "",
        '<a id="policy"></a>',
        "## policy",
        "",
        "See the frozen inputs/grading.md and each captured reference. No inventions of personalized plans, missing administrative values or actual company facts. Correct social replies and unresolved-referent clarification require no course citation.",
        "",
        '<a id="conversation"></a>',
        "## conversation",
        "",
        "Actual setup questions and captured setup answers are preserved in responses.json and reviewed separately above.",
        "",
    ]
    with (directory / (prefix + "source-evidence.md")).open("x", encoding="utf-8") as f:
        f.write("\n".join(evidence_lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "show", "render"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=188)
    parser.add_argument(
        "--prefix", default="", help="New export filenames; never overwrite an earlier export"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory)
    elif args.command == "show":
        show(args.directory, args.start, args.end)
    else:
        result = render(args.directory, prefix=args.prefix)
        print(
            json.dumps(
                {k: result[k] for k in ("reviewed", "verdicts", "acceptable_including_minor")},
                ensure_ascii=False,
            )
        )
