"""Export saved responses and existing assistant judgments to Markdown, offline.

This formats an existing review; it does not call a model, create new judgments,
approve answers, or replace benchmark captures with diagnostic repeats.
"""

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

from report_benchmark_recovery import build

MODES = ("full_context", "hybrid", "hybrid_rerank")
LABELS = {
    "full_context": "Full context",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid + reranker",
}
SCORES = {"G": 5, "P": 4, "R": 2, "F": 0}
SECTIONS = {
    "introduction": "مقدمه",
    "explanation": "توضیح اصلی",
    "example": "مثال",
    "exercise": "تمرین",
}
CASE_ROW = re.compile(r"^\| `(\d{2}-(?:introduction|explanation|example|exercise))` \|")


def cell(value):
    """Keep complete Unicode text; escape HTML/table syntax and preserve line breaks."""
    return html.escape(str(value), quote=False).replace("|", "&#124;").replace("\n", "<br>")


def load_judgments(path):
    judgments = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CASE_ROW.match(line)
        if not match:
            continue
        fields = [part.strip() for part in line.split("|")[1:-1]]
        case_id = "coverage-" + match[1]
        if len(fields) != 6 or case_id in judgments:
            raise ValueError("Duplicate or malformed assistant-review row")
        grades = dict(zip(MODES, fields[2:5], strict=True))
        if any(grade not in SCORES for grade in grades.values()):
            raise ValueError("An assistant-review grade has no explicit score mapping")
        judgments[case_id] = {"grades": grades, "note": fields[5]}
    if not judgments:
        raise ValueError("No assistant judgments found")
    return judgments


def render(reports, judgments, *, created_date):
    lookup = {mode: {o["id"]: o for o in reports[mode]["outcomes"]} for mode in MODES}
    ids = [o["id"] for o in reports["full_context"]["outcomes"]]
    if len(ids) != len(set(ids)) or set(ids) != set(judgments):
        raise ValueError("Reviewed cases and captured cases must match exactly")
    if any(set(lookup[mode]) != set(ids) for mode in MODES):
        raise ValueError("All three modes must have the same questions")
    lines = [
        "# Course chatbot: all questions, three answers and assistant scores",
        "",
        f"Report assembled: {created_date} (Asia/Tehran). Assistant judgments: 2026-10-19.",
        "",
        f"Contains {len(ids)} questions and {len(ids) * 3} complete captured answers. "
        "Persian wording is preserved; line breaks are represented by HTML `<br>` inside tables. "
        "Answers are not regenerated or shortened. Citation markers refer to the original numbered "
        "excerpts in each captured response, not to Markdown footnotes.",
        "",
        "## Scoring rubric",
        "",
        "These are **my assistant judgments**, not automated-check scores or human/instructor "
        "approval. Numeric scores translate the existing G/P/R/F review into four explicit bands "
        "on a 0–5 scale; no new model-based judging was run. Scores are ordinal, not accuracy percentages.",
        "",
        "| Score | Prior grade | Meaning |",
        "|---:|:---:|---|",
        "| 5/5 | G | Correct, complete and adequately supported; no material issue found in the review. |",
        "| 4/5 | P | Correct/supportable, but wording, repetition or question focus should improve. |",
        "| 2/5 | R | Revision needed for an added claim, contextual linkage or citation alignment. The core answer may still be correct. |",
        "| 0/5 | F | Failed to answer an answerable question (false refusal in this benchmark). |",
        "",
        "Only these four bands are used. A 5/5 means no issue found in this review, not guaranteed "
        "truth or production reliability. Different issue types can share a band; read each case note.",
        "",
        "## Score distribution",
        "",
        "| Mode | 5/5 | 4/5 | 2/5 | 0/5 | Answers reviewed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        counts = Counter(judgments[id_]["grades"][mode] for id_ in ids)
        lines.append(
            f"| {LABELS[mode]} | {counts['G']} | {counts['P']} | {counts['R']} | {counts['F']} | {len(ids)} |"
        )
    lines += [
        "",
        "The data combines original answers and explicitly authorized operational recovery. "
        "The capture phase is labeled for every answer; this is not a clean single experiment. "
        "The original full-context false refusal is retained. The later successful diagnostic "
        "repeat is shown separately at the end and excluded from the distribution above.",
        "",
        "[Detailed assistant evaluation and cost comparison](assistant-review.md) · "
        "[Original response comparison](responses.md) · [Collection provenance](summary.json)",
        "",
    ]
    chapter = None
    for number, case_id in enumerate(ids, 1):
        original = lookup["full_context"][case_id]
        match = re.fullmatch(r"coverage-(\d{2})-(.+)", case_id)
        if not match or match[2] not in SECTIONS:
            raise ValueError("Unexpected chapter case ID")
        current_chapter = int(match[1])
        if current_chapter != chapter:
            chapter = current_chapter
            lines += [f"## Chapter {chapter}", ""]
        lines += [
            f"### Question {number} — {SECTIONS[match[2]]}",
            "",
            f"Case: `{case_id}`",
            "",
            "| Question |",
            "|---|",
            f"| {cell(original['question'])} |",
            "",
            "Approved expected meaning(s), included for comparison:",
            "",
        ]
        lines += [f"- {cell(fact['text'])}" for fact in original["gold"]["required_facts"]]
        lines += [
            "",
            "| Mode | Complete captured answer | My score | Capture phase |",
            "|---|---|:---:|---|",
        ]
        for mode in MODES:
            outcome = lookup[mode][case_id]
            if outcome["question"] != original["question"] or outcome.get("error"):
                raise ValueError("Mismatched question or unresolved error in answer table")
            grade = judgments[case_id]["grades"][mode]
            answer = outcome["response"]["answer"]
            lines.append(
                f"| {LABELS[mode]} | {cell(answer)} | {SCORES[grade]}/5 ({grade}) | {outcome['capture_phase']} |"
            )
        lines += ["", "My comparison: " + judgments[case_id]["note"], ""]
    return "\n".join(lines) + "\n"


def retry_appendix(directory):
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    path = directory / "response.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != index["report_sha256"]:
        raise ValueError("The diagnostic response hash differs")
    outcome = json.loads(path.read_text(encoding="utf-8"))["outcomes"][0]
    if index["case_id"] != "coverage-07-explanation" or index["mode"] != "full_context":
        raise ValueError("Unexpected diagnostic repeat")
    if outcome["id"] != index["case_id"] or outcome["actual_status"] != "answered":
        raise ValueError(
            "Diagnostic response differs from the separately reviewed successful retry"
        )
    return "\n".join(
        [
            "",
            "## Appendix: separately requested full-context retry",
            "",
            "This is an additional diagnostic answer, not one of the 132 comparison answers. "
            "Its original false refusal remains 0/5 in Question 26 above. No pipeline fix was made.",
            "",
            "| Question | Mode | Complete diagnostic answer | My score |",
            "|---|---|---|:---:|",
            f"| {cell(outcome['question'])} | Full context, separate retry | {cell(outcome['response']['answer'])} | 5/5 (G) |",
            "",
            "The retry answered correctly with chapter-seven support, but also returned an unused "
            "irrelevant chapter-three excerpt. Its answer score does not certify evidence-selection "
            "precision or establish that the original failure is fixed.",
            "",
            "[Detailed retry assessment and raw capture](../full-context-diagnostic-20261019-185627/review.md)",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--recovery-run", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--diagnostic-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    groups, _ = build(cli.source_run, cli.recovery_run)
    judgments = load_judgments(cli.review)
    result = render(next(iter(groups.values())), judgments, created_date="2026-10-20")
    result += retry_appendix(cli.diagnostic_run)
    with cli.output.open("x", encoding="utf-8") as stream:
        stream.write(result)
    print(
        json.dumps(
            {
                "output": str(cli.output),
                "questions": len(judgments),
                "comparison_answers": len(judgments) * 3,
                "diagnostic_answers": 1,
                "network_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
