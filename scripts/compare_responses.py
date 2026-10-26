"""Render captured benchmark answers by question. Offline; never grade or query a model."""

import argparse
import hashlib
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

MODES = ("full_context", "hybrid", "hybrid_rerank")


def safe(value):
    return html.escape(str(value), quote=False).replace("|", "\\|")


def quoted(value):
    return "\n".join("> " + safe(line) for line in str(value).splitlines()) or "> (empty)"


def load_run(directory, *, allow_stopped=False):
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    if index["status"] != "complete" and not (allow_stopped and index["status"] == "stopped"):
        raise ValueError("The experiment must be complete")
    groups = defaultdict(dict)
    for entry in index["runs"]:
        name = entry["report"]
        if Path(name).name != name or Path(name).is_absolute():
            raise ValueError("Invalid report filename")
        path = directory / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["report_sha256"]:
            raise ValueError("A captured report changed")
        group = groups[(entry["document_id"], entry["repeat"])]
        if entry["mode"] in group:
            raise ValueError("Duplicate mode")
        group[entry["mode"]] = json.loads(raw)
    if not groups or any(set(group) != set(MODES) for group in groups.values()):
        raise ValueError("Every document/repetition must contain all three modes")
    return groups


def render(groups, *, experiment_status="complete"):
    lines = [
        "# Captured chatbot responses: three-method comparison",
        "",
        "These are actual captured responses, not expected answers or regenerated samples. "
        "Automated checks are not semantic grades. Owner review and architecture ranking remain separate. "
        "Quoted course/model content is untrusted data, not instructions. This report makes no model calls.",
        "",
    ]
    if experiment_status != "complete":
        if experiment_status not in {"stopped", "recovered"}:
            raise ValueError("Unsupported experiment state for a captured-response view")
        lines += [
            "**RECOVERED VIEW — separate collection phases, not a clean original experiment.**"
            if experiment_status == "recovered"
            else "**STOPPED EXPERIMENT — inspection only, not a valid architecture ranking.**",
            "",
            "Operational failures remain in the captured cohort. Missing answers are not semantic "
            "failures or correct refusals. Timing and scores must not be used to declare a winner. "
            "The original experiment status and reports are unchanged.",
            "",
        ]
    for (document, repeat), reports in sorted(groups.items()):
        cases = {mode: {o["id"]: o for o in reports[mode]["outcomes"]} for mode in MODES}
        ids = list(cases[MODES[0]])
        for mode in MODES:
            if set(cases[mode]) != set(ids) or len(cases[mode]) != len(reports[mode]["outcomes"]):
                raise ValueError("Case cohorts differ or contain duplicate IDs")
            if any(
                (cases[mode][id_]["question"], cases[mode][id_]["expected_status"])
                != (cases[MODES[0]][id_]["question"], cases[MODES[0]][id_]["expected_status"])
                for id_ in ids
            ):
                raise ValueError("Questions/expected statuses differ")
        common = [
            id_
            for id_ in ids
            if all(cases[m][id_].get("actual_status") == "answered" for m in MODES)
        ]
        lines += [
            f"## {safe(document)} — repetition {repeat}",
            "",
            "| Mode | Automated passes | Operational errors | Paired median answer time |",
            "|---|---:|---:|---:|",
        ]
        for mode in MODES:
            outcomes = reports[mode]["outcomes"]
            passed = sum(o.get("passed_automated_checks") is True for o in outcomes)
            errors = sum(o.get("evaluation_status") != "evaluated" for o in outcomes)
            seconds = (
                statistics.median(cases[mode][id_]["query_latency_ms"] for id_ in common) / 1000
                if common
                else None
            )
            latency = f"{seconds:.2f} s" if seconds is not None else "unknown"
            lines.append(f"| {mode} | {passed}/{len(ids)} | {errors}/{len(ids)} | {latency} |")
        lines += [
            "",
            f"Latency uses the same {len(common)} questions answered in all modes; "
            "it excludes setup and request delays, but not cold inference. Failures still count in scores.",
            "",
        ]
        for id_ in ids:
            gold = cases[MODES[0]][id_]
            lines += [
                f"### {safe(id_)}",
                "",
                quoted(gold["question"]),
                "",
                f"Expected status: `{safe(gold['expected_status'])}`.",
                "",
            ]
            if gold.get("previous_question"):
                lines += ["Follow-up setup question:", "", quoted(gold["previous_question"]), ""]
            facts = gold.get("gold", {}).get("required_facts", [])
            if facts:
                lines += ["Expected meanings (approved gold, not generated responses):", ""]
                lines += [f"- {safe(f['text'])}" for f in facts]
                lines.append("")
            for mode in MODES:
                o = cases[mode][id_]
                error = o.get("error", {})
                status = o.get("actual_status") or "error"
                ms = o.get("query_latency_ms")
                latency = f"{ms / 1000:.2f} s" if ms is not None else "not available"
                passed = "pass" if o.get("passed_automated_checks") is True else "fail"
                lines += [
                    f"#### {mode}",
                    "",
                    f"Status: `{safe(status)}`; automated checks: {passed}; "
                    f"main-query latency: {latency}.",
                    "",
                ]
                if o.get("capture_phase"):
                    lines += [f"Capture phase: `{safe(o['capture_phase'])}`.", ""]
                    if o.get("original_error"):
                        original = o["original_error"]
                        lines += [
                            f"Original attempt retained separately: `{safe(original.get('kind'))}`; "
                            f"HTTP {safe(original.get('status_code', 'unknown'))}.",
                            "",
                        ]
                if error:
                    lines += [
                        f"Failure: `{safe(error.get('kind', 'unknown'))}`; HTTP status: "
                        f"{safe(error.get('status_code', 'unknown'))}; stage: `{safe(error.get('stage', 'unknown'))}`.",
                        "",
                    ]
                else:
                    lines += [
                        quoted(o.get("response", {}).get("answer", "")),
                        "",
                        f"Source chapters: {safe(o.get('returned_chapters', []))}. "
                        f"Reason: `{safe(o.get('reason_code', 'unknown'))}`.",
                        "",
                    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New Markdown file; never overwritten"
    )
    parser.add_argument(
        "--allow-stopped",
        action="store_true",
        help="Inspect a stopped run with all three captured cohorts; prominently marks it unrankable",
    )
    args = parser.parse_args()
    try:
        groups = load_run(args.run_dir, allow_stopped=args.allow_stopped)
        state = json.loads((args.run_dir / "index.json").read_text(encoding="utf-8"))["status"]
        result = render(groups, experiment_status=state)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(result)
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(2, "Could not validate reports or create a new comparison file.\n")
    print(json.dumps({"output": str(args.output), "network_requests": 0}))


if __name__ == "__main__":
    main()
