"""Offline inspection of original captures plus explicitly authorized error recovery.

Does not modify either phase, grade responses, infer a winner or call a model.
Every reused response is labeled by collection phase; original errors stay saved.
"""

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import compare_responses as comparison  # noqa: E402
from recover_benchmark_errors import fixed_config  # noqa: E402

from hrlearnium.eval_tools import digest, runtime_summary  # noqa: E402


def build(source_run: Path, recovery_run: Path):
    groups = comparison.load_run(source_run, allow_stopped=True)
    index = json.loads((recovery_run / "index.json").read_text(encoding="utf-8"))
    if index.get("kind") != "explicit_operational_recovery":
        raise ValueError("Expected an explicit recovery checkpoint")
    if index["source_index_sha256"] != digest(source_run / "index.json"):
        raise ValueError("Original experiment index changed")
    if len(groups) != 1 or index.get("pending_request"):
        raise ValueError("Expected one cohort and no unresolved in-flight request")
    cohort, reports = next(iter(groups.items()))
    view = copy.deepcopy(reports)
    lookup = {mode: {o["id"]: o for o in r["outcomes"]} for mode, r in view.items()}
    original_outcomes = [o for r in reports.values() for o in r["outcomes"]]
    if any(r["dataset_sha256"] != index["dataset_sha256"] for r in reports.values()):
        raise ValueError("Recovery dataset hash differs")
    recovery_outcomes, replacements, seen = [], [], set()
    for entry in index["captures"]:
        name = entry["report"]
        if Path(name).name != name or Path(name).is_absolute():
            raise ValueError("Invalid recovery report path")
        path = recovery_run / name
        if digest(path) != entry["report_sha256"]:
            raise ValueError("A recovery capture changed")
        record = json.loads(path.read_text(encoding="utf-8"))
        if len(record["outcomes"]) != 1 or record["dataset_sha256"] != index["dataset_sha256"]:
            raise ValueError("Recovery report must contain one outcome from the same dataset")
        outcome = record["outcomes"][0]
        key = (entry["mode"], entry["case_id"])
        if (
            key in seen
            or key[0] not in lookup
            or key[1] not in lookup[key[0]]
            or outcome["id"] != key[1]
        ):
            raise ValueError("Duplicate, unknown or mismatched recovery target")
        seen.add(key)
        old = lookup[key[0]][key[1]]
        if not old.get("error") or old["evaluation_status"] == "evaluated":
            raise ValueError("A semantic result must not be replaced by an error recovery")
        for field in ("question", "expected_status", "expected_chapters", "gold"):
            if outcome.get(field) != old.get(field):
                raise ValueError("Recovery question or gold differs from original capture")
        if outcome.get("retrieval_mode") and outcome["retrieval_mode"] != key[0]:
            raise ValueError("Recovered mode mismatch")
        recovery_outcomes.append(outcome)
        if not outcome.get("error") and outcome["evaluation_status"] == "evaluated":
            used = copy.deepcopy(outcome)
            used.update(capture_phase="recovery", original_error=old["error"])
            # Replace only in this display copy, never either immutable source phase.
            position = next(i for i, o in enumerate(view[key[0]]["outcomes"]) if o["id"] == key[1])
            view[key[0]]["outcomes"][position] = used
            replacements.append({"mode": key[0], "case_id": key[1], "report": name})
    for report in view.values():
        for o in report["outcomes"]:
            o.setdefault("capture_phase", "original")
    traces = [o["evaluation"] for o in original_outcomes + recovery_outcomes if o.get("evaluation")]
    configs = {json.dumps(fixed_config(t["configuration"]), sort_keys=True) for t in traces}
    sources = {t.get("source_corpus_sha256") for t in traces}
    if len(configs) != 1 or sources != {index["source_corpus_sha256"]}:
        raise ValueError("Runtime configuration or source differs between collection phases")
    for mode in comparison.MODES:
        mode_configs = {
            json.dumps(o["evaluation"]["configuration"], sort_keys=True)
            for o in original_outcomes + recovery_outcomes
            if o.get("retrieval_mode") == mode and o.get("evaluation")
        }
        if len(mode_configs) != 1:
            raise ValueError("A mode's retrieval/reranker configuration changed between captures")
    original_errors = sum(bool(o.get("error")) for o in original_outcomes)
    if index["status"] == "complete" and len(seen) != original_errors:
        raise ValueError("Completed recovery is missing original operational failures")
    paired = [
        o["id"]
        for o in view["full_context"]["outcomes"]
        if all(
            next(x for x in view[m]["outcomes"] if x["id"] == o["id"]).get("actual_status")
            == "answered"
            for m in comparison.MODES
        )
    ]
    metrics, by_chapter, by_section = [], [], []
    for mode in comparison.MODES:
        outcomes = view[mode]["outcomes"]
        original = reports[mode]["outcomes"]
        metrics.append(
            {
                "mode": mode,
                "cases": len(outcomes),
                "original_operational_errors": sum(bool(o.get("error")) for o in original),
                "recovered_responses": sum(o["capture_phase"] == "recovery" for o in outcomes),
                "view_automated_passes": sum(
                    o.get("passed_automated_checks") is True for o in outcomes
                ),
                "view_false_refusals": sum(
                    o.get("actual_status") == "refused"
                    for o in outcomes
                    if o["expected_status"] == "answered"
                ),
                "view_unresolved_errors": sum(bool(o.get("error")) for o in outcomes),
                "paired_answer_median_ms": statistics.median(
                    [o["query_latency_ms"] for o in outcomes if o["id"] in paired]
                )
                if paired
                else None,
            }
        )
        for chapter in range(1, 12):
            rows = [o for o in outcomes if o["expected_chapters"] == [chapter]]
            by_chapter.append(
                {
                    "mode": mode,
                    "chapter": chapter,
                    "cases": len(rows),
                    "automated_passes": sum(o.get("passed_automated_checks") is True for o in rows),
                    "unresolved_errors": sum(bool(o.get("error")) for o in rows),
                }
            )
        for section in ("introduction", "explanation", "example", "exercise"):
            rows = [o for o in outcomes if o["id"].endswith("-" + section)]
            by_section.append(
                {
                    "mode": mode,
                    "section": section,
                    "cases": len(rows),
                    "automated_passes": sum(o.get("passed_automated_checks") is True for o in rows),
                    "unresolved_errors": sum(bool(o.get("error")) for o in rows),
                }
            )
    summary = {
        "kind": "recovered_inspection_view",
        "clean_single_experiment": False,
        "recovery_status": index["status"],
        "runtime_configuration_and_source_match": True,
        "original_operational_errors_preserved": original_errors,
        "additional_attempted_requests": index["attempted_requests"],
        "recovered_responses": len(replacements),
        "unresolved_original_errors": original_errors - len(replacements),
        "semantic_review_complete": False,
        "winner": None,
        "rank": None,
        "paired_answer_questions": len(paired),
        "metrics": metrics,
        "by_chapter": by_chapter,
        "by_section": by_section,
        "all_phases_reported_runtime": runtime_summary(original_outcomes + recovery_outcomes),
        "replacement_provenance": replacements,
        "limitations": index["limitations"]
        + [
            "No human response grading is inferred. Original billing failures are not erased.",
            "Timing spans two phases and includes any captured environment interruptions and cold loading.",
        ],
    }
    return {cohort: view}, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--recovery-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    groups, summary = build(cli.source_run, cli.recovery_run)
    cli.output.mkdir(parents=True, exist_ok=False)
    with (cli.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (cli.output / "responses.md").open("x", encoding="utf-8") as stream:
        stream.write(comparison.render(groups, experiment_status="recovered"))
    print(
        json.dumps(
            {
                "output": str(cli.output),
                "network_requests_made": 0,
                "recovered_responses": summary["recovered_responses"],
            }
        )
    )


if __name__ == "__main__":
    main()
