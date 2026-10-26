"""Explicit one-query diagnostic repeat; never replaces original benchmark results."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import benchmark as bench
import evaluate
from audit_chapter_coverage import read_source
from recover_benchmark_errors import fixed_config
from report_benchmark_recovery import build

from hrlearnium.config import Settings
from hrlearnium.eval_tools import digest, runtime_summary
from hrlearnium.evaluation import configuration


def select_case(manifest_path, source_run, recovery_run, mode, case_id):
    manifest, profiles = bench.load_manifest(manifest_path)
    groups, _ = build(source_run, recovery_run)
    if len(profiles) != 1 or not bench.plan(manifest, profiles)["execution_ready"]:
        raise ValueError("Expected one approved course profile")
    doc = profiles[0]
    cases = [case for case in doc["cases"] if case["id"] == case_id]
    if len(cases) != 1 or cases[0].get("previous_question"):
        raise ValueError("Select exactly one fresh-conversation case")
    reports = next(iter(groups.values()))
    original = next(o for o in reports[mode]["outcomes"] if o["id"] == case_id)
    if original["question"] != cases[0]["question"]:
        raise ValueError("The approved question differs from the captured question")
    if reports[mode]["dataset_sha256"] != doc["dataset_sha256"]:
        raise ValueError("Approved dataset differs from the captured dataset")
    settings = Settings()
    expected = fixed_config(original["evaluation"]["configuration"])
    if fixed_config(configuration(settings)) != expected:
        raise ValueError("Runtime configuration changed; do not call this an unchanged repeat")
    source, _ = read_source(settings.database_path, doc["tenant"], doc["course"])
    if source["source_corpus_sha256"] != doc["source_corpus_sha256"]:
        raise ValueError("Current source differs from the approved source")
    return manifest, doc, cases[0], original, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--recovery-run", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mode", choices=bench.MODES, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--local-auth", action="store_true")
    parser.add_argument("--subject", default="explicit-single-case-diagnostic")
    parser.add_argument("--timeout", type=float, default=180)
    cli = parser.parse_args()
    cli.delay_seconds = 0
    manifest, doc, case, original, source = select_case(
        cli.manifest, cli.source_run, cli.recovery_run, cli.mode, cli.case_id
    )
    if not cli.execute:
        print(
            json.dumps(
                {
                    "case_id": case["id"],
                    "mode": cli.mode,
                    "planned_queries": 1,
                    "network_requests": 0,
                }
            )
        )
        return
    if cli.output is None:
        parser.error("--execute requires a new --output directory")
    cli.output.mkdir(parents=True, exist_ok=False)
    checkpoint = cli.output / "index.json"
    index = {
        "kind": "explicit_semantic_diagnostic_repeat",
        "authorization": "Owner explicitly requested one repeat after assistant answer/cost review in chat",
        "created_at": datetime.now(UTC).isoformat(),
        "case_id": case["id"],
        "mode": cli.mode,
        "planned_queries": 1,
        "attempted_queries": 0,
        "pending_request": False,
        "original_result_unchanged": True,
        "original_status": original["actual_status"],
        "source_index_sha256": digest(cli.source_run / "index.json"),
        "recovery_index_sha256": digest(cli.recovery_run / "index.json"),
        "dataset_sha256": doc["dataset_sha256"],
        "source_corpus_sha256": source["source_corpus_sha256"],
        "status": "preflight",
        "hidden_warmups": 0,
        "outer_http_retries": 0,
        "limitations": [
            "Diagnostic only; no replacement of the original refusal or score.",
            "One repeat cannot establish a fix or a failure rate.",
            "Existing bounded provider transport retry policy is unchanged.",
        ],
    }
    bench.write_json(checkpoint, index)
    try:
        target_manifest = dict(manifest, modes=[cli.mode])
        bench.preflight(cli, target_manifest, [doc])
        args = bench.request_args(cli, manifest, doc, cli.mode)
        index.update(status="running", pending_request=True, attempted_queries=1)
        bench.write_json(checkpoint, index, exclusive=False)
        outcomes, metadata = evaluate.api_evaluation([case], args)
        report = {
            "kind": "explicit_semantic_diagnostic_repeat",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_sha256": doc["dataset_sha256"],
            "metadata": metadata,
            "outcomes": outcomes,
            "summary": evaluate.api_summary(outcomes),
            "runtime": runtime_summary(outcomes),
        }
        report_path = cli.output / "response.json"
        bench.write_json(report_path, report)
        index.update(
            status="complete" if outcomes[0]["evaluation_status"] == "evaluated" else "stopped",
            pending_request=False,
            report="response.json",
            report_sha256=digest(report_path),
        )
        bench.write_json(checkpoint, index, exclusive=False)
        print(
            json.dumps(
                {
                    "output": str(cli.output),
                    "attempted_queries": 1,
                    "status": outcomes[0].get("actual_status"),
                    "error": outcomes[0].get("error"),
                    "automated_pass": outcomes[0].get("passed_automated_checks"),
                }
            )
        )
    except Exception:
        index["status"] = "stopped"
        bench.write_json(checkpoint, index, exclusive=False)
        raise


if __name__ == "__main__":
    main()
