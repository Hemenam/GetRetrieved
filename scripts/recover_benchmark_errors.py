"""Explicit, fail-fast recovery of operationally failed benchmark requests.

No semantic retries, hidden warmups, in-place report replacement or automatic
resume. Saves every request separately so another interruption loses no completed
capture. A recovery is a second collection phase, never a clean original cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import benchmark as bench  # noqa: E402
import evaluate  # noqa: E402
from audit_chapter_coverage import read_source  # noqa: E402

from hrlearnium.config import Settings  # noqa: E402
from hrlearnium.eval_tools import (  # noqa: E402
    digest,
    latency_summary,
    review_template,
    runtime_summary,
)
from hrlearnium.evaluation import configuration  # noqa: E402


def fixed_config(config):
    return {
        k: v
        for k, v in config.items()
        if k not in {"retrieval_mode", "reranker", "rerank_pool_limit"}
    }


def recovery_plan(run_dir: Path, manifest_path: Path, max_turns: int):
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    manifest, profiles = bench.load_manifest(manifest_path)
    if index["status"] != "stopped" or manifest["repetitions"] != 1 or len(profiles) != 1:
        raise ValueError("Recovery requires a stopped single-document, one-repetition experiment")
    if digest(manifest_path) != index["manifest_sha256"]:
        raise ValueError("The approved manifest differs from the original run")
    doc = profiles[0]
    if not bench.plan(manifest, profiles)["execution_ready"]:
        raise ValueError("The selected reference answers are not approved")
    cases = {c["id"]: c for c in doc["cases"]}
    jobs, configs, modes = [], [], set()
    for entry in index["runs"]:
        name = entry["report"]
        if Path(name).name != name or Path(name).is_absolute():
            raise ValueError("Invalid source report path")
        path = run_dir / name
        if digest(path) != entry["report_sha256"]:
            raise ValueError("An original capture changed")
        report = json.loads(path.read_text(encoding="utf-8"))
        mode = entry["mode"]
        if mode in modes or mode not in manifest["modes"]:
            raise ValueError("Duplicate or unknown source mode")
        modes.add(mode)
        if report["dataset_sha256"] != doc["dataset_sha256"]:
            raise ValueError("The approved dataset differs from the original run")
        if len(report["outcomes"]) != len(cases) or {o["id"] for o in report["outcomes"]} != set(
            cases
        ):
            raise ValueError("The source report has a different case cohort")
        for outcome in report["outcomes"]:
            case = cases[outcome["id"]]
            if outcome["question"] != case["question"] or case.get("previous_question"):
                raise ValueError("Changed questions or follow-up cases are not supported here")
            trace = outcome.get("evaluation")
            if trace:
                configs.append(fixed_config(trace["configuration"]))
                if trace.get("source_corpus_sha256") != doc["source_corpus_sha256"]:
                    raise ValueError("Original source fingerprint mismatch")
            # Never retry a refusal, semantic failure or otherwise evaluated response.
            if outcome["evaluation_status"] == "evaluated" and not outcome.get("error"):
                continue
            if not outcome.get("error"):
                raise ValueError("A non-evaluated case has no operational-error record")
            jobs.append(
                {
                    "mode": mode,
                    "case": case,
                    "original_report": name,
                    "original_report_sha256": entry["report_sha256"],
                    "original_error": outcome["error"],
                }
            )
    if modes != set(manifest["modes"]) or not configs or any(c != configs[0] for c in configs):
        raise ValueError("Original captures lack consistent runtime configuration")
    if not jobs or len(jobs) > max_turns:
        raise ValueError("No failed requests, or recovery would exceed the explicit request cap")
    return manifest, doc, jobs, configs[0]


def execute(cli, manifest, doc, jobs, expected_config):
    settings = Settings()
    if fixed_config(configuration(settings)) != expected_config:
        raise ValueError(
            "Model, prompts, source limits or application code changed; do not merge phases"
        )
    source, _ = read_source(settings.database_path, doc["tenant"], doc["course"])
    if source["source_corpus_sha256"] != doc["source_corpus_sha256"]:
        raise ValueError("Current source differs from the approved source")
    cli.output.mkdir(parents=True, exist_ok=False)
    (cli.output / "inputs").mkdir()
    bench.write_json(cli.output / "inputs/manifest.json", manifest)
    (cli.output / "inputs/cases.jsonl").write_bytes(doc["dataset_path"].read_bytes())
    checkpoint = cli.output / "index.json"
    index = {
        "format_version": 1,
        "kind": "explicit_operational_recovery",
        "authorization": "User continued after pause to charge API on 2026-10-19",
        "created_at": datetime.now(UTC).isoformat(),
        "source_run": str(cli.source_run.resolve()),
        "source_index_sha256": digest(cli.source_run / "index.json"),
        "manifest_sha256": digest(cli.manifest),
        "dataset_sha256": doc["dataset_sha256"],
        "source_corpus_sha256": source["source_corpus_sha256"],
        "planned_requests": len(jobs),
        "max_query_turns": cli.max_query_turns,
        "attempted_requests": 0,
        "hidden_warmup_requests": 0,
        "automatic_http_retries": 0,
        "status": "preflight",
        "captures": [],
        "pending_request": None,
        "limitations": [
            "Separate collection phase after provider billing interruption, not an original clean run.",
            "Original failures are preserved; no architecture winner or human grading is inferred.",
            "Request cap does not cap provider calls or USD spend; usage on failures can be missing.",
        ],
    }
    bench.write_json(checkpoint, index)
    try:
        index["prerequisites"] = bench.preflight(cli, manifest, [doc])
        index["status"] = "running"
        bench.write_json(checkpoint, index, exclusive=False)
        for job in jobs:
            if index["attempted_requests"]:
                time.sleep(cli.delay_seconds)
            if digest(cli.source_run / "index.json") != index["source_index_sha256"]:
                raise ValueError("Original run index changed during recovery")
            if (
                digest(cli.manifest) != index["manifest_sha256"]
                or digest(doc["dataset_path"]) != index["dataset_sha256"]
            ):
                raise ValueError("Approved inputs changed during recovery")
            case = job["case"]
            index["pending_request"] = {"mode": job["mode"], "case_id": case["id"]}
            index["attempted_requests"] += 1
            bench.write_json(checkpoint, index, exclusive=False)
            args = bench.request_args(cli, manifest, doc, job["mode"])
            outcomes, metadata = evaluate.api_evaluation([case], args)
            report = {
                "kind": "operational_retry_capture",
                "created_at": datetime.now(UTC).isoformat(),
                "dataset_sha256": doc["dataset_sha256"],
                "metadata": metadata,
                "original_report": job["original_report"],
                "original_report_sha256": job["original_report_sha256"],
                "original_error": job["original_error"],
                "summary": evaluate.api_summary(outcomes),
                "runtime": runtime_summary(outcomes),
                "latency_ms": latency_summary(
                    [o["latency_ms"] for o in outcomes if "latency_ms" in o]
                ),
                "outcomes": outcomes,
            }
            name = f"{job['mode']}--{case['id']}"
            path = cli.output / f"{name}.json"
            bench.write_json(path, report)
            bench.write_json(cli.output / f"{name}.review.json", review_template(path))
            index["captures"].append(
                {
                    "mode": job["mode"],
                    "case_id": case["id"],
                    "report": path.name,
                    "report_sha256": digest(path),
                }
            )
            index["pending_request"] = None
            bench.write_json(checkpoint, index, exclusive=False)
            outcome = outcomes[0]
            print(
                json.dumps(
                    {
                        "completed": len(index["captures"]),
                        "planned": len(jobs),
                        "mode": job["mode"],
                        "case_id": case["id"],
                        "status": outcome.get("actual_status"),
                        "error": outcome.get("error"),
                    }
                ),
                flush=True,
            )
            if outcome.get("error") or outcome["evaluation_status"] != "evaluated":
                index["status"] = "stopped_on_operational_error"
                break
            trace = outcome.get("evaluation") or {}
            if (
                fixed_config(trace.get("configuration", {})) != expected_config
                or trace.get("source_corpus_sha256") != source["source_corpus_sha256"]
            ):
                raise ValueError(
                    "Captured configuration or source drifted; no further requests sent"
                )
        else:
            index["status"] = "complete"
    except KeyboardInterrupt:
        index["status"] = "interrupted"
        raise
    except (OSError, ValueError, evaluate.EvaluationError):
        index["status"] = "stopped"
        raise
    finally:
        bench.write_json(checkpoint, index, exclusive=False)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-query-turns", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--local-auth", action="store_true")
    parser.add_argument("--subject", default="benchmark-operational-recovery")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--delay-seconds", type=float, default=2)
    cli = parser.parse_args()
    try:
        if (
            cli.max_query_turns < 1
            or not math.isfinite(cli.timeout)
            or cli.timeout <= 0
            or not 0 <= cli.delay_seconds <= 60
        ):
            raise ValueError("Invalid request cap, timeout or delay")
        manifest, doc, jobs, config = recovery_plan(
            cli.source_run, cli.manifest, cli.max_query_turns
        )
        if not cli.execute:
            print(
                json.dumps(
                    {
                        "requests": len(jobs),
                        "network_requests_made": 0,
                        "by_mode": {
                            m: sum(j["mode"] == m for j in jobs) for m in manifest["modes"]
                        },
                        "semantic_failures_retried": 0,
                    }
                )
            )
            return 0
        url = urlparse(cli.base_url)
        if (
            not cli.output
            or url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Execution requires a new output directory and a safe HTTP(S) URL")
        index = execute(cli, manifest, doc, jobs, config)
        return 0 if index["status"] == "complete" else 1
    except (ValueError, OSError, evaluate.EvaluationError) as error:
        print(f"Recovery stopped: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
