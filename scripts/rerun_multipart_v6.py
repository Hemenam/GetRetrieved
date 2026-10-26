"""Fresh, explicitly authorized 18-query multipart rerun; prior captures stay immutable."""

import argparse
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import run_extension_v6 as runner


def schedule_cases(cases, seed):
    selected = [c for c in cases if len(c["parts"]) > 1]
    if len(selected) != 6:
        raise ValueError("Expected exactly six multipart/mixed cases")
    rng, schedule = random.Random(seed), []
    for case in selected:
        modes = ["full_context", "hybrid"]
        rng.shuffle(modes)
        schedule.extend((case, mode) for mode in modes)
    # Finish non-reranked comparisons before entering the potentially CPU-heavy block.
    # This is a documented block-order confound, not a randomized latency experiment.
    schedule.extend((case, "hybrid_rerank") for case in selected)
    return schedule


def execute(cli, manifest, cases, settings, chapters):
    schedule = schedule_cases(cases, manifest["seed"])
    url = urlparse(cli.base_url)
    if (
        cli.max_query_turns != 18
        or cli.output is None
        or not 15 <= cli.delay_seconds <= 60
        or not 0 < cli.timeout <= 1200
        or url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Requires new output, loopback URL, 18-query cap, >=15s pacing, <=1200s timeout"
        )
    config = runner.fixed_config(runner.configuration(settings))
    cli.output.mkdir(parents=True, exist_ok=False)
    (cli.output / "inputs").mkdir()
    for name in ["manifest.json", "cases.jsonl", "catalog.json"]:
        (cli.output / "inputs" / name).write_bytes((cli.pack / name).read_bytes())
    index = {
        "kind": "explicit_multipart_v6_rerun",
        "created_at": datetime.now(UTC).isoformat(),
        "authorization": "User explicitly requested rerun after incomplete multipart results on 2026-10-20.",
        "prior_results_unchanged": True,
        "reference_review": manifest["reference_review"],
        "manifest_sha256": runner.digest(cli.pack / "manifest.json"),
        "dataset_sha256": manifest["cases_sha256"],
        "catalog_sha256": manifest["catalog_sha256"],
        "source_corpus_sha256": manifest["source"]["source_corpus_sha256"],
        "fixed_configuration": config,
        "selected_case_ids": list(dict.fromkeys(c["id"] for c, _ in schedule)),
        "planned_queries": 18,
        "attempted_queries": 0,
        "captures": [],
        "pending_request": None,
        "status": "preflight",
        "timeout_seconds": cli.timeout,
        "delay_seconds": cli.delay_seconds,
        "outer_http_retries": 0,
        "hidden_paid_warmups": 0,
        "runtime_environment": {
            k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "limitations": [
            "Fresh repeat, not replacement of earlier language failure/timeout.",
            "CPU thread cap and timeout differ from the original five-minute-budget test.",
            "Full-context/hybrid pairs are interleaved first; reranker block last. Block order can confound latency.",
            "No semantic or outer HTTP retries. Existing bounded provider transport retry unchanged.",
            "References are assistant source-audited, not independently instructor-approved.",
        ],
    }
    checkpoint = cli.output / "index.json"
    runner.bench.write_json(checkpoint, index)
    doc = {
        "id": manifest["course"],
        "tenant": manifest["tenant"],
        "course": manifest["course"],
        "document_path": None,
    }
    try:
        runner.bench.preflight(cli, manifest, [doc])
        index["status"] = "running"
        for case, mode in schedule:
            time.sleep(cli.delay_seconds)
            current_manifest, _, current_settings, _ = runner.load_pack(cli.pack)
            if (
                current_manifest != manifest
                or runner.fixed_config(runner.configuration(current_settings)) != config
            ):
                raise ValueError("Source/inputs/configuration changed during collection")
            index.update(
                attempted_queries=index["attempted_queries"] + 1,
                pending_request={"case_id": case["id"], "mode": mode},
            )
            runner.bench.write_json(checkpoint, index, exclusive=False)
            outcomes, metadata = runner.evaluate.api_evaluation(
                [runner.api_case(case)], runner.bench.request_args(cli, manifest, doc, mode)
            )
            outcome = outcomes[0]
            outcome["extension_definition"] = case
            outcome["extension_checks"] = runner.extension_checks(case, outcome, chapters)
            trace = outcome.get("evaluation")
            if trace and runner.fixed_config(trace["configuration"]) != config:
                outcome["configuration_mismatch"] = True
            path = cli.output / f"{case['id']}--{mode}.json"
            runner.bench.write_json(
                path,
                {
                    "kind": "explicit_multipart_v6_capture",
                    "dataset_sha256": manifest["cases_sha256"],
                    "metadata": metadata,
                    "outcomes": outcomes,
                    "runtime": runner.runtime_summary(outcomes),
                },
            )
            index["captures"].append(
                {
                    "case_id": case["id"],
                    "mode": mode,
                    "report": path.name,
                    "report_sha256": runner.digest(path),
                }
            )
            index["pending_request"] = None
            stopped = bool(outcome.get("error") or outcome.get("configuration_mismatch"))
            index["status"] = "stopped" if stopped else "running"
            runner.bench.write_json(checkpoint, index, exclusive=False)
            print(
                runner.json.dumps(
                    {
                        "completed": len(index["captures"]),
                        "case_id": case["id"],
                        "mode": mode,
                        "status": outcome.get("actual_status"),
                        "checks": outcome["extension_checks"]["pass"],
                        "error": outcome.get("error"),
                    }
                ),
                flush=True,
            )
            if stopped:
                return index
        index["status"] = "complete"
        runner.bench.write_json(checkpoint, index, exclusive=False)
        return index
    except BaseException:
        index["status"] = "stopped"
        runner.bench.write_json(checkpoint, index, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-query-turns", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--local-auth", action="store_true")
    parser.add_argument("--subject", default="multipart-v6-explicit-rerun")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--delay-seconds", type=float, default=20)
    cli = parser.parse_args()
    manifest, cases, settings, chapters = runner.load_pack(cli.pack)
    schedule = schedule_cases(cases, manifest["seed"])
    if not cli.execute:
        print(
            runner.json.dumps(
                {"cases": len(schedule) // 3, "queries": len(schedule), "network_requests": 0}
            )
        )
        return
    result = execute(cli, manifest, cases, settings, chapters)
    raise SystemExit(0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
