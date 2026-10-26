"""Explicit operational recovery of the stopped v6 extension, never semantic retries."""

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import run_extension_v6 as runner


def remaining(directory, manifest, cases):
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    if index["status"] != "stopped" or index["pending_request"] is not None:
        raise ValueError(
            "Only an explicitly stopped run without an ambiguous request is recoverable"
        )
    if index["dataset_sha256"] != manifest["cases_sha256"]:
        raise ValueError("Dataset changed")
    seen, completed = set(), set()
    for capture in index["captures"]:
        key = (capture["case_id"], capture["mode"])
        if key in seen or Path(capture["report"]).name != capture["report"]:
            raise ValueError("Duplicate or unsafe capture")
        seen.add(key)
        path = directory / capture["report"]
        if runner.digest(path) != capture["report_sha256"]:
            raise ValueError("Captured report changed")
        outcome = json.loads(path.read_text(encoding="utf-8"))["outcomes"][0]
        if not outcome.get("error"):
            completed.add(key)  # Refusals and semantic failures are NOT retried.
        elif outcome["error"] != {"kind": "http_status", "status_code": 429, "stage": "query"}:
            raise ValueError("Recovery only covers the observed local query HTTP 429")
    schedule, rng = [], random.Random(manifest["seed"])
    for case in cases:
        modes = list(manifest["modes"])
        rng.shuffle(modes)
        schedule.extend((case, mode) for mode in modes if (case["id"], mode) not in completed)
    if len(seen) != index["attempted_queries"] or not seen <= {
        (c["id"], m) for c in cases for m in manifest["modes"]
    }:
        raise ValueError("Unexpected original request set")
    return index, schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-query-turns", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--local-auth", action="store_true")
    parser.add_argument("--subject", default="v6-extension-benchmark")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--delay-seconds", type=float, default=15)
    cli = parser.parse_args()
    manifest, cases, settings, chapters = runner.load_pack(cli.pack)
    original, schedule = remaining(cli.original, manifest, cases)
    print(json.dumps({"remaining_queries": len(schedule), "semantic_replays": 0}), flush=True)
    if not cli.execute:
        return
    from urllib.parse import urlparse

    url = urlparse(cli.base_url)
    if (
        cli.output is None
        or cli.max_query_turns != len(schedule)
        or not 15 <= cli.delay_seconds <= 60
        or not 0 < cli.timeout <= 300
        or url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        parser.error("Requires exact pending-query cap, new output, loopback URL, and >=15s pacing")
    config = runner.fixed_config(runner.configuration(settings))
    if config != original["fixed_configuration"]:
        raise ValueError("Configuration changed; not a comparable recovery")
    cli.output.mkdir(parents=True, exist_ok=False)
    index = {
        "kind": "v6_extension_operational_recovery",
        "created_at": datetime.now(UTC).isoformat(),
        "authorization": "User asked to continue on 2026-10-20; explicit recovery disclosed before execution.",
        "original_index_sha256": runner.digest(cli.original / "index.json"),
        "dataset_sha256": manifest["cases_sha256"],
        "fixed_configuration": config,
        "planned_queries": len(schedule),
        "attempted_queries": 0,
        "captures": [],
        "pending_request": None,
        "status": "preflight",
        "delay_seconds": cli.delay_seconds,
        "outer_http_retries": 0,
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
            time.sleep(cli.delay_seconds)  # Also protects the first query after preflight.
            current_manifest, _, current_settings, _ = runner.load_pack(cli.pack)
            if (
                current_manifest != manifest
                or runner.fixed_config(runner.configuration(current_settings)) != config
            ):
                raise ValueError("Inputs/configuration changed during recovery")
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
            name = f"{case['id']}--{mode}.json"
            path = cli.output / name
            runner.bench.write_json(
                path,
                {
                    "kind": "v6_extension_capture",
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
                    "report": name,
                    "report_sha256": runner.digest(path),
                }
            )
            index["pending_request"] = None
            stopped = bool(outcome.get("error") or outcome.get("configuration_mismatch"))
            index["status"] = "stopped" if stopped else "running"
            runner.bench.write_json(checkpoint, index, exclusive=False)
            print(
                json.dumps(
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
                return
        index["status"] = "complete"
        runner.bench.write_json(checkpoint, index, exclusive=False)
    except BaseException:
        index["status"] = "stopped"
        runner.bench.write_json(checkpoint, index, exclusive=False)
        raise


if __name__ == "__main__":
    main()
