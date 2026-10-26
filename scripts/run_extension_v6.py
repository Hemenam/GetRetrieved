"""Explicit checkpointed 36-query v6 extension; no resume or semantic retries."""

import argparse
import json
import random
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
from recover_benchmark_errors import fixed_config  # noqa: E402

from hrlearnium.config import Settings  # noqa: E402
from hrlearnium.eval_tools import digest, runtime_summary  # noqa: E402
from hrlearnium.evaluation import configuration  # noqa: E402


def api_case(case):
    quotes = list(dict.fromkeys(q for p in case["parts"] for q in p["quotes"]))
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_status": case["expected_status"],
        "expected_chapters": [],
        "required_quotes": quotes,
        "required_facts": [],
        "expected_language": case["expected_language"],
        "notes": "V6 references: see extension_definition; assistant-audited, not human-approved.",
        "review_status": "assistant_source_audited",
        "reviewer": "Assistant",
        "split": "development",
    }


def extension_checks(case, outcome, source_chapters):
    if outcome.get("error") or not outcome.get("response"):
        return {"pass": False, "reason": "operational_error"}
    body = outcome["response"]
    if len(case["parts"]) == 1:
        parts = [{**body, "excerpt_ids": [e["id"] for e in body["excerpts"]]}]
    else:
        parts = body.get("parts", [])
    details = []
    for expected, actual in zip(case["parts"], parts):
        checks = {
            "status": expected["status"] == actual["status"],
            "kind": expected["kind"] == actual["answer_kind"],
        }
        own = [e for e in body["excerpts"] if e["id"] in actual.get("excerpt_ids", [])]
        checks["required_evidence_anchors"] = all(
            any(q in e["text"] for e in own) for q in expected["quotes"]
        )
        if expected["status"] != "answered":
            checks["no_unsupported_evidence"] = (
                not own and actual.get("explanation") is None and actual.get("metadata") is None
            )
        elif expected["kind"] == "metadata":
            evidence = actual.get("metadata") or {}
            docs = evidence.get("outline", {}).get("documents", [])
            chapters = docs[0].get("chapters", []) if len(docs) == 1 else []
            checks["metadata_field"] = evidence.get("field") == expected["metadata_field"]
            field = expected["metadata_field"]
            if field in {"chapter_count", "chapter_titles"}:
                checks["complete_structure"] = [
                    (c["number"], c["title"]) for c in chapters
                ] == source_chapters
            elif field in {"chapter_title", "chapter_sections"}:
                checks["target_chapter"] = evidence.get("chapter_number") == expected["chapter"]
            if field == "chapter_sections":
                target = next((c for c in chapters if c["number"] == expected["chapter"]), {})
                checks["sections"] = set(target.get("sections", [])) == {
                    "introduction",
                    "explanation",
                    "example",
                    "exercise",
                }
        details.append(checks)
    shape = len(parts) == len(case["parts"]) and (
        len(parts) == 1 or body.get("answer_kind") == "multi"
    )
    passed = (
        outcome.get("passed_automated_checks") is True
        and shape
        and all(all(c.values()) for c in details)
    )
    return {
        "pass": passed,
        "part_count_matches": shape,
        "parts": details,
        "semantic_accuracy": "requires_assistant_or_human_review",
    }


def load_pack(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cases = [
        json.loads(line)
        for line in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        digest(directory / "cases.jsonl") != manifest["cases_sha256"]
        or digest(directory / "catalog.json") != manifest["catalog_sha256"]
    ):
        raise ValueError("Extension inputs changed after reference audit")
    if len(cases) != 12 or len({c["id"] for c in cases}) != 12 or manifest["max_query_turns"] != 36:
        raise ValueError("Expected the explicitly scoped 12-case/36-query extension")
    settings = Settings()
    source, passages = read_source(settings.database_path, manifest["tenant"], manifest["course"])
    if source != manifest["source"]:
        raise ValueError("Current document differs from source-audited references")
    chapters = sorted({(p["chapter_number"], p["chapter_title"]) for p in passages})
    return manifest, cases, settings, chapters


def execute(cli, manifest, cases, settings, chapters):
    if cli.max_query_turns != 36:
        raise ValueError("An explicit 36-query cap is required; this is not a USD limit")
    expected_config = fixed_config(configuration(settings))
    cli.output.mkdir(parents=True, exist_ok=False)
    (cli.output / "inputs").mkdir()
    for name in ("manifest.json", "cases.jsonl", "catalog.json"):
        (cli.output / "inputs" / name).write_bytes((cli.pack / name).read_bytes())
    checkpoint = cli.output / "index.json"
    index = {
        "kind": "v6_metadata_multipart_extension",
        "created_at": datetime.now(UTC).isoformat(),
        "authorization": "User explicitly requested adding these questions and running them on 2026-10-20",
        "reference_review": manifest["reference_review"],
        "manifest_sha256": digest(cli.pack / "manifest.json"),
        "dataset_sha256": manifest["cases_sha256"],
        "catalog_sha256": manifest["catalog_sha256"],
        "source_corpus_sha256": manifest["source"]["source_corpus_sha256"],
        "fixed_configuration": expected_config,
        "planned_queries": 36,
        "attempted_queries": 0,
        "hidden_warmups": 0,
        "outer_http_retries": 0,
        "captures": [],
        "pending_request": None,
        "status": "preflight",
        "human_review_complete": False,
    }
    bench.write_json(checkpoint, index)
    try:
        doc = {
            "id": manifest["course"],
            "tenant": manifest["tenant"],
            "course": manifest["course"],
            "document_path": None,
        }
        bench.preflight(cli, manifest, [doc])
        index["status"] = "running"
        bench.write_json(checkpoint, index, exclusive=False)
        rng = random.Random(manifest["seed"])
        for case in cases:
            modes = list(manifest["modes"])
            rng.shuffle(modes)
            for mode in modes:
                if index["attempted_queries"]:
                    time.sleep(cli.delay_seconds)
                if (
                    digest(cli.pack / "manifest.json") != index["manifest_sha256"]
                    or digest(cli.pack / "cases.jsonl") != index["dataset_sha256"]
                ):
                    raise ValueError("Benchmark inputs changed during collection")
                if fixed_config(configuration(Settings())) != expected_config:
                    raise ValueError("Application settings/code changed during collection")
                source, _ = read_source(
                    settings.database_path, manifest["tenant"], manifest["course"]
                )
                if source != manifest["source"]:
                    raise ValueError("Source changed during collection")
                index.update(
                    pending_request={"case_id": case["id"], "mode": mode},
                    attempted_queries=index["attempted_queries"] + 1,
                )
                bench.write_json(checkpoint, index, exclusive=False)
                args = bench.request_args(cli, manifest, doc, mode)
                outcomes, metadata = evaluate.api_evaluation([api_case(case)], args)
                outcome = outcomes[0]
                outcome["extension_checks"] = extension_checks(case, outcome, chapters)
                outcome["extension_definition"] = case
                trace = outcome.get("evaluation")
                if trace and fixed_config(trace["configuration"]) != expected_config:
                    outcome["configuration_mismatch"] = True
                name = f"{case['id']}--{mode}.json"
                path = cli.output / name
                bench.write_json(
                    path,
                    {
                        "kind": "v6_extension_capture",
                        "dataset_sha256": index["dataset_sha256"],
                        "metadata": metadata,
                        "outcomes": outcomes,
                        "runtime": runtime_summary(outcomes),
                    },
                )
                index["captures"].append(
                    {
                        "case_id": case["id"],
                        "mode": mode,
                        "report": name,
                        "report_sha256": digest(path),
                    }
                )
                index["pending_request"] = None
                if outcome.get("error") or outcome.get("configuration_mismatch"):
                    index["status"] = "stopped"
                bench.write_json(checkpoint, index, exclusive=False)
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
                if index["status"] == "stopped":
                    return index
        index["status"] = "complete"
        bench.write_json(checkpoint, index, exclusive=False)
        return index
    except Exception:
        index["status"] = "stopped"
        bench.write_json(checkpoint, index, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-query-turns", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--local-auth", action="store_true")
    parser.add_argument("--subject", default="v6-extension-benchmark")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--delay-seconds", type=float, default=2)
    cli = parser.parse_args()
    manifest, cases, settings, chapters = load_pack(cli.pack)
    if not cli.execute:
        print(
            json.dumps(
                {
                    "selected_cases": len(cases),
                    "planned_queries": len(cases) * 3,
                    "network_requests": 0,
                }
            )
        )
        return
    url = urlparse(cli.base_url)
    if (
        cli.output is None
        or url.scheme not in {"http", "https"}
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.username
        or url.password
        or url.query
        or url.fragment
        or not 0 <= cli.delay_seconds <= 60
        or not 0 < cli.timeout <= 300
    ):
        parser.error(
            "Execution requires a new output directory, loopback server URL and bounded timeout/delay"
        )
    index = execute(cli, manifest, cases, settings, chapters)
    raise SystemExit(0 if index["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
