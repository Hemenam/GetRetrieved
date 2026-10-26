"""Plan, run and summarize controlled course-chatbot experiments.

Planning is offline. Execution requires --execute and instructor-approved gold.
This tool queries existing courses; it never ingests or reindexes a document.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate  # noqa: E402

from hrlearnium.eval_tools import (  # noqa: E402
    compare_reports,
    digest,
    latency_summary,
    load_pricing,
    ratio,
    read_report,
    review_template,
    runtime_summary,
    score_review,
)

MODES = {"full_context", "hybrid", "hybrid_rerank"}


def environment() -> dict:
    packages = {}
    for name in (
        "hrlearnium-chatbot",
        "httpx",
        "pydantic",
        "sentence-transformers",
        "transformers",
        "torch",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "hardware_note": "No GPU/CPU utilization or memory benchmark is collected.",
    }


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "name",
        "modes",
        "repetitions",
        "seed",
        "response_mode",
        "max_query_turns",
        "documents",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Manifest has missing or unknown fields")
    if (
        type(manifest["format_version"]) is not int
        or manifest["format_version"] != 1
        or not isinstance(manifest["name"], str)
        or not manifest["name"].strip()
        or manifest["response_mode"] != "explained"
        or not isinstance(manifest["modes"], list)
        or not manifest["modes"]
        or any(not isinstance(m, str) or m not in MODES for m in manifest["modes"])
        or len(set(manifest["modes"])) != len(manifest["modes"])
        or type(manifest["seed"]) is not int
        or any(
            type(manifest[k]) is not int or manifest[k] < 1
            for k in ("repetitions", "max_query_turns")
        )
        or not isinstance(manifest["documents"], list)
        or not manifest["documents"]
    ):
        raise ValueError("Invalid manifest values (v1 benchmarks use explained responses)")
    documents, ids, courses = [], set(), set()
    for doc in manifest["documents"]:
        keys = {"id", "tenant", "course", "dataset", "split"}
        if (
            not isinstance(doc, dict)
            or not keys <= doc.keys() <= keys | {"document", "source_corpus_sha256"}
            or any(not isinstance(doc[k], str) or not doc[k].strip() for k in keys)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", doc["id"])
            or doc["id"] in ids
            or (doc["tenant"], doc["course"]) in courses
            or doc["split"] not in {"development", "held_out"}
            or ("document" in doc and (not isinstance(doc["document"], str) or not doc["document"]))
            or (
                "source_corpus_sha256" in doc
                and (
                    not isinstance(doc["source_corpus_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", doc["source_corpus_sha256"])
                )
            )
        ):
            raise ValueError("Invalid or duplicate document profile")
        ids.add(doc["id"])
        courses.add((doc["tenant"], doc["course"]))
        dataset = (path.parent / doc["dataset"]).resolve()
        all_cases = evaluate.load_cases(dataset, None)
        families = defaultdict(set)
        for case in all_cases:
            if not case.get("question_family_id") or not case.get("expected_language"):
                raise ValueError("Benchmark cases require question_family_id and expected_language")
            families[case["question_family_id"]].add(case.get("split", "development"))
        if any(len(splits) != 1 for splits in families.values()):
            raise ValueError("A question family cannot cross development and held_out splits")
        cases = [c for c in all_cases if c.get("split", "development") == doc["split"]]
        if not cases:
            raise ValueError("A document profile selected no cases")
        document = (path.parent / doc["document"]).resolve() if doc.get("document") else None
        if document and not document.is_file():
            raise ValueError("Optional source document does not exist")
        documents.append(
            doc
            | {
                "dataset_path": dataset,
                "cases": cases,
                "dataset_sha256": digest(dataset),
                "document_path": document,
            }
        )
    if len({d["split"] for d in documents}) != 1:
        raise ValueError("All document profiles in one experiment must use the same split")
    return manifest, documents


def plan(manifest: dict, documents: list[dict]) -> dict:
    multiplier = len(manifest["modes"]) * manifest["repetitions"]
    scored = sum(len(d["cases"]) for d in documents) * multiplier
    setups = (
        sum(bool(c.get("previous_question")) for d in documents for c in d["cases"]) * multiplier
    )
    pending = sum(c.get("review_status") != "approved" for d in documents for c in d["cases"])
    blockers = []
    if pending:
        blockers.append(f"{pending} selected gold cases still require instructor approval")
    if scored + setups > manifest["max_query_turns"]:
        blockers.append("Planned setup + scored turns exceed max_query_turns")
    return {
        "name": manifest["name"],
        "network_requests_made": 0,
        "modes": manifest["modes"],
        "repetitions": manifest["repetitions"],
        "scored_turns": scored,
        "setup_turns_upper_bound": setups,
        "query_turns_upper_bound": scored + setups,
        "max_query_turns": manifest["max_query_turns"],
        "documents": [
            {
                "id": d["id"],
                "course": d["course"],
                "split": d["split"],
                "cases": len(d["cases"]),
                "dataset_sha256": d["dataset_sha256"],
                "categories": sorted({c["category"] for c in d["cases"]}),
            }
            for d in documents
        ],
        "execution_ready": not blockers,
        "blockers": blockers,
        "budget_note": "Turn cap is not a USD cap. One turn can invoke multiple LLM/embedding calls. "
        "Preflight and authenticated evidence GETs are additional HTTP requests. "
        "Provider spend limits must be configured separately.",
    }


def write_json(path: Path, value: dict, *, exclusive=True):
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def request_args(cli, manifest, doc, mode):
    return argparse.Namespace(
        base_url=cli.base_url,
        local_auth=cli.local_auth,
        tenant=doc["tenant"],
        course=doc["course"],
        subject=cli.subject,
        document=doc["document_path"],
        timeout=cli.timeout,
        delay_seconds=cli.delay_seconds,
        response_mode=manifest["response_mode"],
        retrieval_mode=mode,
        expect_retrieval_mode=mode,
    )


def preflight(cli, manifest, documents):
    """Authenticated prerequisite check only; this does not verify provider health."""
    import httpx

    from hrlearnium.schemas import RetrievalOptions

    results = []
    with httpx.Client(
        base_url=cli.base_url.rstrip("/"),
        timeout=cli.timeout,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        for doc in documents:
            args = request_args(cli, manifest, doc, manifest["modes"][0])
            headers = evaluate.auth_headers(args)
            try:
                response = client.get(
                    f"/v1/courses/{quote(doc['course'], safe='')}/retrieval-options",
                    headers=headers(),
                )
                response.raise_for_status()
                result = RetrievalOptions.model_validate(response.json())
            except (httpx.HTTPError, ValueError):
                raise ValueError(
                    f"Prerequisite check failed for {doc['id']}; no scored queries sent"
                ) from None
            available = {m.mode for m in result.modes if m.available}
            if not set(manifest["modes"]) <= available:
                raise ValueError(
                    f"Requested modes are unavailable for {doc['id']}; no scored queries sent"
                )
            results.append({"document_id": doc["id"], "options": result.model_dump()})
    return results


def run(cli, manifest_path: Path, manifest: dict, documents: list[dict]) -> Path:
    planned = plan(manifest, documents)
    if not planned["execution_ready"]:
        raise ValueError("; ".join(planned["blockers"]))
    pricing = load_pricing(cli.pricing)
    # Refuse reuse, including partial directories; rerunning must not silently replay paid calls.
    cli.output.mkdir(parents=True, exist_ok=False)
    snapshots = cli.output / "inputs"
    snapshots.mkdir()
    write_json(snapshots / "manifest.json", manifest)
    if pricing:
        write_json(snapshots / "pricing.json", pricing)
    for doc in documents:
        with (snapshots / f"{doc['id']}.jsonl").open("xb") as handle:
            handle.write(doc["dataset_path"].read_bytes())
        if digest(snapshots / f"{doc['id']}.jsonl") != doc["dataset_sha256"]:
            raise ValueError("Dataset changed after planning")
    index = {
        "format_version": 1,
        "name": manifest["name"],
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": digest(manifest_path),
        "plan": planned,
        "environment": environment(),
        "pricing_sha256": digest(snapshots / "pricing.json") if pricing else None,
        "status": "preflight",
        "runs": [],
        "comparability_errors": [],
    }
    checkpoint = cli.output / "index.json"
    write_json(checkpoint, index)
    try:
        index["prerequisites"] = preflight(cli, manifest, documents)
        index["status"] = "running"
        write_json(checkpoint, index, exclusive=False)
        rng = random.Random(manifest["seed"])
        for repeat in range(1, manifest["repetitions"] + 1):
            schedule = [(doc, mode) for doc in documents for mode in manifest["modes"]]
            rng.shuffle(schedule)
            for doc, mode in schedule:
                if digest(manifest_path) != index["manifest_sha256"]:
                    raise ValueError("Manifest changed during the experiment")
                if digest(doc["dataset_path"]) != doc["dataset_sha256"]:
                    raise ValueError(
                        "Dataset changed during the experiment; stop and start a new run"
                    )
                args = request_args(cli, manifest, doc, mode)
                outcomes, metadata = evaluate.api_evaluation(doc["cases"], args)
                name = f"{doc['id']}--{mode}--r{repeat}"
                report = {
                    "created_at": datetime.now(UTC).isoformat(),
                    "dataset_status": "instructor_approved",
                    "dataset_sha256": doc["dataset_sha256"],
                    "dataset_cases_loaded": len(doc["cases"]),
                    "metadata": metadata | {"document_profile": doc["id"], "repeat": repeat},
                    "runtime": runtime_summary(outcomes, pricing),
                    "summary": evaluate.api_summary(outcomes),
                    "by_category": {
                        category: evaluate.api_summary(
                            [i for i in outcomes if i["category"] == category]
                        )
                        for category in sorted({i["category"] for i in outcomes})
                    },
                    "latency_ms": latency_summary(
                        [i["latency_ms"] for i in outcomes if "latency_ms" in i]
                    ),
                    "outcomes": outcomes,
                }
                report_path = cli.output / f"{name}.json"
                write_json(report_path, report)
                write_json(cli.output / f"{name}.review.json", review_template(report_path))
                index["runs"].append(
                    {
                        "document_id": doc["id"],
                        "mode": mode,
                        "repeat": repeat,
                        "report": report_path.name,
                        "report_sha256": digest(report_path),
                        "review": f"{name}.review.json",
                    }
                )
                write_json(checkpoint, index, exclusive=False)
                traces = [
                    i[key]
                    for i in outcomes
                    for key in ("setup_evaluation", "evaluation")
                    if isinstance(i.get(key), dict)
                ]
                expected_source = doc.get("source_corpus_sha256")
                if expected_source and any(
                    t.get("source_passage_count", 0) > 0
                    and t.get("source_corpus_sha256") != expected_source
                    for t in traces
                ):
                    raise ValueError("Observed corpus does not match the pinned source fingerprint")
                paths = [
                    cli.output / r["report"] for r in index["runs"] if r["document_id"] == doc["id"]
                ]
                # Validate even the first run internally before spending on the next one.
                compare_reports(paths if len(paths) > 1 else paths * 2)
                print(
                    json.dumps(
                        {"completed": name, "errors": report["summary"]["operational_errors"]}
                    )
                )
        index["status"] = "complete"
    except (ValueError, evaluate.EvaluationError) as error:
        index["status"] = "stopped"
        index["comparability_errors"].append(str(error))
        raise
    except (KeyboardInterrupt, OSError):
        index["status"] = "interrupted"
        raise
    finally:
        write_json(checkpoint, index, exclusive=False)
    return checkpoint


def summarize(directory: Path) -> dict:
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    pricing = None
    if index.get("pricing_sha256"):
        pricing_path = directory / "inputs/pricing.json"
        if digest(pricing_path) != index["pricing_sha256"]:
            raise ValueError("Captured pricing changed")
        pricing = load_pricing(pricing_path)
    groups, per_doc_paths, semantic_scores = defaultdict(list), defaultdict(list), defaultdict(list)
    splits = {d["id"]: d["split"] for d in index["plan"]["documents"]}
    seen_runs = set()
    for entry in index["runs"]:
        identity = (entry["document_id"], entry["mode"], entry["repeat"])
        if identity in seen_runs:
            raise ValueError("Duplicate experiment run in index")
        seen_runs.add(identity)
        path = directory / entry["report"]
        if digest(path) != entry["report_sha256"]:
            raise ValueError("A captured report changed; stored reviews cannot be reused")
        report = read_report(path)
        if any(i.get("retrieval_mode") not in (None, entry["mode"]) for i in report["outcomes"]):
            raise ValueError("Report mode does not match the experiment index")
        review_path = directory / entry["review"]
        decisions, language = {}, []
        if review_path.exists():
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if review.get("reviewer") or any(c.get("reviewed") for c in review.get("cases", [])):
                scored = score_review(path, review)
                semantic_scores[(entry["document_id"], entry["mode"])].append(scored)
                decisions = {c["id"]: c["semantic_pass"] for c in scored["decisions"]}
                language = [
                    c["language_correct"]
                    for c in review["cases"]
                    if c.get("reviewed") and "language_correct" in c
                ]
        groups[(entry["document_id"], entry["mode"])].append((report, decisions, language))
        per_doc_paths[entry["document_id"]].append(path)
    comparable = index["status"] == "complete" and not index.get("comparability_errors")
    comparisons = {}
    for doc, paths in per_doc_paths.items():
        try:
            comparisons[doc] = compare_reports(paths) if len(paths) > 1 else {"winner": None}
        except ValueError as error:
            comparisons[doc] = {"error": str(error), "winner": None}
            comparable = False
    expected = {(d["id"], m) for d in index["plan"]["documents"] for m in index["plan"]["modes"]}
    if set(groups) != expected or any(
        len(runs) != index["plan"]["repetitions"] for runs in groups.values()
    ):
        comparable = False
    rows = []
    for (doc, mode), runs in sorted(groups.items()):
        outcomes = [i for report, _, _ in runs for i in report["outcomes"]]
        positives = [i for i in outcomes if i["expected_status"] == "answered"]
        negatives = [i for i in outcomes if i["expected_status"] != "answered"]
        reviewed = sum(len(decisions) for _, decisions, _ in runs)
        evaluated = sum(i["evaluation_status"] == "evaluated" for i in outcomes)
        ready = reviewed == evaluated and all(
            i.get("gold", {}).get("review_status") == "approved" for i in outcomes
        )
        successes = sum(
            decisions.get(i["id"]) is True
            and i.get("actual_status") == "answered"
            and not i.get("setup_failed", False)
            for report, decisions, _ in runs
            for i in report["outcomes"]
            if i["expected_status"] == "answered"
        )
        rows.append(
            {
                "document_id": doc,
                "split": splits[doc],
                "mode": mode,
                "repetitions": len(runs),
                "cases": len(outcomes),
                "answerable_cases": len(positives),
                "non_answer_cases": len(negatives),
                "operational_errors": len(outcomes) - evaluated,
                "review_coverage": ratio(reviewed, evaluated),
                "semantic_review_complete": ready,
                "complete_grounded_answer_rate": ratio(successes, len(positives))
                if ready
                else None,
                "false_answer_rate": ratio(
                    sum(i.get("actual_status") == "answered" for i in negatives), len(negatives)
                ),
                "false_refusal_rate": ratio(
                    sum(i.get("actual_status") == "refused" for i in positives), len(positives)
                ),
                "status_accuracy": ratio(
                    sum(i.get("status_matches") is True for i in outcomes), len(outcomes)
                ),
                "status_confusion_matrix": evaluate.api_summary(outcomes)[
                    "status_confusion_matrix"
                ],
                "semantic_metrics_on_reviewed_responses": {
                    metric: ratio(
                        sum(s[metric]["numerator"] for s in semantic_scores[(doc, mode)]),
                        sum(s[metric]["denominator"] for s in semantic_scores[(doc, mode)]),
                    )
                    for metric in (
                        "status_correctness",
                        "answer_relevance",
                        "answer_completeness",
                        "required_fact_coverage",
                        "statement_support",
                        "citation_support",
                        "answers_with_unsupported_statements",
                    )
                },
                "automated_checks_passed": ratio(
                    sum(i.get("passed_automated_checks") is True for i in outcomes), len(outcomes)
                ),
                "language_compliance": ratio(
                    sum(x is True for _, _, lang in runs for x in lang),
                    sum(len(lang) for _, _, lang in runs),
                ),
                "query_latency_ms": latency_summary(
                    [i["query_latency_ms"] for i in outcomes if "query_latency_ms" in i]
                ),
                "case_latency_ms": latency_summary(
                    [i["latency_ms"] for i in outcomes if "latency_ms" in i]
                ),
                "runtime": runtime_summary(outcomes, pricing),
                "estimated_cost_usd": sum(r["runtime"]["estimated_cost_usd"] for r, _, _ in runs)
                if all(
                    r.get("runtime", {}).get("estimated_cost_usd") is not None for r, _, _ in runs
                )
                else None,
            }
        )
    macro = []
    quality_ready = comparable and bool(rows) and all(r["semantic_review_complete"] for r in rows)
    for mode in index["plan"]["modes"]:
        mode_rows = [r for r in rows if r["mode"] == mode]
        rates = [(r["complete_grounded_answer_rate"] or {}).get("rate") for r in mode_rows]
        score = (
            sum(rates) / len(rates)
            if quality_ready and rates and all(x is not None for x in rates)
            else None
        )
        eligible = score is not None and all(
            r["false_answer_rate"]["denominator"] > 0
            and r["false_answer_rate"]["numerator"] == 0
            and r["operational_errors"] == 0
            and r["automated_checks_passed"]["rate"] == 1
            for r in mode_rows
        )
        macro.append(
            {
                "mode": mode,
                "documents": len(mode_rows),
                "macro_complete_grounded_answer_rate": score,
                "safety_gate_passed": eligible,
                "observed_rank": None,
            }
        )
    eligible_scores = sorted(
        {m["macro_complete_grounded_answer_rate"] for m in macro if m["safety_gate_passed"]},
        reverse=True,
    )
    for item in macro:
        if item["safety_gate_passed"]:
            item["observed_rank"] = (
                eligible_scores.index(item["macro_complete_grounded_answer_rate"]) + 1
            )
    return {
        "name": index["name"],
        "comparable": comparable,
        "quality_review_complete": quality_ready,
        "documents": rows,
        "macro_leaderboard": macro,
        "paired_comparisons": comparisons,
        "winner": None,
        "limitations": [
            "Observed ranks are descriptive, not statistically established superiority or release approval.",
            "This pilot has no confidence intervals. Repeats/paraphrases are not independent questions.",
            "No rank until all evaluated cases have bound human reviews and all selected gold is approved.",
            "Errors and false refusals remain in the answerable denominator; source checks are not semantic grades.",
            "Safety gate: zero observed false answers, no operational errors and all automated checks passed.",
            "Macro quality weights document profiles equally; latency/cost stay per document and mode.",
            "Section-only scope, cross-course authorization and new document ingestion need separate tests.",
        ],
    }


def export_summary(output: Path, result: dict):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result)
    fields = [
        "document_id",
        "split",
        "mode",
        "cases",
        "answerable_cases",
        "non_answer_cases",
        "operational_errors",
        "semantic_review_complete",
        "complete_grounded_answer_rate",
        "false_answer_rate",
        "false_refusal_rate",
        "language_compliance",
        "query_median_ms",
        "query_p95_ms",
        "estimated_cost_usd",
    ]
    with (output / "leaderboard.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["documents"]:
            flat = {k: row[k] for k in fields if k in row}
            for key, value in flat.items():
                if isinstance(value, dict):
                    flat[key] = value["rate"]
            flat.update(
                query_median_ms=row["query_latency_ms"]["median"],
                query_p95_ms=row["query_latency_ms"]["p95"],
            )
            writer.writerow(flat)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        part = sub.add_parser(command)
        part.add_argument("--manifest", type=Path, default=ROOT / "evals/pilot/manifest.json")
        if command == "run":
            part.add_argument(
                "--execute", action="store_true", help="Authorize API queries; otherwise plan only"
            )
            part.add_argument("--base-url", default="http://127.0.0.1:8000")
            part.add_argument("--local-auth", action="store_true")
            part.add_argument("--subject", default="benchmark")
            part.add_argument("--output", type=Path)
            part.add_argument("--timeout", type=float, default=180)
            part.add_argument("--delay-seconds", type=float, default=1)
            part.add_argument("--pricing", type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("--run-dir", type=Path, required=True)
    summary.add_argument(
        "--output", type=Path, required=True, help="New directory for JSON and CSV"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "summarize":
            result = summarize(args.run_dir)
            export_summary(args.output, result)
            print(
                json.dumps({"output": str(args.output), "leaderboard": result["macro_leaderboard"]})
            )
        else:
            manifest, documents = load_manifest(args.manifest)
            if args.command == "plan" or not args.execute:
                print(json.dumps(plan(manifest, documents), ensure_ascii=False, indent=2))
                return 0
            if not args.output:
                raise ValueError("--execute requires a new --output directory")
            url = urlparse(args.base_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError("API URL must be HTTP(S) without credentials, query or fragment")
            if (
                not math.isfinite(args.timeout)
                or args.timeout <= 0
                or not 0 <= args.delay_seconds <= 60
            ):
                raise ValueError(
                    "Timeout must be positive/finite and delay between 0 and 60 seconds"
                )
            checkpoint = run(args, args.manifest, manifest, documents)
            print(
                json.dumps(
                    {"index": str(checkpoint), "next": "Complete response reviews, then summarize"}
                )
            )
        return 0
    except (ValueError, evaluate.EvaluationError) as error:
        print(f"Benchmark error: {error}", file=sys.stderr)
        return 2
    except OSError:
        print(
            "Benchmark error: could not read input or create a new output directory",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print(
            "Benchmark interrupted; completed reports were preserved. Do not reuse that output directory.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
