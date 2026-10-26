"""Checkpointed natural-student benchmark through the actual embedded LMS component.

No server/JWT, source hints in prompts, warmups, outer retries or paid judging.
Proposed references remain proposed; automatic checks are not semantic accuracy.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import prepare_natural_benchmark as pack  # noqa: E402
from audit_chapter_coverage import read_source  # noqa: E402
from evaluate import response_contract_checks  # noqa: E402

from hrlearnium.component import create_assistant  # noqa: E402
from hrlearnium.core_config import CoreSettings  # noqa: E402
from hrlearnium.eval_tools import digest, latency_summary, runtime_summary  # noqa: E402
from hrlearnium.evaluation import configuration  # noqa: E402
from hrlearnium.language import explanation_language_matches  # noqa: E402
from hrlearnium.schemas import Principal, QueryRequest, QueryResponse  # noqa: E402

DEFAULT = ROOT / "evals/natural-student-v2"


def write_json(path, value, *, exclusive=False):
    if exclusive and path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(settings):
    """Detect settings/code changes without serializing keys, endpoints or credentials."""
    value = settings.model_dump(mode="json", exclude={"api_key"})
    value["api_key"] = hashlib.sha256(settings.api_key.get_secret_value().encode()).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def request_payload(question, mode, conversation_id=None):
    body = {
        "question": question,
        "response_mode": "explained",
        "retrieval_mode": mode,
        "include_evaluation": True,
    }
    if conversation_id:
        body["conversation_id"] = conversation_id
    return body


def failed_selection(directory, manifest, cases, pack_directory, mode):
    """Select prior diagnostic failures only, from an immutable matching run."""
    from report_natural_benchmark import load_run

    index, report, report_hash = load_run(directory)
    if (
        index["mode"] != mode
        or index["source"] != manifest["source"]
        or index["dataset_sha256"] != digest(pack_directory / "cases.jsonl")
        or index["manifest_sha256"] != digest(pack_directory / "manifest.json")
    ):
        raise ValueError("Parent mode, source or references differ")
    by_id = {c["id"]: c for c in cases}
    outcomes = report["outcomes"]
    ids = [o["id"] for o in outcomes if not o.get("checks", {}).get("pass")]
    if not ids or len({o["id"] for o in outcomes}) != len(outcomes):
        raise ValueError("No failures or duplicate parent outcomes")
    for outcome in outcomes:
        if by_id.get(outcome["id"]) != outcome["reference"]:
            raise ValueError("Parent case/reference differs")
    selected = [c for c in cases if c["id"] in set(ids)]
    return selected, {
        "kind": "prior_automatic_check_failures_only",
        "parent_directory": str(directory.resolve()),
        "parent_results_sha256": report_hash,
        "parent_index_sha256": digest(directory / "index.json"),
        "selected_case_ids": [c["id"] for c in selected],
        "limitation": "Paired diagnostic subset only; prior successes are not retested. Not a new full-benchmark score.",
    }


def metadata_value(metadata):
    field = metadata["field"]
    docs = metadata["outline"]["documents"]
    if field == "course_title":
        return [d["title"] for d in docs]
    if len(docs) != 1:
        return None
    chapters = docs[0]["chapters"]
    if field == "chapter_count":
        return len(chapters)
    if field == "chapter_titles":
        return [{"number": c["number"], "title": c["title"]} for c in chapters]
    chapter = next((c for c in chapters if c["number"] == metadata["chapter_number"]), None)
    if chapter and field == "chapter_title":
        return chapter["title"]
    if chapter and field == "chapter_sections":
        return chapter["sections"]
    return None


def explicit_selection(value, cases):
    ids = value.split(",")
    known = {c["id"] for c in cases}
    if not ids or len(set(ids)) != len(ids) or any(i not in known for i in ids):
        raise ValueError("Explicit selection contains unknown, duplicate or empty case IDs")
    selected = [c for c in cases if c["id"] in set(ids)]
    return selected, {
        "kind": "operator_selected_diagnostic_subset",
        "selected_case_ids": [c["id"] for c in selected],
        "limitation": "Explicit subset, including semantic failures automatic checks missed; not a full-suite accuracy score.",
    }


def reference_checks(case, response, source):
    """Part-aware routing, source anchors and metadata checks, not meaning-based grading."""
    body = response.model_dump(mode="json")
    actual = (
        body["parts"]
        if body["answer_kind"] == "multi"
        else [body | {"excerpt_ids": [e["id"] for e in body["excerpts"]]}]
    )
    details = []
    for expected, part in zip(case["parts"], actual):
        own = [e for e in body["excerpts"] if e["id"] in part.get("excerpt_ids", [])]
        kind = expected["kind"] if expected["kind"] in {"metadata", "content"} else "content"
        checks = {
            "status_matches": expected["status"] == part["status"],
            "kind_matches": kind == part["answer_kind"],
        }
        anchors = [a for fact in expected["facts"] for a in fact["evidence"]]
        checks["reference_anchors_in_own_evidence"] = all(
            any(
                a["quote"] in e["text"]
                and e["citation"]["chapter_number"] == a["chapter"]
                and e["citation"]["section_kind"] == a["section"]
                for e in own
            )
            for a in anchors
        )
        if expected["status"] == "answered" and kind == "metadata":
            metadata = part.get("metadata")
            checks["metadata_field_matches"] = bool(metadata) and (
                metadata["field"] == expected["metadata_field"]
                and metadata["chapter_number"] == expected.get("chapter_number")
            )
            checks["metadata_value_matches"] = bool(metadata) and (
                metadata_value(metadata) == expected.get("expected_value")
            )
            docs = metadata["outline"]["documents"] if metadata else []
            checks["metadata_source_matches"] = len(docs) == 1 and all(
                docs[0].get(k) == source.get(k) for k in ("document_id", "version", "source_sha256")
            )
        elif expected["status"] != "answered":
            checks["no_source_claim_payload"] = not own and part.get("explanation") is None
        if part.get("explanation"):
            from hrlearnium.schemas import Explanation

            checks["answer_script_matches"] = explanation_language_matches(
                Explanation.model_validate(part["explanation"]),
                [e for e in response.excerpts if e.id in part["excerpt_ids"]],
                case["expected_language"],
            )
        details.append(checks)
    checks = {
        "status_matches": body["status"] == case["expected_status"],
        "answer_kind_matches": body["answer_kind"] == case["expected_answer_kind"],
        "part_count_matches": len(actual) == len(case["parts"]),
        "excerpt_source_matches": all(
            e["citation"]["document_id"] == source["document_id"]
            and e["citation"]["version"] == source["version"]
            and e["citation"]["source_sha256"] == source["source_sha256"]
            for e in body["excerpts"]
        ),
        "parts": details,
    }
    contract = response_contract_checks(response, "explained")
    # answered_has_excerpts is deliberately false for valid metadata-only answers.
    contract_keys = (
        "response_mode_matches",
        "explanation_contract_valid",
        "explanation_citation_links_valid",
        "answer_canonical_render",
        "control_message_valid",
        "unique_excerpt_ids",
    )
    checks["response_contract_valid"] = all(contract[k] is not False for k in contract_keys)
    checks["pass"] = all(v for k, v in checks.items() if k != "parts") and all(
        all(p.values()) for p in details
    )
    checks["semantic_accuracy"] = "not_scored_requires_meaning_based_review"
    # Keep the strict historical pass definition, but do not confuse an output
    # shape/anchor diagnostic with an independently reviewed semantic error.
    checks["diagnostic_categories"] = {
        "status": checks["status_matches"] and all(p["status_matches"] for p in details),
        "grouping_and_routing": checks["part_count_matches"]
        and checks["answer_kind_matches"]
        and all(p["kind_matches"] for p in details),
        "reference_evidence_anchors": all(p["reference_anchors_in_own_evidence"] for p in details),
        "source_provenance": checks["excerpt_source_matches"]
        and all(p.get("metadata_source_matches", True) for p in details),
        "metadata_values": all(p.get("metadata_value_matches", True) for p in details),
        "response_contract": checks["response_contract_valid"],
    }
    return checks


def call(service, principal, course, payload):
    started = time.perf_counter()
    try:
        response = service.query(principal, course, QueryRequest.model_validate(payload))
        response = QueryResponse.model_validate(response.model_dump(mode="json"))
        return {
            "response": response.model_dump(mode="json"),
            "evaluation": response.evaluation,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": None,
        }
    except Exception as error:
        # Never persist exception strings or provider bodies (could contain credentials).
        return {
            "response": None,
            "evaluation": getattr(error, "evaluation_trace", None),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": {
                "type": type(error).__name__,
                "code": getattr(error, "diagnostic_code", "unclassified"),
                "provider_status": getattr(error, "provider_status", None),
            },
            "failed_request_usage": "unknown_may_have_incurred_charges",
        }


def summarize(outcomes, planned):
    evaluated = [o for o in outcomes if o.get("response")]
    part_failures = {
        o["id"]: [p for p in o["response"].get("parts", []) if p["status"] == "error"]
        for o in evaluated
    }
    part_error_cases = sum(bool(p) for p in part_failures.values())
    tracks = {}
    for name in sorted({o["track"] for o in outcomes}):
        own = [o for o in outcomes if o["track"] == name]
        tracks[name] = {
            "collected": len(own),
            "errors": sum(bool(o.get("error")) or bool(part_failures.get(o["id"])) for o in own),
            "status_matches": sum(o.get("checks", {}).get("status_matches", False) for o in own),
            "automatic_checks_passed": sum(o.get("checks", {}).get("pass", False) for o in own),
        }
    return {
        "planned_scored_cases": planned,
        "collected_cases": len(outcomes),
        "valid_response_bodies": len(evaluated),
        "successful_responses": len(evaluated) - part_error_cases,
        "operational_errors": sum(bool(o.get("error")) for o in outcomes) + part_error_cases,
        "request_exceptions": sum(bool(o.get("error")) for o in outcomes),
        "cases_with_part_errors": part_error_cases,
        "failed_parts": sum(len(p) for p in part_failures.values()),
        "setup_errors": sum(bool(o.get("setup_error")) for o in outcomes),
        "unattempted_cases": planned - len(outcomes),
        "status_matches": sum(o.get("checks", {}).get("status_matches", False) for o in outcomes),
        "automatic_checks_passed": sum(o.get("checks", {}).get("pass", False) for o in outcomes),
        "actual_statuses": dict(Counter(o["response"]["status"] for o in evaluated)),
        "failed_case_ids": [o["id"] for o in outcomes if not o.get("checks", {}).get("pass")],
        "latency_ms": latency_summary([o["latency_ms"] for o in evaluated]),
        "tracks": tracks,
        "runtime": runtime_summary(outcomes),
        "reference_review": "proposed_not_owner_approved",
        "semantic_accuracy": None,
        "semantic_review_complete": False,
        "diagnostic_category_counts": {
            name: sum(
                o.get("checks", {}).get("diagnostic_categories", {}).get(name) is True
                for o in outcomes
            )
            for name in (
                "status",
                "grouping_and_routing",
                "reference_evidence_anchors",
                "source_provenance",
                "metadata_values",
                "response_contract",
            )
        },
    }


def report_markdown(outcomes, summary, path):
    lines = [
        "# Hybrid natural-student benchmark",
        "",
        "References: proposed. Automatic checks are diagnostic, not answer-quality scores.",
        "",
        f"Collected: {summary['collected_cases']}/{summary['planned_scored_cases']}. "
        f"Operational errors: {summary['operational_errors']}. "
        f"Status matches: {summary['status_matches']}. "
        f"All automatic checks passed: {summary['automatic_checks_passed']}.",
        "",
        "| ID | Track | Expected → actual | Automatic checks | Question | Answer |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    for o in outcomes:
        response = o.get("response") or {}
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [
                        o["id"],
                        o["track"],
                        o["expected_status"] + " → " + response.get("status", "ERROR"),
                        "pass" if o.get("checks", {}).get("pass") else "flag",
                        o["question"],
                        response.get("answer", json.dumps(o.get("error"))),
                    ],
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(cli, manifest, cases, settings, *, factory=create_assistant):
    count = len(cases) + sum(bool(c.get("previous_question")) for c in cases)
    if cli.max_query_turns != count:
        raise ValueError(f"Explicit --max-query-turns {count} required (not a USD limit)")
    cli.output.mkdir(parents=True, exist_ok=False)
    inputs = cli.output / "inputs"
    inputs.mkdir()
    for name in (
        "manifest.json",
        "cases.jsonl",
        "assisted-pairs.jsonl",
        "source-audit.json",
        "grading.md",
    ):
        (inputs / name).write_bytes((cli.pack / name).read_bytes())
    expected = configuration(settings)
    original_fingerprint = fingerprint(settings)
    index = {
        "kind": "natural_student_embedded_benchmark",
        "created_at": datetime.now(UTC).isoformat(),
        "authorization": "Operator supplied --execute and an explicit query-turn cap; collection only, not reference approval",
        "mode": cli.mode,
        "planned_scored_cases": len(cases),
        "planned_query_turns": count,
        "max_query_turns": count,
        "attempted_query_turns": 0,
        "hidden_warmups": 0,
        "outer_retries": 0,
        "assisted_diagnostic_cases_included": 0,
        "reference_review": manifest["review_status"],
        "human_review_complete": False,
        "dataset_sha256": digest(cli.pack / "cases.jsonl"),
        "manifest_sha256": digest(cli.pack / "manifest.json"),
        "runner_sha256": digest(Path(__file__)),
        "configuration": expected,
        "source": manifest["source"],
        "captures": [],
        "pending_request": None,
        "status": "preflight",
    }
    index["selected_case_ids"] = [c["id"] for c in cases]
    if getattr(cli, "selection", None):
        index["selection"] = cli.selection
        write_json(inputs / "selection.json", cli.selection, exclusive=True)
        index["selection_sha256"] = digest(inputs / "selection.json")
    checkpoint = cli.output / "index.json"
    write_json(checkpoint, index, exclusive=True)
    service, outcomes = None, []
    try:
        # Isolate benchmark conversation writes; never reindex or mutate the LMS source DB.
        snapshot = cli.output / "benchmark.sqlite3"
        with sqlite3.connect(
            settings.database_path.resolve().as_uri() + "?mode=ro", uri=True
        ) as src:
            with sqlite3.connect(snapshot) as dst:
                src.backup(dst)
        run_settings = settings.model_copy(update={"database_path": snapshot})
        service = factory(run_settings)
        principal = Principal(
            sub="natural-benchmark-" + uuid4().hex,
            tenant_id=manifest["source"]["tenant"],
            course_ids=[manifest["source"]["course"]],
            scopes=["query"],
        )
        course = manifest["source"]["course"]
        options = service.retrieval_options(principal, course)
        write_json(cli.output / "preflight.json", options.model_dump(mode="json"), exclusive=True)
        selected = next(o for o in options.modes if o.mode == cli.mode)
        if not selected.available:
            raise ValueError("Requested mode unavailable: " + "; ".join(selected.reasons))
        index["status"] = "running"
        consecutive_errors = 0
        conversations = set()
        for case in cases:
            if (
                fingerprint(CoreSettings()) != original_fingerprint
                or configuration(settings) != expected
            ):
                raise ValueError("Settings or pipeline changed during collection")
            if (
                digest(cli.pack / "cases.jsonl") != index["dataset_sha256"]
                or digest(cli.pack / "manifest.json") != index["manifest_sha256"]
                or digest(Path(__file__)) != index["runner_sha256"]
            ):
                raise ValueError("Benchmark inputs/runner changed during collection")
            source, _ = read_source(snapshot, principal.tenant_id, course)
            if source != manifest["source"]:
                raise ValueError("Snapshot source differs from audited references")
            outcome = {
                "id": case["id"],
                "track": case["track"],
                "question": case["question"],
                "expected_status": case["expected_status"],
                "reference": case,
                "has_previous_question": bool(case.get("previous_question")),
            }
            conversation_id = None
            for turn in ["setup", "scored"] if case.get("previous_question") else ["scored"]:
                if index["attempted_query_turns"] >= count:
                    raise ValueError("Query-turn cap exhausted")
                if index["attempted_query_turns"]:
                    time.sleep(cli.delay_seconds)
                question = case["previous_question"] if turn == "setup" else case["question"]
                payload = request_payload(question, cli.mode, conversation_id)
                index["pending_request"] = {"id": case["id"], "turn": turn}
                index["attempted_query_turns"] += 1
                write_json(checkpoint, index)
                result = call(service, principal, course, payload)
                # Persist the paid result BEFORE any parsing/grading/config assertions.
                name = f"{case['id']}--{turn}.json"
                write_json(cli.output / name, {"payload": payload, **result}, exclusive=True)
                index["captures"].append(
                    {
                        "id": case["id"],
                        "turn": turn,
                        "report": name,
                        "sha256": digest(cli.output / name),
                    }
                )
                index["pending_request"] = None
                write_json(checkpoint, index)
                response = result["response"]
                if response:
                    if response["retrieval_mode"] != cli.mode:
                        raise ValueError("Actual retrieval mode differs from requested mode")
                    config = dict(response["evaluation"]["configuration"])
                    config.pop("retrieval_mode", None)
                    if config != expected:
                        raise ValueError(
                            "Actual response configuration differs from frozen configuration"
                        )
                    if conversation_id and response["conversation_id"] != conversation_id:
                        raise ValueError("Follow-up conversation changed")
                    if not conversation_id:
                        if response["conversation_id"] in conversations:
                            raise ValueError("Conversation reused across independent cases")
                        conversations.add(response["conversation_id"])
                    if turn == "setup":
                        conversation_id = response["conversation_id"]
                if turn == "setup":
                    outcome.update(
                        setup_response=response,
                        setup_evaluation=result["evaluation"],
                        setup_error=result["error"],
                        setup_latency_ms=result["latency_ms"],
                    )
                    if response:
                        ref = case["setup_reference"]
                        setup_case = {
                            "parts": [ref],
                            "expected_status": ref["status"],
                            "expected_answer_kind": ref["kind"]
                            if ref["kind"] in {"metadata", "content"}
                            else "content",
                            "expected_language": case["expected_language"],
                        }
                        outcome["setup_checks"] = reference_checks(
                            setup_case, QueryResponse.model_validate(response), source
                        )
                    else:
                        outcome.update(
                            response=None,
                            evaluation=None,
                            error=result["error"],
                            latency_ms=0,
                            evaluation_status="setup_failed",
                            scored_turn_attempted=False,
                        )
                else:
                    outcome.update(
                        result,
                        evaluation_status="evaluated" if response else "error",
                        scored_turn_attempted=True,
                    )
                    if response:
                        outcome["checks"] = reference_checks(
                            case, QueryResponse.model_validate(response), source
                        )
                        if outcome.get("setup_checks", {}).get("pass") is False:
                            outcome["checks"]["pass"] = False
                if result["error"]:
                    break
            outcomes.append(outcome)
            summary = summarize(outcomes, len(cases))
            write_json(
                cli.output / "results.json",
                {"metadata": index, "summary": summary, "outcomes": outcomes},
            )
            report_markdown(outcomes, summary, cli.output / "answers.md")
            consecutive_errors = consecutive_errors + 1 if outcome.get("error") else 0
            print(
                json.dumps(
                    {
                        "completed_cases": len(outcomes),
                        "query_turns": index["attempted_query_turns"],
                        "id": case["id"],
                        "status": (outcome.get("response") or {}).get("status"),
                        "automatic_checks": outcome.get("checks", {}).get("pass"),
                        "error": outcome.get("error"),
                    }
                ),
                flush=True,
            )
            error = outcome.get("error") or {}
            if (
                error.get("provider_status") in {400, 401, 402, 403, 404, 429}
                or consecutive_errors >= 3
            ):
                index["status"] = "stopped"
                index["stop_reason"] = "provider_or_repeated_operational_failure"
                break
        else:
            index["status"] = "complete"
        index["finished_at"] = datetime.now(UTC).isoformat()
        write_json(checkpoint, index)
        write_json(
            cli.output / "results.json",
            {"metadata": index, "summary": summarize(outcomes, len(cases)), "outcomes": outcomes},
        )
        return index
    except Exception as error:
        index["status"] = "stopped"
        index["stop_reason"] = type(error).__name__
        write_json(checkpoint, index)
        raise
    finally:
        if service:
            service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT)
    parser.add_argument("--mode", choices=["hybrid"], default="hybrid")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-query-turns", type=int, default=0)
    parser.add_argument("--delay-seconds", type=float, default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--failed-from", type=Path, help="Rerun only flagged cases from this run"
    )
    selection.add_argument(
        "--case-ids", help="Comma-separated exact case IDs for an explicit subset"
    )
    cli = parser.parse_args()
    settings = CoreSettings()
    manifest, cases, _ = pack.validate_pack(cli.pack, settings.database_path)
    if cli.failed_from:
        cases, cli.selection = failed_selection(
            cli.failed_from, manifest, cases, cli.pack, cli.mode
        )
    if cli.case_ids:
        cases, cli.selection = explicit_selection(cli.case_ids, cases)
    if not cli.execute:
        print(
            json.dumps(
                {
                    "selected_cases": len(cases),
                    "mode": cli.mode,
                    "setup_turns": sum(bool(c.get("previous_question")) for c in cases),
                    "query_turns": len(cases)
                    + sum(bool(c.get("previous_question")) for c in cases),
                    "provider_requests_made": 0,
                    "reference_review": manifest["review_status"],
                }
            )
        )
        return
    if cli.output is None or not 0 <= cli.delay_seconds <= 60:
        parser.error("Execution requires a new output directory and delay between 0 and 60 seconds")
    result = execute(cli, manifest, cases, settings)
    raise SystemExit(0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
