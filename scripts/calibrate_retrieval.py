"""Replay search floors against captured scores and a read-only, source-matched database.

No credentials, provider requests, fresh embeddings or semantic answer grades.
"""

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

from hrlearnium.retrieval import Candidate, SearchPassage, filter_relevance


def replay(
    report_path: Path, database: Path, floors: list[float], *, dataset: Path, min_bm25=1.0, limit=8
):
    if not floors or any(not math.isfinite(f) or not 0 <= f <= 1 for f in floors):
        raise ValueError("Cosine floors must be finite numbers between zero and one")
    if limit < 1 or not math.isfinite(min_bm25) or min_bm25 < 0:
        raise ValueError("Invalid candidate limit or BM25 floor")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not any("retrieved_candidates" in o.get("evaluation", {}) for o in report["outcomes"]):
        raise ValueError("This report has no raw retrieval scores to replay")
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != report["dataset_sha256"]:
        raise ValueError("Use the exact snapshotted dataset associated with this report")
    gold = {
        c["id"]: c
        for c in (
            json.loads(line)
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    metadata = report["metadata"]
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """SELECT p.id,p.text,p.chapter_number,p.chapter_title,p.section_kind
               FROM passages p JOIN documents d ON d.id=p.document_id AND d.version=p.version
               WHERE d.tenant_id=? AND d.course_id=? AND d.active=1""",
            (metadata["tenant_label"], metadata["course"]),
        ).fetchall()
    corpus = sorted(
        (r["chapter_number"] or 0, r["chapter_title"], r["section_kind"], r["text"]) for r in rows
    )
    fingerprint = hashlib.sha256(json.dumps(corpus, ensure_ascii=False).encode("utf-8")).hexdigest()
    passages = {
        r["id"]: SearchPassage(
            r["id"], r["text"], r["chapter_number"], r["chapter_title"], r["section_kind"]
        )
        for r in rows
    }
    fingerprints = {
        o.get("evaluation", {}).get("source_corpus_sha256")
        for o in report["outcomes"]
        if o.get("evaluation", {}).get("source_passage_count", 0) > 0
    }
    if fingerprints != {fingerprint} or not passages:
        raise ValueError("Captured scores do not match the current authorized source corpus")
    sweeps = []
    for floor in floors:
        cases = []
        for outcome in report["outcomes"]:
            ranked = []
            for item in outcome.get("evaluation", {}).get("retrieved_candidates", []):
                if item["id"] not in passages:
                    raise ValueError(
                        "Captured passage IDs are stale; do not guess evidence mappings"
                    )
                ranked.append(
                    Candidate(
                        passages[item["id"]],
                        item["lexical_score"],
                        item["semantic_score"],
                        item["rank_score"],
                    )
                )
            admitted = filter_relevance(ranked, min_cosine=floor, min_bm25=min_bm25)
            selected = admitted[:limit]
            case = gold[outcome["id"]]
            if any(
                case[key] != outcome[key]
                for key in ("question", "expected_status", "expected_chapters")
            ):
                raise ValueError("Report and dataset questions or expectations differ")
            quotes = set(case["required_quotes"])
            quotes.update(
                q
                for fact in outcome.get("gold", {}).get("required_facts", [])
                for q in fact["evidence_quotes"]
            )
            available = "\n".join(c.passage.text for c in selected)
            chapters = {c.passage.chapter_number for c in selected}
            cases.append(
                {
                    "id": outcome["id"],
                    "expected_status": outcome["expected_status"],
                    "admitted_candidates": len(admitted),
                    "retained_candidates": len(selected),
                    "source_coverage_complete": (
                        all(q in available for q in quotes)
                        and set(outcome["expected_chapters"]) <= chapters
                    )
                    if outcome["expected_status"] == "answered"
                    else None,
                }
            )
        positives = [c for c in cases if c["expected_status"] == "answered"]
        negatives = [c for c in cases if c["expected_status"] != "answered"]
        sweeps.append(
            {
                "min_cosine": floor,
                "answerable_cases": len(positives),
                "source_coverage_complete": sum(c["source_coverage_complete"] for c in positives),
                "missing_source_cases": [
                    c["id"] for c in positives if not c["source_coverage_complete"]
                ],
                "non_answer_cases": len(negatives),
                "non_answer_cases_exposed_to_candidates": sum(
                    c["retained_candidates"] > 0 for c in negatives
                ),
                "cases": cases,
            }
        )
    return {
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "source_corpus_sha256": fingerprint,
        "candidate_limit": limit,
        "min_bm25": min_bm25,
        "new_provider_calls": 0,
        "answer_quality_retested": False,
        "scope": "Cached query scores only; not translated-query retrieval, reranking or semantic grading",
        "limitations": "Candidate exposure on negatives is not a false-answer rate. Development tuning is not held-out validation.",
        "sweeps": sweeps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Exact dataset snapshot used by the report"
    )
    parser.add_argument("--database", type=Path, default=Path("data/hrlearnium.sqlite3"))
    parser.add_argument("--cosine-floors", type=float, nargs="+", default=[0.30, 0.25, 0.24, 0.20])
    parser.add_argument("--min-bm25", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    try:
        result = replay(
            args.report,
            args.database,
            args.cosine_floors,
            dataset=args.dataset,
            min_bm25=args.min_bm25,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error):
        # Reports can contain learner/source text; do not echo malformed input.
        parser.exit(
            2,
            "Calibration failed: check report schema, source fingerprint and passage revisions.\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
