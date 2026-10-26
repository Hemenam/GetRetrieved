"""Offline coverage-pack checks, not chatbot answer-quality measurements."""

import importlib.util
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "chapter_coverage_audit", ROOT / "scripts/audit_chapter_coverage.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
BENCH_SPEC = importlib.util.spec_from_file_location(
    "chapter_coverage_benchmark", ROOT / "scripts/benchmark.py"
)
bench = importlib.util.module_from_spec(BENCH_SPEC)
BENCH_SPEC.loader.exec_module(bench)
PACK = ROOT / "evals/chapter-coverage-v1"


def synthetic():
    cases, passages = [], []
    start = 0
    for index, section in enumerate(audit.SECTIONS):
        text = f"Synthetic {section} evidence."
        passages.append(
            {
                "id": f"p{index}",
                "chapter_number": 1,
                "chapter_title": "Synthetic chapter",
                "section_kind": section,
                "paragraph_start": index + 1,
                "paragraph_end": index + 1,
                "source_start": start,
                "source_end": start + len(text),
                "text": text,
            }
        )
        start += len(text) + 1
        cases.append(
            {
                "id": f"coverage-01-{section}",
                "expected_status": "answered",
                "expected_chapters": [1],
                "expected_language": "fa",
                "category": "direct",
                "required_quotes": [text],
                "required_facts": [{"id": "fact", "text": "Meaning", "evidence_quotes": [text]}],
                "review_status": "proposed",
                "reviewer": "",
                "split": "development",
            }
        )
    return cases, passages


def test_real_pack_is_complete_proposed_and_approval_blocked():
    manifest, profiles = bench.load_manifest(PACK / "manifest.json")
    cases = profiles[0]["cases"]
    assert len(cases) == 44
    assert {tuple(c["expected_chapters"]) for c in cases} == {(i,) for i in range(1, 12)}
    assert {c["id"] for c in cases} == {
        f"coverage-{chapter:02d}-{section}"
        for chapter in range(1, 12)
        for section in audit.SECTIONS
    }
    assert all(c["review_status"] == "proposed" and c["reviewer"] == "" for c in cases)
    assert all(c["expected_status"] == "answered" and not c.get("previous_question") for c in cases)
    plan = bench.plan(manifest, profiles)
    assert plan["network_requests_made"] == 0
    assert plan["scored_turns"] == plan["query_turns_upper_bound"] == 132
    assert plan["setup_turns_upper_bound"] == 0
    assert not plan["execution_ready"]
    assert len(plan["blockers"]) == 1 and "44" in plan["blockers"][0]


def test_audit_snapshot_matches_cases_and_review():
    cases = audit.load_cases(PACK / "cases.jsonl", None)
    snapshot = json.loads((PACK / "source-audit.json").read_text(encoding="utf-8"))
    assert snapshot["dataset_sha256"] == audit.sha((PACK / "cases.jsonl").read_bytes())
    assert snapshot["cases"] == snapshot["chapter_section_pairs"] == 44
    assert snapshot["chapters"] == 11
    assert snapshot["anchor_checks"] == 68
    assert snapshot["provider_calls_made"] == snapshot["network_requests_made"] == 0
    assert snapshot["owner_approved"] is snapshot["semantic_approval_inferred"] is False
    lookup = {b["case_id"]: b for b in snapshot["bindings"]}
    for case in cases:
        binding = lookup[case["id"]]
        required = set(case["required_quotes"]) | {
            quote for fact in case["required_facts"] for quote in fact["evidence_quotes"]
        }
        assert required == {a["quote"] for a in binding["anchors"]}
        for anchor in binding["anchors"]:
            assert binding["source_start"] <= anchor["source_start"] < anchor["source_end"]
            assert anchor["source_end"] <= binding["source_end"]
            assert anchor["source_end"] - anchor["source_start"] == len(anchor["quote"])
    assert (PACK / "review.md").read_text(encoding="utf-8") == audit.render_review(cases, snapshot)


def test_old_approved_pilot_is_unchanged():
    assert audit.sha((ROOT / "evals/pilot/cases.jsonl").read_bytes()) == (
        "2876c2142c64253e878551735ede137cbe13578ffaa7acf489ba3b1aff4e2a77"
    )


def test_owner_approved_snapshot_changes_only_approval_fields():
    proposed = audit.load_cases(PACK / "cases.jsonl", None)
    approved = audit.load_cases(PACK / "approved-20261019/cases.jsonl", None)
    assert len(approved) == len(proposed) == 44
    for old, new in zip(proposed, approved, strict=True):
        assert {k: v for k, v in old.items() if k not in {"review_status", "reviewer"}} == {
            k: v for k, v in new.items() if k not in {"review_status", "reviewer"}
        }
        assert new["review_status"] == "approved"
        assert new["reviewer"] == "Project owner (explicit approval in chat, 2026-10-19)"


def test_approved_snapshot_provenance_and_plan():
    folder = PACK / "approved-20261019"
    approval = json.loads((folder / "approval.json").read_text(encoding="utf-8"))
    assert approval["reviewer_role"] == "project_owner"
    assert approval["instructor_credentials_asserted"] is False
    assert approval["collected_response_grades_approved"] is False
    assert approval["expected_questions_and_facts_changed"] is False
    assert approval["proposed_dataset_sha256"] == audit.sha((PACK / "cases.jsonl").read_bytes())
    assert approval["approved_dataset_sha256"] == audit.sha((folder / "cases.jsonl").read_bytes())
    manifest, profiles = bench.load_manifest(folder / "manifest.json")
    assert set(approval["approved_case_ids"]) == {c["id"] for c in profiles[0]["cases"]}
    plan = bench.plan(manifest, profiles)
    assert plan["execution_ready"]
    assert plan["scored_turns"] == plan["query_turns_upper_bound"] == 132
    assert plan["setup_turns_upper_bound"] == plan["network_requests_made"] == 0


def test_exact_anchors_bind_to_correct_source_offsets():
    cases, passages = synthetic()
    bindings = audit.audit_cases(cases, passages)
    assert len(bindings) == 4
    for binding, passage in zip(bindings, passages, strict=True):
        assert binding["anchors"] == [
            {
                "quote": passage["text"],
                "source_start": passage["source_start"],
                "source_end": passage["source_end"],
            }
        ]


@pytest.mark.parametrize("mutation", ["wrong_quote", "wrong_chapter", "duplicate", "missing"])
def test_wrong_or_missing_evidence_targets_are_rejected(mutation):
    cases, passages = synthetic()
    if mutation == "wrong_quote":
        # This quote exists in another section, which must not be accepted.
        cases[0]["required_quotes"] = [passages[1]["text"]]
    elif mutation == "wrong_chapter":
        cases[0]["expected_chapters"] = [2]
    elif mutation == "duplicate":
        cases.append(deepcopy(cases[0]))
    else:
        cases.pop()
    with pytest.raises(ValueError):
        audit.audit_cases(cases, passages)


@pytest.mark.parametrize(
    "extra",
    [
        {"category": "multi_section"},
        {"previous_question": "Earlier question"},
        {"review_status": "approved", "reviewer": "Someone"},
        {"split": "held_out"},
    ],
)
def test_scope_and_proposed_gold_contract_is_enforced(extra):
    cases, passages = synthetic()
    cases[0].update(extra)
    with pytest.raises(ValueError):
        audit.audit_cases(cases, passages)


def test_source_must_have_all_four_sections():
    cases, passages = synthetic()
    with pytest.raises(ValueError, match="four sections"):
        audit.audit_cases(cases, passages[:-1])


def test_read_only_source_rejects_missing_db_without_creating_it(tmp_path):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        audit.read_source(path, "test", "test")
    assert not path.exists()


def test_read_source_checks_hashes_spans_and_preserves_database(tmp_path):
    _, passages = synthetic()
    full_text = "\n".join(p["text"] for p in passages)
    source_hash = audit.sha(full_text.encode("utf-8"))
    original = b"synthetic-file-bytes-not-a-real-docx"
    path = tmp_path / "source.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE documents(id,filename,version,tenant_id,course_id,active);
            CREATE TABLE revisions(document_id,version,title,full_text,source_sha256,
                                   file_sha256,original_docx,paragraph_count);
            CREATE TABLE passages(id,document_id,version,chapter_number,chapter_title,
                                  section_kind,paragraph_start,paragraph_end,
                                  source_start,source_end,text);
        """)
        db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?)", ("d", "test.docx", 1, "t", "c", 1))
        db.execute(
            "INSERT INTO revisions VALUES(?,?,?,?,?,?,?,?)",
            ("d", 1, "Synthetic", full_text, source_hash, audit.sha(original), original, 4),
        )
        for p in passages:
            db.execute(
                "INSERT INTO passages VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (p["id"], "d", 1, *[p[k] for k in list(p)[1:]]),
            )
    before = path.read_bytes()
    metadata, loaded = audit.read_source(path, "t", "c")
    assert loaded == passages
    assert metadata["source_sha256"] == source_hash
    assert metadata["passage_count"] == 4
    assert path.read_bytes() == before
    with sqlite3.connect(path) as db:
        db.execute("UPDATE revisions SET source_sha256='wrong'")
    with pytest.raises(ValueError, match="text hash"):
        audit.read_source(path, "t", "c")
