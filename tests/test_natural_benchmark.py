"""Natural question pack preparation/guardrails only; never invoke a model or server."""

import copy
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_natural_benchmark as pack  # noqa: E402


@pytest.fixture
def dataset():
    return pack.validate_pack(pack.DEFAULT)


def test_comprehensive_pack_counts_statuses_languages_and_no_claimed_approval(dataset):
    manifest, cases, assisted = dataset
    assert len(cases) == 160 and len(assisted) == 12
    assert len({c["id"] for c in cases + assisted}) == 172
    assert len({c["question"] for c in cases + assisted}) == 172
    assert len(manifest["tracks"]) == 10
    assert Counter(c["expected_status"] for c in cases) == {
        "answered": 126,
        "refused": 22,
        "partial": 6,
        "clarification": 3,
        "conversation": 3,
    }
    assert Counter(c["expected_language"] for c in cases) == {"fa": 145, "en": 15}
    assert not manifest["execution_authorized"]
    assert {c["review_status"] for c in cases + assisted} == {"proposed"}
    assert {c["reviewer"] for c in cases + assisted} == {""}
    assert {c["split"] for c in cases + assisted} == {"development"}
    assert all(c["expected_answer_kind"] in {"content", "metadata", "multi"} for c in cases)


def test_all_44_teaching_units_covered_without_chapter_location_hints(dataset):
    _, cases, _ = dataset
    coverage = [c for c in cases if c["track"] == "source_coverage"]
    assert len(coverage) == 44
    assert {
        (c["source_coverage_target"]["chapter"], c["source_coverage_target"]["section"])
        for c in coverage
    } == {
        (n, s) for n in range(1, 12) for s in ("introduction", "explanation", "example", "exercise")
    }
    for case in cases:
        assert not case["location_assisted"]
        if all(p["kind"] != "metadata" for p in case["parts"]):
            assert not re.search(
                r"\bchapter\s+\w+|فصل\s+(?:اول|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم|یازدهم|[۰-۹0-9]+)",
                case["question"],
                re.I,
            )


def test_assisted_pairs_keep_identical_gold_and_cover_every_chapter(dataset):
    _, cases, assisted = dataset
    originals = {c["id"]: c for c in cases}
    chapters = set()
    for pair in assisted:
        original = originals[pair["paired_natural_id"]]
        assert pair["location_assisted"] and original["question"] in pair["question"]
        assert pair["parts"] == original["parts"]
        assert pair["question_family_id"] == original["question_family_id"]
        chapters.add(original["source_coverage_target"]["chapter"])
    assert chapters == set(range(1, 12))
    assert pack.plan(cases)["query_turns_three_modes"] == 510
    assert pack.plan(cases, assisted, include_assisted=True)["query_turns_three_modes"] == 546
    assert pack.plan(cases)["network_requests_made"] == 0


def test_followups_have_separate_setup_gold_and_never_leak_it_to_api_payload(dataset):
    _, cases, _ = dataset
    setups = [c for c in cases if c.get("previous_question")]
    assert len(setups) == 10
    for case in cases:
        payload = pack.student_payload(case)
        assert payload == {
            "question": case["question"],
            "response_mode": "explained",
            "include_evaluation": True,
        }
        assert (
            not {"parts", "setup_reference", "expected_chapters", "facts", "source_coverage_target"}
            & payload.keys()
        )
        adapted = pack.api_case(case)
        assert adapted["review_status"] == "proposed"
        if case.get("previous_question"):
            assert adapted["previous_question"] == case["previous_question"]
            assert case["setup_reference"]["status"] in {"answered", "conversation"}


def test_partial_cases_preserve_positive_and_negative_parts_and_all_refused_is_not_partial(dataset):
    _, cases, _ = dataset
    partials = [c for c in cases if c["expected_status"] == "partial"]
    assert len(partials) == 6
    for case in partials:
        statuses = {p["status"] for p in case["parts"]}
        assert "answered" in statuses and "refused" in statuses
    negative_bundle = [
        c for c in cases if len(c["parts"]) > 1 and {p["status"] for p in c["parts"]} == {"refused"}
    ]
    assert len(negative_bundle) == 1 and negative_bundle[0]["expected_status"] == "refused"


def test_exact_anchor_audit_and_structural_values_have_provenance(dataset):
    manifest, cases, assisted = dataset
    audit = json.loads((pack.DEFAULT / "source-audit.json").read_text(encoding="utf-8"))
    assert audit["source"] == manifest["source"]
    assert audit["all_anchors_present"] and audit["binding_count"] == len(audit["bindings"])
    assert audit["binding_count"] > 300
    for binding in audit["bindings"]:
        assert binding["source_end"] - binding["source_start"] == len(binding["quote"])
        assert (
            0
            <= binding["source_start"]
            < binding["source_end"]
            <= manifest["source"]["canonical_character_count"]
        )
        assert binding["document_id"] == manifest["source"]["document_id"]
    for case in cases + assisted:
        for part in case["parts"]:
            if part["kind"] == "metadata":
                assert (part["status"] == "answered") == (part["expected_value"] is not None)
            if part["kind"] == "content" and part["status"] == "answered":
                assert all(f["evidence"] for f in part["facts"])


@pytest.mark.parametrize(
    "artifact", ["cases.jsonl", "review.md", "source-audit.json", "assisted-pairs.jsonl"]
)
def test_artifact_edits_are_detected_and_never_silently_overwritten(tmp_path, artifact):
    destination = tmp_path / "pack"
    shutil.copytree(pack.DEFAULT, destination)
    file = destination / artifact
    file.write_text(file.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed artifact"):
        pack.validate_pack(destination)
    with pytest.raises(ValueError, match="Changed artifact"):
        pack.build(destination, Path("unused.db"), "tenant", "course", refresh_draft=True)
    assert file.read_text(encoding="utf-8").endswith("changed")


def test_existing_and_approved_snapshots_cannot_be_replaced(tmp_path):
    destination = tmp_path / "pack"
    shutil.copytree(pack.DEFAULT, destination)
    with pytest.raises(FileExistsError):
        pack.build(destination, Path("unused.db"), "tenant", "course")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_status"] = "approved"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Never replace"):
        pack.build(destination, Path("unused.db"), "tenant", "course", refresh_draft=True)


@pytest.mark.parametrize("fault", ["locator", "unbound", "status", "metadata"])
def test_bad_reference_contracts_are_rejected(dataset, fault):
    manifest, cases, _ = dataset
    case = (
        copy.deepcopy(next(c for c in cases if c["track"] == "course_metadata"))
        if fault == "metadata"
        else copy.deepcopy(cases[0])
    )
    if fault == "locator":
        case["question"] = "طبق فصل اول، " + case["question"]
    elif fault == "unbound":
        case["parts"][0]["facts"][0]["evidence"] = []
    elif fault == "status":
        case["expected_status"] = "partial"
    else:
        case["parts"][0]["expected_value"] = ["invented course title"]
    with pytest.raises(ValueError):
        pack.bind_and_validate([case], [], manifest["source"])


def test_review_contains_all_questions_and_never_claims_new_owner_approval(dataset):
    _, cases, assisted = dataset
    review = (pack.DEFAULT / "review.md").read_text(encoding="utf-8")
    questions = (pack.DEFAULT / "questions.md").read_text(encoding="utf-8")
    for case in cases:
        assert case["id"] in review and case["question"] in review
        assert case["id"] in questions and case["question"] in questions
    for pair in assisted:
        assert pair["id"] in review
    assert "All references are **proposed**" in review
    assert "Current policy versus product aspirations" in (pack.DEFAULT / "grading.md").read_text(
        encoding="utf-8"
    )


def test_changed_source_revision_is_not_accepted(dataset, monkeypatch):
    manifest, _, _ = dataset
    source = copy.deepcopy(manifest["source"])
    source["version"] += 1
    monkeypatch.setattr(pack, "read_source", lambda *a: (source, []))
    with pytest.raises(ValueError, match="Source revision changed"):
        pack.validate_pack(pack.DEFAULT, Path("unused.db"))


def test_private_source_reproduces_all_generated_artifacts(tmp_path):
    database = os.environ.get("HRLEARNIUM_TEST_BENCHMARK_DB")
    if not database:
        pytest.skip("Set HRLEARNIUM_TEST_BENCHMARK_DB for the private read-only source audit")
    destination = tmp_path / "reproduced"
    pack.build(destination, Path(database), "customer-demo", "captain-storm")
    for name in pack.GENERATED:
        assert (destination / name).read_bytes() == (pack.DEFAULT / name).read_bytes()
    pack.validate_pack(destination, Path(database))
