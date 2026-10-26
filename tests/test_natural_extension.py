"""V2 extends reference coverage, never approval, source access, or paid execution."""

import copy
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_natural_extension as extension  # noqa: E402

pack = extension.pack


@pytest.fixture
def dataset():
    return pack.validate_pack(extension.DEFAULT)


def test_v2_preserves_parent_cases_pairs_and_provenance(dataset):
    manifest, cases, assisted = dataset
    parent, original, original_pairs = pack.validate_pack(pack.DEFAULT)
    assert cases[:160] == original and assisted == original_pairs
    assert manifest["parent_manifest_sha256"] == pack.digest(pack.DEFAULT / "manifest.json")
    assert manifest["parent_cases_sha256"] == parent["cases_sha256"]
    assert len(cases) == 188 and len(assisted) == 12 and manifest["new_case_count"] == 28
    assert len({c["id"] for c in cases + assisted}) == 200
    assert len({c["question"] for c in cases + assisted}) == 200
    tracks = Counter(c["track"] for c in cases)
    assert tracks["course_metadata"] == 24 and tracks["multipart"] == 32
    assert Counter(c["expected_status"] for c in cases) == {
        "answered": 143,
        "refused": 27,
        "partial": 12,
        "clarification": 3,
        "conversation": 3,
    }
    assert {c["review_status"] for c in cases + assisted} == {"proposed"}
    assert not manifest["execution_authorized"]


def test_added_metadata_covers_allowed_fields_and_unknowns(dataset):
    _, cases, _ = dataset
    metadata = cases[160:172]
    assert len(metadata) == 12 and {c["track"] for c in metadata} == {"course_metadata"}
    assert {c["parts"][0]["metadata_field"] for c in metadata} == pack.ALLOWED_FIELDS
    for c in metadata:
        p = c["parts"][0]
        assert p["kind"] == "metadata"
        assert (p["expected_value"] is not None) == (c["expected_status"] == "answered")
    absent = next(c for c in metadata if c["parts"][0]["chapter_number"] == 99)
    assert absent["expected_status"] == "refused"


def test_multipart_has_ordered_spans_per_part_status_and_no_gold_in_payload(dataset):
    _, cases, _ = dataset
    multipart = cases[172:]
    assert len(multipart) == 16 and {c["track"] for c in multipart} == {"multipart"}
    assert {len(c["parts"]) for c in multipart} == {2, 3}
    for c in cases[160:]:
        cursor = 0
        for part in c["parts"]:
            start = c["question"].find(part["question_text"], cursor)
            assert start >= 0
            cursor = start + len(part["question_text"])
        payload = pack.student_payload(c)
        assert set(payload) == {"question", "response_mode", "include_evaluation"}
        assert "parts" not in payload
    assert any(c["expected_status"] == "partial" for c in multipart)
    all_refused = [c for c in multipart if {p["status"] for p in c["parts"]} == {"refused"}]
    assert len(all_refused) == 1 and all_refused[0]["expected_status"] == "refused"
    comparison = next(
        c for c in multipart if "joint_comparison_plus_independent_question" in c["behaviors"]
    )
    assert len(comparison["parts"]) == 2 and comparison["parts"][0]["grouping"] == "joint"
    assert len(cases[-1]["parts"]) == 2 and cases[-1]["parts"][1]["grouping"] == "joint"


def test_reordered_or_invented_part_spans_are_rejected(dataset):
    manifest, cases, _ = dataset
    c = copy.deepcopy(cases[172])
    c["parts"][0]["question_text"] = "an invented question"
    with pytest.raises(ValueError, match="exact ordered student span"):
        pack.bind_and_validate([c], [], manifest["source"])


def test_planning_is_offline_and_counts_setups_and_diagnostics(dataset):
    _, cases, assisted = dataset
    main = pack.plan(cases)
    assert main["query_turns_three_modes"] == 594
    assert main["network_requests_made"] == 0 and not main["execution_authorized"]
    assert pack.plan(cases, assisted, include_assisted=True)["query_turns_three_modes"] == 630


def test_dedicated_additions_review_contains_only_the_new_questions(dataset):
    _, cases, _ = dataset
    review = (extension.DEFAULT / "additions.md").read_text(encoding="utf-8")
    for case in cases[160:]:
        assert case["id"] in review and case["question"] in review
    assert "natural-001" not in review and "28 main cases" in review


def test_extension_refuses_to_overwrite_a_snapshot(tmp_path):
    output = tmp_path / "existing"
    shutil.copytree(extension.DEFAULT, output)
    with pytest.raises(FileExistsError):
        extension.build(output, Path("unused.db"))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_authorized"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Never replace"):
        extension.build(output, Path("unused.db"), refresh_draft=True)


def test_private_source_reproduces_v2_without_changing_parent(tmp_path):
    database = os.environ.get("HRLEARNIUM_TEST_BENCHMARK_DB")
    if not database:
        pytest.skip("Set HRLEARNIUM_TEST_BENCHMARK_DB for read-only private source validation")
    before = {name: pack.digest(pack.DEFAULT / name) for name in pack.GENERATED}
    output = tmp_path / "reproduced"
    extension.build(output, Path(database))
    for name in (*pack.GENERATED, "additions.md"):
        assert (output / name).read_bytes() == (extension.DEFAULT / name).read_bytes()
    assert before == {name: pack.digest(pack.DEFAULT / name) for name in pack.GENERATED}
    pack.validate_pack(output, Path(database))
