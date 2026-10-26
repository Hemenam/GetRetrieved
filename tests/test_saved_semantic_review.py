"""Review arithmetic/validation only; no provider calls or private fixtures."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_saved_natural_answers as review  # noqa: E402


def rating(verdict="pass"):
    value = {
        "s": [2] * 8,
        "verdict": verdict,
        "note": "Explicit reviewer judgment.",
        "e": ["policy"],
    }
    if verdict == "minor":
        value["s"][6] = 1
    elif verdict == "fail":
        value["s"][2] = 1
    return value


@pytest.mark.parametrize("bad", [True, 3, -1, "2", 1.0])
def test_scores_require_real_integers_not_truthy_values(bad):
    value = rating()
    value["s"][0] = bad
    with pytest.raises(ValueError, match="dimension"):
        review.validate_rating(value, set())


@pytest.mark.parametrize("dimension", [0, 2, 3, 4, 5, 7])
def test_automatic_or_manual_pass_cannot_hide_a_core_deficit(dimension):
    value = rating("minor")
    value["s"][dimension] = 1
    with pytest.raises(ValueError, match="core-quality"):
        review.validate_rating(value, set())


def test_nonapplicable_social_dimensions_do_not_receive_free_full_credit():
    value = rating()
    value["s"] = [2, 2, None, None, None, None, 2, None]
    before = copy.deepcopy(value)
    review.validate_rating(value, set())
    assert value == before


def test_unknown_evidence_and_missing_rationale_are_rejected():
    value = rating()
    value["e"] = ["made-up-source"]
    with pytest.raises(ValueError, match="Unknown source"):
        review.validate_rating(value, set())
    value["e"], value["note"] = ["policy"], " "
    with pytest.raises(ValueError, match="rationale"):
        review.validate_rating(value, set())


def test_parts_and_bad_setups_are_not_averaged_out():
    good, bad = rating(), rating("fail")
    rows = [
        {
            "id": "a",
            "reference": {"parts": [{"status": "answered"}, {"status": "refused"}]},
            "response": {"parts": [{"status": "answered"}, {"status": "refused"}]},
            "manual_review": good | {"parts": [good, good]},
        },
        {
            "id": "b",
            "reference": {"parts": [{"status": "answered"}, {"status": "answered"}]},
            "response": {"parts": [{"status": "answered"}, {"status": "error"}]},
            "manual_review": bad | {"parts": [good, bad]},
        },
        {
            "id": "c",
            "reference": {"previous_question": "setup"},
            "manual_review": good | {"setup": bad},
        },
        {
            "id": "d",
            "reference": {"previous_question": "setup"},
            "manual_review": rating("minor") | {"setup": good},
        },
    ]
    result = review.additional_metrics(rows)
    assert result["all_parts_acceptable"] == {"numerator": 1, "denominator": 2}
    assert result["supported_answer_part_quality"]["numerator"] == 2
    assert result["supported_answer_part_quality"]["denominator"] == 3
    assert result["supported_answer_part_delivery"] == {"numerator": 2, "denominator": 3}
    assert result["setup_quality"]["failed_setup_ids"] == ["c"]
    assert result["eligible_followup_quality"] == {"numerator": 1, "denominator": 1}
    assert result["end_to_end_acceptable_including_setup"] == 2


def test_incomplete_manual_review_cannot_be_published(monkeypatch, tmp_path):
    monkeypatch.setattr(review, "verify_inputs", lambda _: {})
    review.write(tmp_path / "responses.json", [])
    review.write(tmp_path / "source-passages.json", [])
    (tmp_path / "ratings.jsonl").touch()
    with pytest.raises(ValueError, match="exactly one explicit"):
        review.render(tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_outputs_are_exclusive_and_preserve_existing_review(tmp_path):
    target = tmp_path / "summary.json"
    review.write(target, {"old": "review"})
    with pytest.raises(FileExistsError):
        review.write(target, {"new": "review"})
    assert review.read(target) == {"old": "review"}
