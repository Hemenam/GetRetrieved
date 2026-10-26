"""Scored Markdown exports preserve captures and never invent review grades."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_scored_answers as exporter  # noqa: E402


def test_cells_preserve_unicode_and_lines_without_breaking_tables():
    assert exporter.cell("سلام | <script>\nخط دوم & سوم") == (
        "سلام &#124; &lt;script&gt;<br>خط دوم &amp; سوم"
    )


def test_loads_existing_judgments_without_inventing_scores(tmp_path):
    path = tmp_path / "review.md"
    path.write_text(
        "| `01-introduction` | Required fact | G | P | R | Existing note. |\n",
        encoding="utf-8",
    )
    review = exporter.load_judgments(path)
    assert review["coverage-01-introduction"] == {
        "grades": {"full_context": "G", "hybrid": "P", "hybrid_rerank": "R"},
        "note": "Existing note.",
    }


@pytest.mark.parametrize("grade", ["unknown", ""])
def test_unrecognized_grade_is_rejected(tmp_path, grade):
    path = tmp_path / "review.md"
    path.write_text(f"| `01-introduction` | Fact | G | {grade} | R | Note |\n", encoding="utf-8")
    with pytest.raises(ValueError):
        exporter.load_judgments(path)


def test_render_includes_all_three_complete_answers_and_original_failure():
    case_id = "coverage-01-introduction"
    reports = {
        mode: {
            "outcomes": [
                {
                    "id": case_id,
                    "question": "پرسش کامل؟",
                    "gold": {"required_facts": [{"text": "معنای مرجع"}]},
                    "response": {"answer": f"{mode}\nپاسخ کامل [1]"},
                    "capture_phase": "original",
                }
            ]
        }
        for mode in exporter.MODES
    }
    reviews = {
        case_id: {
            "grades": {"full_context": "F", "hybrid": "G", "hybrid_rerank": "P"},
            "note": "Original refusal remains.",
        }
    }
    rendered = exporter.render(reports, reviews, created_date="2026-10-20")
    assert "پرسش کامل؟" in rendered
    for mode in exporter.MODES:
        assert exporter.cell(f"{mode}\nپاسخ کامل [1]") in rendered
    assert "0/5 (F)" in rendered and "5/5 (G)" in rendered and "4/5 (P)" in rendered
    assert "Original refusal remains." in rendered
    with pytest.raises(ValueError, match="match exactly"):
        exporter.render(reports, {}, created_date="2026-10-20")
