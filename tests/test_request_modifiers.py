"""Classification-index contracts, not paid or course-specific intent tests."""

from types import SimpleNamespace

import pytest

from hrlearnium.request_modifiers import attach_modifiers
from hrlearnium.request_units import request_units


@pytest.mark.parametrize(
    "question,units,parents,expected",
    [
        (
            "What is photosynthesis? Name its two stages briefly. What is gravity?",
            ["What is photosynthesis?", "Name its two stages briefly.", "What is gravity?"],
            {0: None, 1: None, 2: None},
            [(0,), (1,), (2,)],
        ),
        (
            "سه اصل یادگیری چیست؟ کنار هر اصل فقط یک توضیح کوتاه بنویس. نیرو چیست؟",
            ["سه اصل یادگیری چیست؟", "کنار هر اصل فقط یک توضیح کوتاه بنویس.", "نیرو چیست؟"],
            {0: None, 1: 0, 2: None},
            [(0, 1), (2,)],
        ),
        (
            "What are the stages? Put a short label beside each one. Use bullet points. What is mass?",
            [
                "What are the stages?",
                "Put a short label beside each one.",
                "Use bullet points.",
                "What is mass?",
            ],
            {0: None, 1: 0, 2: 0, 3: None},
            [(0, 1, 2), (3,)],
        ),
    ],
)
def test_supplied_classifications_preserve_independent_questions_and_attach_modifiers(
    question, units, parents, expected
):
    grouped = attach_modifiers(question, units, parents)
    assert [g.unit_indices for g in grouped] == expected
    assert sorted(i for g in grouped for i in g.unit_indices) == list(range(len(units)))
    assert all(g.text == question[g.source_start : g.source_end] for g in grouped)
    assert all(all(units[i] in g.text for i in g.unit_indices) for g in grouped)


def test_units_are_not_rewritten_and_original_separators_and_coordinator_survive():
    question = "What is mass?\n\n  and show the answer as two bullets.\nWhat is heat?"
    units = ["What is mass?", "show the answer as two bullets.", "What is heat?"]
    groups = attach_modifiers(question, units, {0: None, 1: 0, 2: None})
    assert groups[0].text == "What is mass?\n\n  and show the answer as two bullets."
    assert groups[1].text == "What is heat?"


def test_records_may_be_dicts_or_typed_objects_and_order_is_source_order():
    question, units = "Define X. Use a table.", ["Define X.", "Use a table."]
    records = [
        {"unit_index": 1, "parent_unit_index": 0, "intent": "content"},
        SimpleNamespace(unit_index=0, parent_unit_index=None),
    ]
    assert attach_modifiers(question, units, records)[0].text == question


def test_exact_repeated_units_align_left_to_right_without_reusing_a_previous_match():
    question = "Explain X? Briefly. Explain X? Briefly."
    units = ["Explain X?", "Briefly.", "Explain X?", "Briefly."]
    groups = attach_modifiers(question, units, {0: None, 1: 0, 2: None, 3: 2})
    assert groups[0].text == groups[1].text == "Explain X? Briefly."
    assert groups[0].source_end < groups[1].source_start


@pytest.mark.parametrize(
    "question",
    [
        'What does the quoted question "What changed?" mean? Use v2.1 notation.',
        "عبارت «چه شد؟» چه معنایی دارد؟ آن را کوتاه بگو.",
        "What is a 3.14 approximation? Explain it simply.",
    ],
)
def test_presegmented_quotes_decimal_versions_and_merged_dependents_stay_literal(question):
    units = request_units(question)
    groups = attach_modifiers(question, units, {i: None for i in range(len(units))})
    assert [g.text for g in groups] == units
    assert all(g.text == question[g.source_start : g.source_end] for g in groups)


@pytest.mark.parametrize(
    "parents,error",
    [
        ({0: None}, "complete_unique"),
        ({0: None, 1: None, 3: None}, "complete_unique"),
        ({0: None, 1: None, 2: None, 3: None}, "complete_unique"),
        ([{"unit_index": 0, "parent_unit_index": None}] * 3, "complete_unique"),
        ({0: None, 1: 3, 2: None}, "earlier_unit"),
        ({0: 1, 1: 0, 2: None}, "earlier_unit"),
        ({0: None, 1: 1, 2: None}, "earlier_unit"),
        ({0: None, 1: -1, 2: None}, "earlier_unit"),
        ({0: None, 1: True, 2: None}, "earlier_unit"),
        ({0: None, 1: 0, 2: 1}, "independent_root"),
        ({0: None, 1: None, 2: 0}, "crosses_independent"),
        ([{"unit_index": 0}] * 3, "missing_index_field"),
        ([{"unit_index": True, "parent_unit_index": None}] * 3, "complete_unique"),
    ],
)
def test_invalid_index_graphs_fail_closed_instead_of_merging_or_dropping_requests(parents, error):
    with pytest.raises(ValueError, match=error):
        attach_modifiers(
            "Define X. Define Y. Briefly.", ["Define X.", "Define Y.", "Briefly."], parents
        )


@pytest.mark.parametrize(
    "question,units,parents,error",
    [
        ("Define X. Briefly.", ["Define Z.", "Briefly."], {0: None, 1: 0}, "not_aligned"),
        ("Define X. Briefly.", ["Briefly.", "Define X."], {0: None, 1: 0}, "not_aligned"),
        ("Define X.", ["Define X.", "Define X."], {0: None, 1: 0}, "not_aligned"),
        ("Define X. Briefly.", [" Define X.", "Briefly."], {0: None, 1: 0}, "original_text"),
        (
            "Define X. Explain Y. Briefly.",
            ["Define X.", "Briefly."],
            {0: None, 1: 0},
            "unclassified_text",
        ),
        ("Define X.", [], {}, "count_out_of_bounds"),
        ("Define X.", "Define X.", {0: None}, "count_out_of_bounds"),
        (" ", ["Define X."], {0: None}, "original_request_required"),
    ],
)
def test_foreign_omitted_overlapping_or_malformed_units_never_become_new_group_text(
    question, units, parents, error
):
    with pytest.raises(ValueError, match=error):
        attach_modifiers(question, units, parents)


def test_bound_is_eight_and_input_collections_are_not_modified():
    units = [f"Question {i}?" for i in range(8)]
    parents = {i: None for i in range(8)}
    original_units, original_parents = list(units), dict(parents)
    assert len(attach_modifiers(" ".join(units), units, parents)) == 8
    assert units == original_units and parents == original_parents
    with pytest.raises(ValueError, match="count_out_of_bounds"):
        attach_modifiers(
            " ".join(units + ["Question 8?"]), units + ["Question 8?"], {i: None for i in range(9)}
        )
