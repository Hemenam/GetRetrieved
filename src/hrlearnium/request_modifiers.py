"""Bounded, source-literal attachment of classified presentation modifiers.

This module validates index relationships, not whether a clause is semantically
format-only. The classifier must make that decision from the original request.
No generated paraphrase becomes request authority and no unit can disappear.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AttachedUnit:
    text: str
    unit_indices: tuple[int, ...]
    source_start: int
    source_end: int


def _parents_by_index(parents, count):
    if isinstance(parents, Mapping):
        rows = list(parents.items())
    else:
        if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes)):
            raise ValueError("modifier_classification_must_be_indexed")
        rows = []
        for record in parents:
            if isinstance(record, Mapping):
                if "unit_index" not in record or "parent_unit_index" not in record:
                    raise ValueError("modifier_classification_missing_index_field")
                rows.append((record["unit_index"], record["parent_unit_index"]))
            else:
                if not hasattr(record, "unit_index") or not hasattr(record, "parent_unit_index"):
                    raise ValueError("modifier_classification_missing_index_field")
                rows.append((record.unit_index, record.parent_unit_index))
    indices = [index for index, _ in rows]
    if (
        any(type(index) is not int for index in indices)
        or len(indices) != count
        or set(indices) != set(range(count))
    ):
        raise ValueError("modifier_indices_not_complete_unique")
    result = dict(rows)
    for index, parent in result.items():
        if parent is not None and (type(parent) is not int or not 0 <= parent < index):
            raise ValueError("modifier_parent_must_be_earlier_unit")
        if parent is not None and result[parent] is not None:
            raise ValueError("modifier_parent_must_be_independent_root")
    return result


def _unit_spans(question, units):
    spans, cursor = [], 0
    for text in units:
        if not isinstance(text, str) or not text or text != text.strip():
            raise ValueError("modifier_units_must_be_nonempty_original_text")
        start = question.find(text, cursor)
        if start < 0:
            raise ValueError("modifier_unit_not_aligned_to_original_request")
        spans.append((start, start + len(text)))
        cursor = start + len(text)
    return spans


def _joinable_gap(text):
    # request_units may remove a leading coordinator from a following unit.
    # Retain it in the exact source slice, but never absorb an omitted clause.
    return (
        re.fullmatch(
            r"[\s،,؛;.!?؟:]*(?:(?:and|also|then|و|بعد)\b[\s،,؛;.!?؟:]*)?",
            text,
            flags=re.I,
        )
        is not None
    )


def attach_modifiers(question: str, units: Sequence[str], parents) -> list[AttachedUnit]:
    """Attach contiguous, classified modifiers to earlier independent units.

    ``parents`` is either ``{unit_index: parent_index_or_None}`` or a sequence
    of records exposing ``unit_index`` and ``parent_unit_index``. Every input
    unit must occur exactly once. A non-null parent means this unit was already
    classified as a presentation-only modifier, not as another content question.

    Parents must be independent roots. All attached units must be contiguous:
    attaching across another independent question, forward references, chains,
    duplicate/missing IDs and foreign text are rejected instead of guessed.
    Repeated exact text aligns left-to-right after the preceding unit's end.

    Return order follows root order; each group's text is one exact substring of
    ``question``. Offsets describe request text, never course evidence. Callers
    should fall back to their unchanged independent-unit path on invalid input.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("modifier_original_request_required")
    if (
        not isinstance(units, Sequence)
        or isinstance(units, (str, bytes))
        or not 1 <= len(units) <= 8
    ):
        raise ValueError("modifier_unit_count_out_of_bounds")
    relation = _parents_by_index(parents, len(units))
    spans = _unit_spans(question, units)
    groups = []
    for index in range(len(units)):
        parent = relation[index]
        if parent is None:
            groups.append([index])
            continue
        if not groups or groups[-1][0] != parent:
            raise ValueError("modifier_attachment_crosses_independent_unit")
        previous = groups[-1][-1]
        if not _joinable_gap(question[spans[previous][1] : spans[index][0]]):
            raise ValueError("modifier_attachment_contains_unclassified_text")
        groups[-1].append(index)
    return [
        AttachedUnit(
            text=question[spans[indices[0]][0] : spans[indices[-1]][1]],
            unit_indices=tuple(indices),
            source_start=spans[indices[0]][0],
            source_end=spans[indices[-1]][1],
        )
        for indices in groups
    ]
