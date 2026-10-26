"""Local contracts for model-classified answer groups; no arbitrary operations or source claims."""

import json

from hrlearnium.schemas import (
    QuestionPlan,
    RequestGroup,
    RequestPlan,
    RequestPlanVerification,
    RoutePlan,
)
from hrlearnium.text import normalize_persian


def validate_request_plan(plan, original: str, language: str, *, translate: bool) -> bool:
    return request_plan_error(plan, original, language, translate=translate) is None


def request_plan_error(plan, original: str, language: str, *, translate: bool) -> str | None:
    """Stable, non-sensitive repair diagnostics, never a claim that the user is unclear."""
    if not isinstance(plan, RequestPlan) or plan.answer_language != language:
        return "invalid_plan_or_language"
    if plan.needs_clarification:
        return "clarification_with_groups" if plan.groups else None
    if not plan.groups or sum(len(g.requirements) for g in plan.groups) > 8:
        return "invalid_requirement_count"
    end, seen = 0, set()
    for group in plan.groups:
        start = original.find(group.original_text, end)
        if start < 0 or group.original_text in seen:
            return "original_text_not_exact_ordered_unique"
        end = start + len(group.original_text)
        seen.add(group.original_text)
        if len(set(group.requirements)) != len(group.requirements):
            return "duplicate_requirements"
        if group.intent == "metadata":
            if group.kind != "single" or group.metadata_field is None:
                return "invalid_metadata_group"
            targeted = group.metadata_field in {"chapter_title", "chapter_sections"}
            if targeted != (group.chapter_number is not None):
                return "invalid_chapter_target"
        elif group.metadata_field is not None or group.chapter_number is not None:
            return "content_has_metadata_target"
        if not translate and group.persian_query is not None:
            return "unexpected_translation"
        if translate and group.intent == "content" and group.persian_query is None:
            return "missing_translation"
    return None


def verified_request_plan(check) -> bool:
    return isinstance(check, RequestPlanVerification) and all(check.model_dump().values())


def search_plan(group) -> QuestionPlan:
    return QuestionPlan(
        search_query=group.search_query,
        persian_query=group.persian_query,
        requirements=group.requirements,
    )


def course_routing_context(outline, limit=8000):
    """Bounded, authorized headings only. This is untrusted context, not executable instructions."""
    result = {"documents": [], "truncated": False}
    used = 0
    for document in outline.documents:
        item = {"title": document.title[:200], "chapters": []}
        cost = len(json.dumps(item, ensure_ascii=False))
        if used + cost > limit:
            result["truncated"] = True
            break
        used += cost
        result["documents"].append(item)
        for chapter in document.chapters:
            entry = {"number": chapter.number, "title": chapter.title[:200]}
            cost = len(json.dumps(entry, ensure_ascii=False))
            if used + cost > limit:
                result["truncated"] = True
                break
            item["chapters"].append(entry)
            used += cost
    return result


def _matching_text(text):
    """Return a matching-only representation plus offsets into the unchanged original."""
    letters, positions = [], []
    for index, char in enumerate(text):
        for value in normalize_persian(char):
            value = value if value.isalnum() else " "
            if value == " " and (not letters or letters[-1] == " "):
                continue
            letters.append(value)
            positions.append(index)
        if char.isspace() or char == "\u200c":
            if letters and letters[-1] != " ":
                letters.append(" ")
                positions.append(index)
    return "".join(letters).rstrip(), positions


def canonical_route(route, original):
    """Restore source spelling locally; never fuzzy-match, remove a group or invent a topic."""
    if not isinstance(route, RoutePlan):
        raise ValueError("invalid_route")
    groups, end = [], 0
    normalized, positions = _matching_text(original)
    for group in route.groups:
        if len(route.groups) == 1:
            # One group must answer the WHOLE message, not a planner's shortened version.
            text = original
        else:
            start = original.find(group.text, end)
            if start >= 0:
                text, end = original[start : start + len(group.text)], start + len(group.text)
            else:
                needle, _ = _matching_text(group.text)
                offset = next((i for i, p in enumerate(positions) if p >= end), len(normalized))
                start = normalized.find(needle, offset) if needle else -1
                # Require full word boundaries; matching a substring inside another word
                # could silently alter entities or negation.
                stop = start + len(needle)
                if (
                    start < 0
                    or (start and normalized[start - 1] != " ")
                    or (stop < len(normalized) and normalized[stop] != " ")
                ):
                    raise ValueError("unaligned_original_text")
                begin, end = positions[start], positions[stop - 1] + 1
                text = original[begin:end]
        from hrlearnium.course_metadata import metadata_request
        from hrlearnium.language import strip_language_directive

        exact = metadata_request(strip_language_directive(text))
        if exact:
            # Full-match structural lookups also guard multipart groups against
            # an otherwise approved content-route mistake.
            group = group.model_copy(
                update={
                    "intent": "metadata",
                    "metadata_fields": [exact[0]],
                    "chapter_number": exact[1],
                }
            )
        fields = list(dict.fromkeys(group.metadata_fields))
        if group.intent == "content":
            fields, number = [], None
        else:
            if not fields:
                raise ValueError("missing_metadata_operation")
            if (
                exact
                and exact[0] == "chapter_titles"
                and "chapter_count" in fields
                and "chapter_titles" not in fields
            ):
                fields.append("chapter_titles")
            number = group.chapter_number
            if any(f in {"chapter_title", "chapter_sections"} for f in fields) and number is None:
                raise ValueError("missing_chapter_target")
            if not any(f in {"chapter_title", "chapter_sections"} for f in fields):
                number = None
            # A chapter-title listing already contains the count. Never discard titles.
            if "chapter_titles" in fields and "chapter_count" in fields:
                fields.remove("chapter_count")
        groups.append(
            group.model_copy(
                update={"text": text, "metadata_fields": fields, "chapter_number": number}
            )
        )
    return RoutePlan(groups=groups)


def compile_route(route, language, *, translate):
    """Requirements, output language and operation bounds are owned by the server."""
    groups = []
    for group in route.groups:
        for field in group.metadata_fields if group.intent == "metadata" else [None]:
            groups.append(
                RequestGroup(
                    original_text=group.text,
                    kind="single",
                    intent=group.intent,
                    metadata_field=field,
                    chapter_number=group.chapter_number
                    if field in {"chapter_title", "chapter_sections"}
                    else None,
                    search_query=group.search_query,
                    persian_query=group.persian_query if translate and field is None else None,
                    requirements=[group.text],
                )
            )
    if not 1 <= len(groups) <= 8:
        raise ValueError("too_many_operations")
    return RequestPlan(answer_language=language, needs_clarification=False, groups=groups)


def original_request_plan(question, language):
    """Safe degraded path: preserve the ENTIRE request and all normal evidence checks.

    It may lose independent/metadata routing, but never executes a fabricated plan,
    drops a clause, supplies an answer without evidence, or calls a plan fault ambiguity.
    """
    return RequestPlan(
        answer_language=language,
        needs_clarification=False,
        groups=[
            RequestGroup(
                original_text=question,
                kind="joint",
                intent="content",
                metadata_field=None,
                chapter_number=None,
                search_query=question,
                persian_query=None,
                requirements=[question],
            )
        ],
    )
