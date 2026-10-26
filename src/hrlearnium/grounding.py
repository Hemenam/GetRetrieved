"""Request-local coverage contracts. Search plans are never factual evidence."""

import re

from hrlearnium.answer_contracts import unsupported_literal_quotes
from hrlearnium.models import ModelUnavailable
from hrlearnium.schemas import (
    AnswerReview,
    CompleteSelection,
    CompleteVerification,
    EvidenceBrief,
    Explanation,
    QuestionPlan,
    Selection,
    SpanAnswerReview,
)
from hrlearnium.source_rules import explicit_rule_conflict, source_status_preserved
from hrlearnium.source_spans import source_spans
from hrlearnium.text import normalize_persian, tokenize


def requested_value_kind(question):
    text = normalize_persian(question or "")
    if re.search(r"(?:اسم|نام|name).{0,100}(?:سخنگو|جانشین|spokesperson|successor)", text):
        return "person_name"
    if re.search(r"(?:چه|کدام|what|which).{0,60}(?:دارو|medication|drug)", text):
        return "medication"
    if re.search(r"شماره.{0,15}تماس|phone number|contact number", text):
        return "phone_number"
    return None


def filter_brief_rules(brief, candidates):
    if brief.status != "answered":
        return brief
    spans = {s["id"]: s for s in source_spans([c.passage for c in candidates])}
    sources = {c.passage.id: c.passage.text for c in candidates}
    valid = [
        f
        for f in brief.facts
        if not any(
            explicit_rule_conflict(f.text, sources.get(spans[s]["citation_id"], ""))
            for s in f.span_ids
            if s in spans
        )
    ]
    if not valid:
        raise ModelUnavailable("No verified source facts remain", code="invalid_evidence_brief")
    return brief.model_copy(update={"facts": valid})


def overview_brief(candidates, max_sources):
    """Source-literal, chapter-spread context for broad course summaries.

    Only use eligible introduction passages. Selection is structural, not an
    invented course summary or a bypass of authorization/relevance gates.
    """
    chapters = {}
    for candidate in candidates:
        p = candidate.passage
        if p.section_kind == "introduction" and p.chapter_number is not None:
            chapters.setdefault((p.chapter_number, p.chapter_title or ""), p)
    sources = [chapters[k] for k in sorted(chapters)]
    if not sources:
        return None
    count = min(max_sources, len(sources))
    indices = (
        [0] if count == 1 else [round(i * (len(sources) - 1) / (count - 1)) for i in range(count)]
    )
    return EvidenceBrief(
        status="answered",
        reason_code="supported",
        complete=True,
        facts=[
            {
                "text": sources[i].text[:1000],
                "span_ids": [s["id"] for s in source_spans([sources[i]])],
            }
            for i in indices
        ],
    )


def brief_selection(brief, candidates, *, max_excerpts, question=None):
    if not isinstance(brief, EvidenceBrief):
        raise ModelUnavailable("Invalid evidence brief", code="invalid_evidence_brief")
    if brief.status != "answered":
        if brief.facts or brief.complete or brief.reason_code == "supported":
            raise ModelUnavailable("Nonanswer brief contains facts", code="invalid_evidence_brief")
        return CompleteSelection(
            selection=Selection(status=brief.status, reason_code=brief.reason_code), coverage=[]
        )
    if not brief.complete or not brief.facts:
        return CompleteSelection(
            selection=Selection(status="refused", reason_code="insufficient_evidence"), coverage=[]
        )
    spans = {s["id"]: s for s in source_spans([c.passage for c in candidates])}
    ids = []
    for fact in brief.facts:
        if any(s not in spans for s in fact.span_ids):
            raise ModelUnavailable("Unknown fact source", code="invalid_evidence_brief")
        ids.extend(spans[s]["citation_id"] for s in fact.span_ids)
    ids = list(dict.fromkeys(ids))
    kind = requested_value_kind(question)
    if kind:
        sources = {c.passage.id: c.passage.text for c in candidates}
        valid = bool(brief.requested_values) and all(
            v.source_id in ids
            and v.value.strip()
            and v.value in sources.get(v.source_id, "")
            and not re.search(r"[؟?]|تعیین|مشخص نشده|ندارد|چه کسی", v.value)
            and (kind != "person_name" or not re.search(r"سخنگو|جانشین|مدیرعامل", v.value))
            and (kind != "phone_number" or len(re.sub(r"\D", "", v.value)) >= 5)
            for v in brief.requested_values
        )
        if not valid:
            return CompleteSelection(
                selection=Selection(status="refused", reason_code="insufficient_evidence"),
                coverage=[],
            )
    if brief.reason_code != "supported" or len(ids) > max_excerpts:
        raise ModelUnavailable(
            "Evidence brief exceeds source budget", code="answer_source_budget_exceeded"
        )
    return CompleteSelection(
        selection=Selection(status="answered", reason_code="supported", passage_ids=ids),
        coverage=[{"requirement_index": 0, "passage_ids": ids}],
    )


def supplement_general_rule(brief, candidates, question, max_excerpts):
    """A concrete example alone cannot establish a requested universal rule."""
    if brief.status != "answered" or not re.search(
        r"همیشه|همه|always|\ball\b|\bevery\b|هر کار|هر بار", normalize_persian(question)
    ):
        return brief
    spans = {s["id"]: s for s in source_spans([c.passage for c in candidates])}
    ids = {spans[s]["citation_id"] for f in brief.facts for s in f.span_ids if s in spans}
    from hrlearnium.schemas import EvidenceFact

    # For blanket-rule questions, an introductory objective is not the operative
    # rule. Replace an intro-only note with the ACTUAL explanation text, never
    # relabel the old claim as though another source supported it. The final
    # writer/reviewer must still establish what that text means for the request.
    replacements = {}
    for candidate in candidates:
        p = candidate.passage
        if p.id not in ids or p.section_kind != "introduction":
            continue
        rules = [
            c
            for c in candidates
            if c.passage.section_kind == "explanation"
            and c.passage.chapter_number == p.chapter_number
            and c.passage.chapter_title == p.chapter_title
        ]
        related = [
            f
            for f in brief.facts
            if any(spans[s]["citation_id"] == p.id for s in f.span_ids if s in spans)
        ]
        if len(rules) == 1 and all(
            {spans[s]["citation_id"] for s in f.span_ids if s in spans} == {p.id} for f in related
        ):
            rule = rules[0].passage
            replacements[p.id] = EvidenceFact(
                text=rule.text[:1000], span_ids=[s["id"] for s in source_spans([rule])]
            )
    if replacements:
        facts = []
        for fact in brief.facts:
            own = {spans[s]["citation_id"] for s in fact.span_ids if s in spans}
            changed = next((replacements[i] for i in own if i in replacements), fact)
            if changed not in facts:
                facts.append(changed)
        brief = brief.model_copy(update={"facts": facts})
        ids = {spans[s]["citation_id"] for f in facts for s in f.span_ids if s in spans}
    selected = [c for c in candidates if c.passage.id in ids]
    # Check each example's own chapter, not the entire mixed selection. An
    # exercise from a different chapter does not establish this example's rule.
    facts = list(brief.facts)
    for example in (c.passage for c in selected if c.passage.section_kind == "example"):
        same_chapter = [
            c
            for c in candidates
            if c.passage.chapter_number == example.chapter_number
            and c.passage.chapter_title == example.chapter_title
            and c.passage.section_kind == "explanation"
        ]
        if any(c.passage.id in ids for c in same_chapter) or not same_chapter:
            continue
        if len(ids) >= max_excerpts or len(facts) >= 8:
            break
        rule = same_chapter[0].passage
        facts.append(
            EvidenceFact(text=rule.text[:1000], span_ids=[s["id"] for s in source_spans([rule])])
        )
        ids.add(rule.id)
    return brief.model_copy(update={"facts": facts})


def wants_persian_search(question: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", question)) and not re.search(r"[آ-ی]", question)


def validate_plan(plan: QuestionPlan, *, translated: bool) -> None:
    if not isinstance(plan, QuestionPlan) or len(set(plan.requirements)) != len(plan.requirements):
        raise ModelUnavailable("Invalid question decomposition", code="invalid_question_plan")
    if translated and (not plan.persian_query or not re.search(r"[آ-ی]", plan.persian_query)):
        raise ModelUnavailable(
            "Missing Persian search translation", code="invalid_search_translation"
        )
    if not translated and plan.persian_query is not None:
        raise ModelUnavailable("Unexpected search translation", code="invalid_search_translation")


def search_variants(plan: QuestionPlan, original: str) -> list[str]:
    # Include the original intent even if a planner mistranslates or over-summarizes it.
    # Only strip a narrow trailing presentation request, never an actual course question.
    original = re.sub(
        r"(?:[.!?]\s*|\s+)(?:please\s+)?(?:answer|respond)\s+in\s+English[.!?\s]*$",
        "",
        original,
        flags=re.I,
    ).strip()
    values = [original, plan.search_query]
    if plan.persian_query:
        values.append(plan.persian_query)
    if len(plan.requirements) > 1:
        values.extend(plan.requirements)
    result, seen = [], set()
    for value in values:
        key = tuple(tokenize(value))
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def selection_covers_question(
    result: CompleteSelection,
    plan: QuestionPlan,
    *,
    candidate_ids: set[str] | None = None,
    max_excerpts: int = 8,
) -> Selection:
    if not isinstance(result, CompleteSelection):
        raise ModelUnavailable("Invalid complete evidence selection")
    selection = result.selection
    if (
        (
            selection.status == "answered"
            and (not selection.passage_ids or selection.reason_code != "supported")
        )
        or (selection.status != "answered" and selection.passage_ids)
        or len(selection.passage_ids) != len(set(selection.passage_ids))
        or len(selection.passage_ids) > max_excerpts
        or (candidate_ids is not None and not set(selection.passage_ids) <= candidate_ids)
    ):
        raise ModelUnavailable("Invalid evidence selection")
    if selection.status != "answered":
        if result.coverage:
            raise ModelUnavailable("Non-answer contains coverage claims")
        return selection
    indices = [part.requirement_index for part in result.coverage]
    if len(indices) != len(set(indices)) or any(i >= len(plan.requirements) for i in indices):
        raise ModelUnavailable("Invalid requirement evidence indices")
    selected = set(selection.passage_ids)
    for part in result.coverage:
        if (
            len(part.passage_ids) != len(set(part.passage_ids))
            or not set(part.passage_ids) <= selected
        ):
            raise ModelUnavailable("Requirement references unselected evidence")
    if set(indices) != set(range(len(plan.requirements))) or any(
        not part.passage_ids for part in result.coverage
    ):
        return Selection(status="refused", passage_ids=[], reason_code="insufficient_evidence")
    if {id_ for part in result.coverage for id_ in part.passage_ids} != selected:
        raise ModelUnavailable("Selection contains unexplained evidence")
    return selection


def explanation_covers_question(
    verification: CompleteVerification,
    plan: QuestionPlan,
    explanation: Explanation,
    selection: CompleteSelection,
) -> bool:
    if not isinstance(verification, CompleteVerification):
        raise ModelUnavailable("Invalid completeness verification")
    indices = [part.requirement_index for part in verification.coverage]
    if len(indices) != len(set(indices)) or any(i >= len(plan.requirements) for i in indices):
        raise ModelUnavailable("Invalid verification requirement indices")
    for part in verification.coverage:
        if len(part.statement_indices) != len(set(part.statement_indices)) or any(
            type(i) is not int or not 0 <= i < len(explanation.statements)
            for i in part.statement_indices
        ):
            raise ModelUnavailable("Invalid verification statement indices")
    if not verification.supported or not verification.all_question_parts_identified:
        return False
    if set(indices) != set(range(len(plan.requirements))):
        return False
    evidence = {part.requirement_index: set(part.passage_ids) for part in selection.coverage}
    return all(
        part.covered
        and part.statement_indices
        and any(
            evidence[part.requirement_index].intersection(explanation.statements[i].citation_ids)
            for i in part.statement_indices
        )
        for part in verification.coverage
    )


def review_supports_statements(review, explanation, excerpts):
    """Check exact support-spans and ownership, not just the reviewer's global boolean."""
    if isinstance(review, SpanAnswerReview):
        if not all(review.intent.model_dump().values()):
            return False
        indices = [p.statement_index for p in review.statements]
        if len(indices) != len(set(indices)) or set(indices) != set(
            range(len(explanation.statements))
        ):
            return False
        spans = {s["id"]: s for s in source_spans(excerpts)}
        for part in review.statements:
            own = explanation.statements[part.statement_index].citation_ids
            statement = explanation.statements[part.statement_index]
            if unsupported_literal_quotes(statement.text, [e for e in excerpts if e.id in own]):
                return False
            if (
                not part.supported
                or not part.span_ids
                or len(set(part.span_ids)) != len(part.span_ids)
            ):
                return False
            if any(
                identifier not in spans or spans[identifier]["citation_id"] not in own
                for identifier in part.span_ids
            ):
                return False
        return True
    if not isinstance(review, AnswerReview):
        raise ModelUnavailable("Invalid answer review")
    indices = [part.statement_index for part in review.statements]
    if len(indices) != len(set(indices)) or set(indices) != set(range(len(explanation.statements))):
        return False
    by_id = {e.id: e for e in excerpts}
    for part in review.statements:
        statement = explanation.statements[part.statement_index]
        if not part.supported or not part.evidence:
            return False
        for evidence in part.evidence:
            source = by_id.get(evidence.citation_id)
            if (
                source is None
                or evidence.citation_id not in statement.citation_ids
                or not evidence.quote.strip()
                or evidence.quote not in source.text
            ):
                return False
    return True


def prune_incidental_statements(draft, review, excerpts, plan):
    """Propose dropping rejected extras only; a fresh full review is still mandatory.

    Never delete a statement the reviewer assigned to a requested requirement.
    Missing coverage or intent failures require evidence reselection instead.
    """
    if not isinstance(review, SpanAnswerReview) or not all(review.intent.model_dump().values()):
        return None
    verification = review.verification
    if not verification.all_question_parts_identified:
        return None
    coverage = verification.coverage
    if (
        len(coverage) != len(plan.requirements)
        or {c.requirement_index for c in coverage} != set(range(len(plan.requirements)))
        or any(not c.covered or not c.statement_indices for c in coverage)
    ):
        return None
    rows = review.statements
    if len(rows) != len(draft.statements) or {r.statement_index for r in rows} != set(
        range(len(draft.statements))
    ):
        return None
    spans = {s["id"]: s for s in source_spans(excerpts)}
    by_id = {e.id: e for e in excerpts}
    bad = set()
    for row in rows:
        statement = draft.statements[row.statement_index]
        own = [by_id[i] for i in statement.citation_ids if i in by_id]
        if (
            not row.supported
            or not row.span_ids
            or any(
                s not in spans or spans[s]["citation_id"] not in statement.citation_ids
                for s in row.span_ids
            )
            or not source_status_preserved(statement.text, own)
            or any(explicit_rule_conflict(statement.text, e.text) for e in own)
            or unsupported_literal_quotes(statement.text, own)
        ):
            bad.add(row.statement_index)
    required = {i for c in coverage for i in c.statement_indices}
    if (
        not bad
        or bad & required
        or not required <= set(range(len(draft.statements)))
        or len(bad) == len(draft.statements)
    ):
        return None
    return Explanation(statements=[s for i, s in enumerate(draft.statements) if i not in bad])
