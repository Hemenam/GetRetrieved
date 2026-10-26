"""Local reviewer consistency and literal-quotation checks, not semantic proof."""

import re

from hrlearnium.text import normalize_persian, tokenize


def review_contract_issues(review, statement_count, requirement_count):
    """Reject contradictory review data; never turn a rejection into acceptance."""
    issues = []
    rows = review.statements
    coverage = review.verification.coverage
    if sorted(r.statement_index for r in rows) != list(range(statement_count)):
        issues.append("statement_checks_must_cover_each_statement_once")
    if sorted(c.requirement_index for c in coverage) != list(range(requirement_count)):
        issues.append("coverage_must_identify_each_requirement_once")
    for row in rows:
        if row.supported != bool(row.span_ids) or len(set(row.span_ids)) != len(row.span_ids):
            issues.append("statement_support_and_spans_disagree")
    for check in coverage:
        if (
            check.covered != bool(check.statement_indices)
            or any(i >= statement_count for i in check.statement_indices)
            or len(set(check.statement_indices)) != len(check.statement_indices)
        ):
            issues.append("coverage_and_answering_statements_disagree")
    details_pass = (
        all(review.intent.model_dump().values())
        and review.verification.all_question_parts_identified
        and len(rows) == statement_count
        and all(r.supported for r in rows)
        and len(coverage) == requirement_count
        and all(c.covered for c in coverage)
    )
    if review.verification.supported != details_pass:
        issues.append("overall_support_disagrees_with_detailed_checks")
    reasons = [r.strip() for r in review.rejection_reasons if r.strip()]
    if not details_pass and not reasons:
        issues.append("rejection_requires_a_specific_reason")
    if details_pass and reasons:
        issues.append("passing_review_contains_rejection_reasons")
    return list(dict.fromkeys(issues))


def conservative_rejection(review, issues):
    """Retain detailed rejection votes if only the positive aggregate is wrong.

    This may only turn overall True into False. It cannot approve a draft, remove
    a negative vote, invent semantic evidence, or repair malformed coverage.
    """
    allowed = {
        "overall_support_disagrees_with_detailed_checks",
        "rejection_requires_a_specific_reason",
    }
    if not issues or set(issues) - allowed or not review.verification.supported:
        return None
    reasons = [
        f"Reviewer rejected statement {r.statement_index}."
        for r in review.statements
        if not r.supported
    ]
    reasons += [
        f"Reviewer rejected intent: {key}."
        for key, value in review.intent.model_dump().items()
        if not value
    ]
    reasons += [
        f"Reviewer did not cover requirement {c.requirement_index}."
        for c in review.verification.coverage
        if not c.covered
    ]
    if not review.verification.all_question_parts_identified:
        reasons.append("Reviewer did not identify all original question parts.")
    if not reasons:
        return None
    result = review.model_copy(deep=True)
    result.verification.supported = False
    result.rejection_reasons = review.rejection_reasons or reasons[:8]
    return result


def unsupported_literal_quotes(text, sources):
    """Check same-script reported quotations in their OWN sources.

    Not a paraphrase/translation checker. Punctuation, marks, digits and harmless
    Persian spelling/joiner differences are normalized; words and negation remain.
    Short quoted labels are left to semantic review. No source text is modified.
    """
    quotes = re.finditer(r'«([^»]+)»|“([^”]+)”|"([^"\n]+)"', text)
    failures = []
    for match in quotes:
        quote = next(g for g in match.groups() if g is not None)
        words = tokenize(quote)
        if len(words) < 3:
            continue
        prefix = normalize_persian(text[max(0, match.start() - 100) : match.start()])
        if not re.search(
            r"\b(?:می پرسد|می گوید|پرسید|گفت|بپرسید|بگویید|asks?|asked|says?|said|quotes?)\b",
            prefix,
        ):
            continue
        if re.search(r"\b(?:not|never|نمی|نگفت|نپرسید)\b", prefix):
            continue
        persian = bool(re.search(r"[\u0600-\u06ff]", quote))
        comparable = [s for s in sources if bool(re.search(r"[\u0600-\u06ff]", s.text)) == persian]
        if not comparable:  # Cross-language wording needs semantic review, not exact matching.
            continue
        needle = " " + " ".join(words) + " "
        if not any(needle in " " + " ".join(tokenize(s.text)) + " " for s in comparable):
            failures.append(quote)
    return failures
