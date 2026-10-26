"""Narrow, deterministic checks for explicit source instructions and hypothetical scope.

These are not a general entailment engine. Unmatched paraphrases still need model review.
"""

import re

from hrlearnium.text import normalize_persian, tokenize


def explicit_rule_conflict(text, source):
    if normalize_persian(text) in normalize_persian(source):
        return False
    rules, polarity = [], None
    for line in source.splitlines():
        value = normalize_persian(line).strip(" :؛.")
        if value in {"بایدها", "dos", "do", "must"}:
            polarity = True
        elif value in {"نبایدها", "donts", "don'ts", "do not", "must not"}:
            polarity = False
        elif value and polarity is not None:
            terms = tokenize(value)
            if len(terms) >= 3:
                rules.append((terms, polarity))
    words = tokenize(text)
    negative = {
        "نیست",
        "نباشد",
        "نباید",
        "نکنید",
        "نکنند",
        "نکنیم",
        "ممنوع",
        "عدم",
        "not",
        "never",
        "forbidden",
    }
    for terms, positive in rules:
        for i in range(len(words) - len(terms) + 1):
            if words[i : i + len(terms)] != terms:
                continue
            nearby = words[max(0, i - 3) : i] + words[i + len(terms) : i + len(terms) + 2]
            negated = bool(set(nearby) & negative)
            # Explicit list headings govern every item, not only the first one.
            # A later heading or explicit must/must-not instruction resets scope.
            headings = {
                "بایدها": True,
                "نبایدها": False,
                "dos": True,
                "donts": False,
                "باید": True,
                "نباید": False,
            }
            scope = next((headings[w] for w in reversed(words[:i]) if w in headings), None)
            if scope is not None:
                negated = negated or not scope
            if positive == negated:
                return True
    return False


def source_status_preserved(text, sources):
    if not sources or any(e.citation.section_kind not in {"example", "exercise"} for e in sources):
        return True
    return bool(
        re.search(
            r"مثال|مثلا|نمونه|فرضی|تمرین|سناریو|\b(?:example|hypothetical|exercise|practice|scenario)\b|\bfor instance\b",
            normalize_persian(text),
            re.I,
        )
    )


def attribute_source_scope(draft, sources, language):
    """Add server-owned scope labels before review, without changing factual prose.

    A scoped statement still needs full entailment/coverage review. This does not
    make an incorrect fact correct or establish any learner-company fact.
    """
    from hrlearnium.schemas import Explanation, GroundedStatement

    by_id = {e.id: e for e in sources}
    statements = []
    for statement in draft.statements:
        own = [by_id[i] for i in statement.citation_ids if i in by_id]
        if source_status_preserved(statement.text, own):
            statements.append(statement)
            continue
        exercise = any(e.citation.section_kind == "exercise" for e in own)
        if language == "fa":
            prefix = "طبق تمرین پیشنهادی دوره: " if exercise else "در مثال آموزشی دوره: "
        else:
            prefix = "The course exercise proposes: " if exercise else "In the course example: "
        text = prefix + statement.text
        statements.append(
            GroundedStatement(text=text, citation_ids=statement.citation_ids)
            if len(text) <= 1200
            else statement
        )
    return Explanation(statements=statements)
