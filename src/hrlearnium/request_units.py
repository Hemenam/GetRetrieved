"""Conservative original-text request boundaries and bounded unit routing."""

import re

from hrlearnium.course_metadata import metadata_request, page_location_request
from hrlearnium.language import strip_language_directive
from hrlearnium.schemas import RequestGroup, RequestPlan, UnitRoutes
from hrlearnium.text import normalize_persian


def request_units(question):
    # Keep quoted questions (e.g. a textbook's 'magic question') and parentheses
    # inside their request; split only complete, independent requests.
    pieces, start, quote, depth = [], 0, None, 0
    pairs = {"«": "»", '"': '"', "“": "”"}
    for i, char in enumerate(question):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in pairs:
            quote = pairs[char]
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif not depth and char in "؟?\n;؛.!":
            # Do not cut decimal numbers or a version/abbreviation mid-token.
            if (
                char == "."
                and i
                and i + 1 < len(question)
                and question[i - 1].isalnum()
                and question[i + 1].isalnum()
            ):
                continue
            if question[start : i + 1].strip():
                pieces.append(question[start : i + 1].strip())
            start = i + 1
    if question[start:].strip():
        pieces.append(question[start:].strip())
    result = []
    for piece in pieces:
        # A coordinating new interrogative or an explicitly recognizable metadata
        # prefix can be separated without asking a model to retype the text.
        boundaries = (
            []
            if metadata_request(piece)
            else [m.start() for m in re.finditer(r"\s+(?:and|و)\s+", piece, re.I)]
        )
        cuts = [0]
        for boundary in boundaries:
            left, right = piece[cuts[-1] : boundary].strip(), piece[boundary:].strip()
            right_clean = re.sub(r"^(?:and|و)\s+", "", right, flags=re.I)
            dependent = re.match(
                r"(?:what are (?:they|their)|how (?:does|do) (?:it|they)|چه نام|چگونه با|چه ارتباط)",
                right_clean,
                re.I,
            )
            if dependent:
                continue
            if (
                metadata_request(left)
                or (
                    re.search(
                        r"(?:\b(?:mean|means|called|is|are)\b|چیست|چیه|دارد|داره)\s*[,،]?$",
                        left,
                        re.I,
                    )
                )
                or (
                    re.match(r"(?:what|how|who|which|explain|define|describe)\b", right_clean, re.I)
                    and re.search(r"\b(?:explain|define|describe|is|are|mean)\b", left, re.I)
                )
            ):
                cuts.append(boundary)
        cuts.append(len(piece))
        result.extend(piece[a:b].strip() for a, b in zip(cuts, cuts[1:]) if piece[a:b].strip())
    # Do not turn language/format directives, greetings or a dependent follow-up
    # sentence into independent content requests.
    merged = []
    from hrlearnium.policy import conversational_reply, is_bare_followup

    for piece in result:
        plain = re.sub(r"^(?:و|and|بعد|then|also)\s+", "", piece, flags=re.I)
        request_like = bool(
            re.search(
                r"[؟?]|(?:چیست|چیه|چی بود|چه بود|کدام|چقدر|چه |چرا |چگونه |چند |توضیح بده|را بگو|رو بگو|بنویس|explain|what |how |who |which |list |describe )",
                plain,
                re.I,
            )
        )
        if merged and (
            not strip_language_directive(piece).strip(" .?!؟")
            or is_bare_followup(plain)
            or not request_like
            or re.match(
                r"^(?:آن را|این را|آن ها را|it should|make it|(?:how|why|what) (?:does|is|do|are) (?:it|they|this|that)\b|explain (?:it|this|that)\b|answer in|respond in|لطفا کوتاه|کوتاه جواب|مقایسه شان|چه ارتباطی)",
                normalize_persian(plain),
            )
        ):
            begin = question.find(merged[-1])
            end = question.find(piece, begin + len(merged[-1]))
            merged[-1] = (
                question[begin : end + len(piece)]
                if begin >= 0 and end >= 0
                else merged[-1] + " " + piece
            )
        elif conversational_reply(piece) is None or len(result) == 1:
            merged.append(re.sub(r"^(?:و|and)\s+", "", piece, flags=re.I))
    return merged or [question]


def is_course_overview(question):
    text = normalize_persian(question)
    return bool(
        re.search(
            r"(?:جمع بندی|خلاصه|نمای کلی|overview|summary|summarize).{0,80}(?:دوره|course)|(?:دوره|course).{0,80}(?:چه چیزهایی|چه موضوعاتی|what.*teach|what.*cover)",
            text,
            re.I,
        )
    )


def compile_units(units, proposed, language, *, translate=False, original_question=None):
    if len(units) > 8:
        return RequestPlan(answer_language=language, needs_clarification=True, groups=[])
    routes = {r.unit_index: r for r in proposed.units} if isinstance(proposed, UnitRoutes) else {}
    if isinstance(proposed, UnitRoutes) and (
        len(routes) != len(proposed.units) or set(routes) != set(range(len(units)))
    ):
        raise ValueError("unit_indices_not_complete_unique")
    from hrlearnium.request_modifiers import attach_modifiers

    if any(
        r.parent_unit_index is not None
        and (
            r.intent != "content"
            or r.metadata_fields
            or r.chapter_number is not None
            or r.persian_query is not None
        )
        for r in routes.values()
    ):
        raise ValueError("modifier_must_not_add_metadata_or_search_task")
    attached = attach_modifiers(
        original_question if original_question is not None else " ".join(units),
        units,
        {i: routes[i].parent_unit_index if i in routes else None for i in range(len(units))},
    )
    groups = []
    for item in attached:
        i, text = item.unit_indices[0], item.text
        route = routes.get(i)
        exact = metadata_request(strip_language_directive(text))
        fields = (
            [exact[0]]
            if exact
            else route.metadata_fields
            if route and route.intent == "metadata"
            else []
        )
        number = exact[1] if exact else route.chapter_number if route else None
        if page_location_request(text):
            # Pagination is unavailable in the current DOCX content contract.
            # Keep it a content-location refusal, not a fabricated metadata value.
            fields, number = [], None
        # Definitions of a named concept cannot become a title lookup merely
        # because that concept happens to appear in a chapter heading.
        normalized = normalize_persian(text)
        if not exact and re.search(
            r"(?:دوره|متن).{0,20}(?:می گوید|می گه|یاد می دهد|آموزش)|(?:course|text).{0,20}(?:say|teach)",
            normalized,
            re.I,
        ):
            fields, number = [], None
        if not exact and not re.search(
            r"فصل|دوره|مدرس|پیش نیاز|صفحه|chapter|course|instructor|prerequisite|page|certificate",
            normalized,
            re.I,
        ):
            fields, number = [], None
        for field in fields or [None]:
            groups.append(
                RequestGroup(
                    original_text=text,
                    kind="single",
                    intent="metadata" if field else "content",
                    metadata_field=field,
                    chapter_number=number
                    if field in {"chapter_title", "chapter_sections"}
                    else None,
                    search_query=text,
                    persian_query=route.persian_query
                    if route and translate and not field
                    else None,
                    requirements=[text],
                )
            )
    if len(groups) > 8:
        return RequestPlan(answer_language=language, needs_clarification=True, groups=[])
    return RequestPlan(answer_language=language, needs_clarification=False, groups=groups)


def evidence_obligations(question):
    text = normalize_persian(question)
    if is_course_overview(question):
        return [
            "Give a representative overview of learning themes across the whole course, not just its first few chapters. Combine facts concisely but do not discard major distinct themes."
        ]
    if re.search(r"همیشه|همه|\b(?:always|all|every)\b|هر کار", text):
        return [
            "Identify the applicable general rule.",
            "Identify its relevant boundary, exception or example, including any stated condition that must remain true.",
            "Answer whether the learner's blanket extrapolation is justified. A stated rule or example can support a negative answer; do not refuse merely because the proposed universal rule is not present.",
        ]
    if re.search(r"فرق|تفاوت|مقایسه|compare|difference|contrast", text):
        return [
            "Give the requested facts about the first item.",
            "Give the requested facts about the second item.",
            "Explain their actual difference without borrowing claims from one item's source for the other.",
        ]
    return ["Answer exactly what was requested; omit unrelated background and advice."]
