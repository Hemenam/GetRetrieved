"""Conservative bilingual structural lookup. Headings are data, never instructions."""

import re

from hrlearnium.ingestion import _ORDINALS
from hrlearnium.language import response_language
from hrlearnium.schemas import MetadataEvidence, QueryResponse
from hrlearnium.text import normalize_persian


def page_location_request(question):
    text = normalize_persian(question).lower().strip(" .?!؟")
    text = re.sub(r"^(?:(?:please|and|then|also|لطفا|و|بعد|همچنین)\s+)+", "", text)
    return bool(
        re.fullmatch(
            r"شماره صفحه.{0,250}(?:چند است|چنده|چیست)|(?:what|which) page (?:is|does|contains|covers|number|has|can).{0,250}|.{0,200}(?:on|in) which page",
            text,
            re.I,
        )
    )


def metadata_request(question: str) -> tuple[str, int | None] | None:
    text = normalize_persian(question).lower().strip(" .?!؟")
    text = re.sub(r"^(?:(?:please|and|then|also|لطفا|و|بعد|همچنین)\s+)+", "", text)
    # Full matches avoid swallowing a content question that accompanies a metadata question.
    for label, fa in (("first", "اول|اولین|نخستین"), ("last", "آخر|آخرین|پایانی")):
        target = rf"(?:{fa}) فصل|فصل (?:{fa})"
        context = r"(?: (?:این )?(?:دوره|مستند|سند|فایل|کتاب))?"
        if any(
            re.fullmatch(pattern, text)
            for pattern in (
                rf"(?:{target}){context} (?:چه (?:نامی?|عنوانی?|اسمی) (?:دارد|داره)|اسمش چیست|چی نام دارد)",
                rf"(?:نام|عنوان|اسم) (?:{target}){context} (?:چیست|چیه)",
                rf"what is (?:the )?(?:name|title) of (?:the )?{label} chapter(?: (?:in|of) (?:this|the) (?:course|document|file|book))?",
                rf"what is (?:the )?{label} chapter(?: (?:in|of) (?:this|the) (?:course|document|file|book))? (?:called|named)",
            )
        ):
            return f"{label}_chapter_title", None
    count = (
        r"how many chapters(?: (?:are there(?: in (?:this|the) course)?|does (?:this|the) course have))?",
        r"(?:what is )?(?:the )?(?:total )?number of chapters(?: in (?:this|the) course)?",
        r"(?:این )?(?:دوره|درس) چند (?:تا )?فصل (?:دارد|داره)",
        r"(?:تعداد فصل های (?:این )?دوره (?:چند است|چند تاست|چقدر است|چیست))",
        r"(?:این )?دوره چند فصل (?:است|هست)",
    )
    listing = (
        r"(?:همه )?(?:عنوان های|نام های|اسم های) فصل ها (?:را |رو )?(?:بگو|بنویس|لیست کن)",
        r"(?:list|name) (?:all )?(?:the )?chapters(?: (?:of|in) (?:this|the) course)?",
        r"what are (?:the )?(?:chapter (?:names|titles)|(?:names|titles) of (?:the )?chapters)(?: (?:of|in) (?:this|the) course)?",
        r"(?:what are (?:the )?chapters(?: (?:of|in) (?:this|the) course)?)",
        r"(?:show (?:me )?)?(?:the )?(?:course outline|table of contents)",
        r"(?:فهرست|سرفصل های|نام فصل های|عنوان فصل های) (?:این )?دوره(?: (?:را|رو))?(?: (?:بگو|بنویس|نشان بده|چیست|چیستند))?",
        r"(?:فصل های|فصول) (?:این )?دوره (?:کدامند|کدام اند|چیست|چیستند|چه هستند)",
    )
    for pattern in count:
        if re.fullmatch(pattern, text):
            return "chapter_count", None
        if re.fullmatch(
            pattern
            + r" (?:and what are (?:they|their (?:names|titles))|و (?:نام آن ?ها چیست|نامشان چیست|اسمشان چیست|چه هستند|(?:اسم|نام|عنوان) (?:همه )?(?:فصل ها|فصول) (?:چیست|چیستند)))",
            text,
        ):
            return "chapter_titles", None
    if any(re.fullmatch(pattern, text) for pattern in listing):
        return "chapter_titles", None
    for field, pattern in (
        (
            "course_title",
            r"(?:what is (?:the )?(?:course (?:name|title)|name of (?:this|the) course)|(?:نام|عنوان) (?:این )?دوره چیست)",
        ),
        (
            "instructor",
            r"(?:who (?:teaches (?:this|the) course|is (?:the )?(?:course )?instructor)|مدرس (?:این )?دوره (?:کیست|کیه))",
        ),
        (
            "duration",
            r"(?:how long is (?:this|the) course|what is (?:the )?course duration|(?:مدت|مدت زمان) (?:این )?دوره (?:چقدر است|چقدر هست|چیست))",
        ),
        (
            "prerequisites",
            r"(?:what are (?:the )?(?:course )?prerequisites|پیش نیاز(?:های)? (?:رسمی )?(?:این )?دوره (?:چیست|چیستند))",
        ),
    ):
        if re.fullmatch(pattern, text):
            return field, None
    number = r"(\d+|" + "|".join(_ORDINALS) + r")"
    targets = (
        (
            "chapter_title",
            r"(?:what is (?:the )?(?:name|title) of chapter "
            + number
            + r"|(?:نام|عنوان) فصل "
            + number
            + r" چیست)",
        ),
        (
            "chapter_sections",
            r"(?:what (?:sections|parts) are in chapter "
            + number
            + r"|how many sections (?:are there )?in chapter "
            + number
            + r"|فصل "
            + number
            + r" (?:چه بخش هایی دارد|چند بخش دارد))",
        ),
    )
    for field, pattern in targets:
        match = re.fullmatch(pattern, text)
        if match:
            raw = next(value for value in match.groups() if value is not None)
            return field, int(raw) if raw.isdigit() else _ORDINALS[raw]
    return None


def metadata_response(outline, request, *, request_id, conversation_id, mode, operation):
    field, number = operation
    persian = response_language(request.question) == "fa"
    documents = outline.documents
    answer = None
    if field in {"first_chapter_title", "last_chapter_title"}:
        # Resolve from this authorized snapshot, never the model's guess or a
        # hard-coded chapter count. Multiple documents have no unified order.
        if len(documents) == 1 and documents[0].chapters:
            chapter = documents[0].chapters[0 if field == "first_chapter_title" else -1]
            field, number = "chapter_title", chapter.number
    # Missing administrative fields are never inferred from textbook prose or a filename.
    if documents and field == "course_title":
        names = "؛ ".join(doc.title for doc in documents)
        answer = (
            f"عنوان مستندات واردشده: {names}" if persian else f"Imported document title(s): {names}"
        )
    elif documents and field in {"chapter_count", "chapter_titles"}:
        if all(doc.chapters for doc in documents) and not any(
            "Conflicting titles" in w for d in documents for w in d.warnings
        ):
            blocks = []
            for doc in documents:
                intro = (
                    f"در مستند «{doc.title}»، {len(doc.chapters)} فصل شناسایی شده است."
                    if persian
                    else f"The imported document “{doc.title}” has {len(doc.chapters)} recognized chapters."
                )
                if field == "chapter_titles":
                    intro += "\n" + "\n".join(f"{c.number}. {c.title}" for c in doc.chapters)
                blocks.append(intro)
            answer = "\n\n".join(blocks)
    elif len(documents) == 1 and field in {"chapter_title", "chapter_sections"}:
        doc = documents[0]
        chapter = next((c for c in doc.chapters if c.number == number), None)
        if chapter and not any("Conflicting titles" in w for w in doc.warnings):
            if field == "chapter_title":
                answer = f"{number}. {chapter.title}"
            else:
                names = {
                    "introduction": "مقدمه",
                    "explanation": "توضیح اصلی",
                    "example": "مثال",
                    "exercise": "تمرین",
                }
                sections = ", ".join(names.get(s, s) if persian else s for s in chapter.sections)
                answer = (
                    f"در فصل {number}، {len(chapter.sections)} بخش آموزشی شناسایی شده است: {sections}"
                    if persian
                    else f"Chapter {number} has {len(chapter.sections)} recognized teaching sections: {sections}"
                )
    supported = answer is not None
    if supported:
        caveat = (
            "این اطلاعات از ساختار مستندات واردشده استخراج شده است، نه فهرست رسمی LMS."
            if persian
            else "This describes imported documentation, not a verified LMS syllabus."
        )
        if len(documents) > 1:
            caveat += (
                " شمارش‌ها برای هر مستند جداست؛ تعداد کل فصل‌های دوره تأیید نشده است."
                if persian
                else " Counts are per document; a unified course chapter total is not established."
            )
        answer += "\n\n" + caveat
    else:
        answer = (
            "این اطلاعات در فرادادهٔ مستندات واردشده موجود یا قابل تأیید نیست."
            if persian
            else "This information is unavailable or cannot be verified from the imported course metadata."
        )
    return QueryResponse(
        request_id=request_id,
        status="answered" if supported else "refused",
        answer=answer,
        excerpts=[],
        conversation_id=conversation_id,
        reason_code="metadata_supported" if supported else "metadata_unavailable",
        retrieval_mode=mode,
        response_mode=request.response_mode,
        answer_kind="metadata",
        metadata=MetadataEvidence(
            field=field, chapter_number=number, language="fa" if persian else "en", outline=outline
        )
        if supported
        else None,
    )
