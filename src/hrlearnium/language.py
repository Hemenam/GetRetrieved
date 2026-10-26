"""Request-scoped answer language, independent of rewritten/translated search text.

The final script check is conservative FA/EN detection, not proof of fluent language.
Only exact source quotations and source titles are exempted from that check.
"""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from hrlearnium.text import normalize_persian


@dataclass(frozen=True)
class ResponseContext:
    language: str
    original_message: str


_context: ContextVar[ResponseContext | None] = ContextVar("course_response_context", default=None)
_PERSIAN_WORD = r"[\u0621-\u063A\u0641-\u064A\u066E-\u066F\u0671-\u06D3]+"
_ACRONYMS = {"HR", "KPI", "IT", "AI", "LLM", "RAG", "GPT", "LMS", "API", "PDF", "DOCX", "HTML"}
_DIRECTIVE = re.compile(
    r"(?:(?:please\s+)?(?:answer|respond|reply)\s+in\s+(?P<english>English|Persian|Farsi)"
    r"|(?:لطفا\s+)?(?:به\s+)?(?:زبان\s+)?(?P<persian>انگلیسی|فارسی)\s+(?:پاسخ\s+بده(?:ید)?|جواب\s+بده(?:ید)?|بگو))",
    re.I,
)


def _outside_quotes(text):
    # Preserve positions for slicing a trailing directive, but do not treat quoted
    # textbook text/code as the student's own output-language instruction.
    return re.sub(
        r'```[\s\S]*?```|`[^`\n]*`|«[^»]*»|"[^"\n]*"|“[^”]*”',
        lambda match: " " * len(match.group()),
        text,
    )


def infer_language(question: str) -> str:
    normalized = normalize_persian(question)
    outside = _outside_quotes(normalized)
    matches = list(_DIRECTIVE.finditer(outside))
    if matches:
        value = (matches[-1].group("english") or matches[-1].group("persian")).lower()
        return "en" if value in {"english", "انگلیسی"} else "fa"
    population = outside if re.search(_PERSIAN_WORD + r"|[A-Za-z]", outside) else question
    persian = re.findall(_PERSIAN_WORD, population)
    latin = [w for w in re.findall(r"[A-Za-z]+", population) if w not in _ACRONYMS]
    # A Persian technical question can contain HR/KPI/RAG or an English product name.
    return "fa" if persian and len(persian) >= len(latin) else "en"


def strip_language_directive(question: str) -> str:
    # Only a trailing presentation request; never strip arbitrary content in the middle.
    text = normalize_persian(question)
    matches = list(_DIRECTIVE.finditer(_outside_quotes(text)))
    if matches and not text[matches[-1].end() :].strip(" .!?؟"):
        return text[: matches[-1].start()].rstrip(" .!?؟،,") or question
    return question


def response_language(question: str) -> str:
    current = _context.get()
    return current.language if current else infer_language(question)


def response_context(question: str) -> dict:
    current = _context.get()
    return {
        "target_language": current.language if current else infer_language(question),
        "original_student_message": current.original_message if current else question,
    }


@contextmanager
def language_scope(question: str):
    context = ResponseContext(infer_language(question), question)
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def explanation_language_matches(explanation, excerpts, target: str) -> bool:
    def narrative(text, evidence):
        for title in {e.citation.chapter_title for e in evidence if e.citation.chapter_title}:
            text = text.replace(title, " ")

        def quote(match):
            inner = match.group()[1:-1]
            return " " if any(inner in e.text for e in evidence) else match.group()

        text = re.sub(r'«[^»]+»|"[^"\n]+"|“[^”]+”', quote, text)
        return text

    by_id = {e.id: e for e in excerpts}
    for statement in explanation.statements:
        own = [by_id[i] for i in statement.citation_ids if i in by_id]
        text = narrative(statement.text, own)
        persian = re.findall(_PERSIAN_WORD, text)
        latin = [w for w in re.findall(r"[A-Za-z]+", text) if w not in _ACRONYMS]
        if target == "en" and len(persian) >= 2:
            return False
        if target == "fa" and (
            (not persian and sum(map(len, latin)) >= 12)
            or (len(latin) >= 4 and len(latin) > len(persian) / 2)
        ):
            return False
    return True
