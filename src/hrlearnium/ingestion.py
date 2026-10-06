"""Bounded, text-only DOCX ingestion into exact, attributable source spans.

The canonical source is nonempty body paragraphs joined with one newline.
Run text, tabs, line breaks and original Unicode are preserved. Paragraph
references are one-based and inclusive in that canonical text. Formatting,
automatic list numbering, headers, footers and images are not synthesized.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile
from zlib import error as ZlibError

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from .text import normalize_persian

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 256
MAX_EXPANDED_BYTES = 40 * 1024 * 1024
MAX_DOCUMENT_XML_BYTES = 8 * 1024 * 1024
MAX_PARAGRAPHS = 20_000
MAX_TEXT_CHARACTERS = 2_000_000
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_ORDINALS = {
    word: number
    for number, word in enumerate(
        (
            "اول",
            "دوم",
            "سوم",
            "چهارم",
            "پنجم",
            "ششم",
            "هفتم",
            "هشتم",
            "نهم",
            "دهم",
            "یازدهم",
            "دوازدهم",
            "سیزدهم",
            "چهاردهم",
            "پانزدهم",
            "شانزدهم",
            "هفدهم",
            "هجدهم",
            "نوزدهم",
            "بیستم",
        ),
        start=1,
    )
}
_CHAPTER = re.compile(r"^فصل\s+(\d+|" + "|".join(_ORDINALS) + r")(?:\s*[-–—:؛]\s*(.*))?$")


class IngestionError(ValueError):
    """The supplied archive cannot be safely parsed into course text."""


@dataclass(frozen=True)
class ParsedPassage:
    text: str
    chapter_number: int | None
    chapter_title: str
    section_kind: str
    paragraph_start: int
    paragraph_end: int
    source_start: int
    source_end: int


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    full_text: str
    passages: list[ParsedPassage]
    paragraph_count: int
    warnings: list[str]


def _chapter(text: str) -> tuple[int, str] | None:
    match = _CHAPTER.fullmatch(normalize_persian(text))
    if not match:
        return None
    number = int(match[1]) if match[1].isdigit() else _ORDINALS[match[1]]
    # Metadata should preserve spelling in the original heading, too.
    original_title = re.split(r"[-–—:؛]", text, maxsplit=1)
    return number, original_title[1].strip() if len(original_title) == 2 else ""


def _section_kind(text: str) -> str | None:
    normalized = normalize_persian(text)
    if len(normalized) > 140:
        return None
    if re.fullmatch(
        r"(?:یک\s+)?مثال(?:\s+را)?(?:\s+بررسی کنیم|\s+عملی|\s+کاربردی)?\s*[:：]?", normalized
    ):
        return "example"
    if re.fullmatch(r"تمرین(?:\s+(?:عملی|واقعی|پایانی|فردی|گروهی))?\s*[:：]?", normalized):
        return "exercise"
    return None


def _paragraph_text(element) -> str:
    # Deleted revisions and field instructions are not visible body evidence.
    parts: list[str] = []

    def visit(node) -> None:
        if node.tag in {_W + "del", _W + "moveFrom", _W + "instrText"}:
            return
        if node.tag == _W + "t":
            parts.append(node.text or "")
        elif node.tag == _W + "tab":
            parts.append("\t")
        elif node.tag in {_W + "br", _W + "cr"}:
            parts.append("\n")
        elif node.tag == _W + "noBreakHyphen":
            parts.append("\u2011")
        elif node.tag == _W + "softHyphen":
            parts.append("\u00ad")
        else:
            for child in node:
                # Text boxes have nested paragraphs; don't duplicate them.
                if child.tag != _W + "p":
                    visit(child)

    visit(element)
    return "".join(parts)


def _read_paragraphs(data: bytes) -> tuple[list[str], list[str]]:
    if not data:
        raise IngestionError("The uploaded DOCX is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestionError("The DOCX exceeds the 10 MiB upload limit")
    warnings: list[str] = []
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise IngestionError("The DOCX archive contains too many entries")
            names = [entry.filename for entry in entries]
            if len(set(names)) != len(names):
                raise IngestionError("Duplicate entries in the DOCX archive")
            if sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES:
                raise IngestionError("The expanded DOCX exceeds the size limit")
            if any(entry.flag_bits & 1 for entry in entries):
                raise IngestionError("Encrypted DOCX archives are not supported")
            if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                raise IngestionError("The file is not a supported DOCX document")
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_DOCUMENT_XML_BYTES:
                raise IngestionError("The DOCX document XML exceeds the size limit")
            with archive.open(info) as stream:
                xml = stream.read(MAX_DOCUMENT_XML_BYTES + 1)
            if len(xml) > MAX_DOCUMENT_XML_BYTES:
                raise IngestionError("The DOCX document XML exceeds the size limit")
            root = SafeET.fromstring(
                xml, forbid_dtd=True, forbid_entities=True, forbid_external=True
            )
            body = root.find(_W + "body")
            if root.tag != _W + "document" or body is None:
                raise IngestionError("The DOCX has no supported document body")
            elements = list(body.iter(_W + "p"))
            if len(elements) > MAX_PARAGRAPHS:
                raise IngestionError("The DOCX contains too many paragraphs")
            paragraphs = [
                text for element in elements if (text := _paragraph_text(element)).strip()
            ]
            if sum(map(len, paragraphs)) > MAX_TEXT_CHARACTERS:
                raise IngestionError("The extracted document text exceeds the size limit")
            if not paragraphs:
                raise IngestionError(
                    "No readable body text was found; image-only documents require OCR"
                )
            if any(
                element.tag in {_W + "drawing", _W + "pict", _W + "object"}
                for element in body.iter()
            ):
                warnings.append(
                    "Images and embedded objects were not extracted; inspect them for missing course material."
                )
            if any(element.tag == _W + "numPr" for element in body.iter()):
                warnings.append(
                    "Automatic list markers are formatting and were not synthesized; literal list text is preserved."
                )
            if any(element.tag == _W + "tbl" for element in body.iter()):
                warnings.append(
                    "Table cells were read in document order as paragraphs; review their relationships before publication."
                )
            if any(
                element.tag in {_W + "del", _W + "moveFrom", _W + "ins", _W + "moveTo"}
                for element in body.iter()
            ):
                warnings.append(
                    "Tracked revisions were read as the current text: insertions included, deletions excluded."
                )
            if any(
                name.startswith(("word/header", "word/footer", "word/footnotes", "word/endnotes"))
                for name in names
            ):
                warnings.append(
                    "Only document body text was extracted; headers, footers, footnotes and endnotes were excluded."
                )
            return paragraphs, warnings
    except IngestionError:
        raise
    except (
        BadZipFile,
        KeyError,
        OSError,
        RuntimeError,
        NotImplementedError,
        ParseError,
        DefusedXmlException,
        RecursionError,
        ZlibError,
    ) as exc:
        raise IngestionError(
            "The file is corrupt, unsafe, or not a supported DOCX document"
        ) from exc


def _explanation_boundary(paragraphs: list[str], start: int, end: int) -> int | None:
    """Find the first explanation/list introduction after a substantial opening.

    Keep the entire following list together, including short labels and their
    explanation paragraphs. Short adjacent lead-ins are pulled into the list.
    """
    for index in range(start + 1, end):
        text = paragraphs[index].strip()
        previous_length = sum(len(value) for value in paragraphs[start:index])
        if previous_length < 250 or len(text) > 250 or not text.endswith((":", "：")):
            continue
        candidate = index
        while candidate > start + 1:
            preceding = paragraphs[candidate - 1].strip()
            if len(preceding) >= 120 or normalize_persian(preceding).startswith("هدف"):
                break
            candidate -= 1
        return candidate
    return None


def parse_docx(data: bytes, filename: str = "course.docx") -> ParsedDocument:
    """Extract exact spans; uploaded prose never executes or becomes instructions."""
    paragraphs, warnings = _read_paragraphs(data)
    full_text = "\n".join(paragraphs)
    offsets: list[int] = []
    cursor = 0
    for paragraph in paragraphs:
        offsets.append(cursor)
        cursor += len(paragraph) + 1
    headings = [
        (index, heading)
        for index, text in enumerate(paragraphs)
        if (heading := _chapter(text)) is not None
    ]
    # A leading consecutive run of chapter headings is a contents listing.
    contents: set[int] = set()
    if headings:
        run: list[int] = []
        for index, (_, inline_title) in headings:
            if run and (index != run[-1] + 1 or not inline_title):
                if len(run) >= 3:
                    contents.update(run)
                run = []
            if inline_title:
                run.append(index)
        if len(run) >= 3:
            contents.update(run)
    headings = [(index, heading) for index, heading in headings if index not in contents]
    if contents:
        warnings.append(
            "The chapter contents listing is retained in source text but excluded from answerable passages."
        )
    first_text = paragraphs[0]
    title = (
        first_text.strip()
        if _chapter(first_text) is None and len(first_text) <= 200
        else filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    )
    passages: list[ParsedPassage] = []

    def append(start: int, end: int, number: int | None, chapter_title: str, kind: str) -> None:
        if start >= end:
            return
        source_start = offsets[start]
        source_end = offsets[end - 1] + len(paragraphs[end - 1])
        passages.append(
            ParsedPassage(
                full_text[source_start:source_end],
                number,
                chapter_title,
                kind,
                start + 1,
                end,
                source_start,
                source_end,
            )
        )

    def sections(start: int, end: int, number: int | None, chapter_title: str) -> None:
        boundaries = [
            (index, kind)
            for index in range(start, end)
            if (kind := _section_kind(paragraphs[index])) is not None
        ]
        introductory_end = boundaries[0][0] if boundaries else end
        explanation_start = _explanation_boundary(paragraphs, start, introductory_end)
        if explanation_start is not None:
            append(start, explanation_start, number, chapter_title, "introduction")
            append(explanation_start, introductory_end, number, chapter_title, "explanation")
        elif start < introductory_end:
            append(start, introductory_end, number, chapter_title, "explanation")
        for position, (section_start, kind) in enumerate(boundaries):
            section_end = boundaries[position + 1][0] if position + 1 < len(boundaries) else end
            # Never make a heading-only answerable passage.
            if section_end > section_start + 1:
                append(section_start, section_end, number, chapter_title, kind)

    if headings:
        for position, (index, (number, chapter_title)) in enumerate(headings):
            end = headings[position + 1][0] if position + 1 < len(headings) else len(paragraphs)
            start = index + 1
            if not chapter_title and start < end:
                chapter_title = paragraphs[start].strip()
                start += 1
            if start < end:
                sections(start, end, number, chapter_title)
        preamble_end = headings[0][0]
        substantive_preamble = [
            index
            for index in range(preamble_end)
            if index not in contents and len(paragraphs[index].strip()) > 200
        ]
        if substantive_preamble:
            warnings.append(
                "Text before the first chapter was excluded from passages; review it before publication."
            )
    elif contents:
        raise IngestionError(
            "Only a chapter contents listing was found, with no answerable course body"
        )
    else:
        warnings.append(
            "No Persian chapter structure detected; review extracted passage boundaries before publication."
        )
        start = 1 if len(paragraphs) > 1 and len(paragraphs[0].strip()) < 120 else 0
        sections(start, len(paragraphs), None, title)
    if not passages:
        raise IngestionError("The document contains no answerable course passages")
    if any(len(passage.text) > 6000 for passage in passages):
        warnings.append(
            "Some sections exceed 6000 characters; review their boundaries before publication."
        )
    return ParsedDocument(title, full_text, passages, len(paragraphs), warnings)
