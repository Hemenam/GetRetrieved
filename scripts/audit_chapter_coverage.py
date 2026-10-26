"""Offline source audit and readable review for single-focus chapter cases.

Reads SQLite in mode=ro. Never imports settings, calls a model, approves gold,
modifies a course or overwrites an existing review. Quote presence is not proof
of semantic correctness; an owner must review the proposed meanings separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate import load_cases  # noqa: E402

SECTIONS = ("introduction", "explanation", "example", "exercise")
SECTION_LABELS = {
    "introduction": "مقدمه",
    "explanation": "توضیح اصلی",
    "example": "مثال",
    "exercise": "تمرین",
}
CASE_ID = re.compile(r"coverage-(\d{2})-(introduction|explanation|example|exercise)")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_source(database: Path, tenant: str, course: str) -> tuple[dict, list[dict]]:
    """Read one active document without opening a writable Storage connection."""
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        revisions = db.execute(
            """SELECT d.id,d.filename,d.version,r.title,r.full_text,r.source_sha256,
                      r.file_sha256,r.original_docx,r.paragraph_count
               FROM documents d JOIN revisions r
                 ON r.document_id=d.id AND r.version=d.version
               WHERE d.tenant_id=? AND d.course_id=? AND d.active=1""",
            (tenant, course),
        ).fetchall()
        if len(revisions) != 1:
            raise ValueError("This coverage pack requires exactly one active source document")
        revision = dict(revisions[0])
        if sha(revision["full_text"].encode("utf-8")) != revision["source_sha256"]:
            raise ValueError("Canonical text hash mismatch")
        if sha(revision["original_docx"]) != revision["file_sha256"]:
            raise ValueError("Original file hash mismatch")
        passages = [
            dict(row)
            for row in db.execute(
                """SELECT id,chapter_number,chapter_title,section_kind,paragraph_start,
                          paragraph_end,source_start,source_end,text
                   FROM passages WHERE document_id=? AND version=? ORDER BY source_start""",
                (revision["id"], revision["version"]),
            )
        ]
    for passage in passages:
        start, end = passage["source_start"], passage["source_end"]
        if not (0 <= start < end <= len(revision["full_text"])):
            raise ValueError("Invalid source offsets")
        if revision["full_text"][start:end] != passage["text"]:
            raise ValueError("Passage differs from its canonical source span")
    corpus_values = sorted(
        (p["chapter_number"] or 0, p["chapter_title"], p["section_kind"], p["text"])
        for p in passages
    )
    metadata = {
        "tenant": tenant,
        "course": course,
        "document_id": revision["id"],
        "filename": revision["filename"],
        "version": revision["version"],
        "title": revision["title"],
        "source_sha256": revision["source_sha256"],
        "file_sha256": revision["file_sha256"],
        "source_corpus_sha256": sha(json.dumps(corpus_values, ensure_ascii=False).encode("utf-8")),
        "paragraph_count": revision["paragraph_count"],
        "canonical_character_count": len(revision["full_text"]),
        "passage_count": len(passages),
    }
    return metadata, passages


def audit_cases(cases: list[dict], passages: list[dict]) -> list[dict]:
    """Require one single-focus proposed case per teaching section, with exact anchors."""
    sources = {}
    for p in passages:
        key = (p["chapter_number"], p["section_kind"])
        if key[0] is None or key[1] not in SECTIONS or key in sources:
            raise ValueError("Expected one source passage for each chapter/section pair")
        sources[key] = p
    chapters = sorted({key[0] for key in sources})
    expected = {(chapter, section) for chapter in chapters for section in SECTIONS}
    if not chapters or chapters != list(range(1, max(chapters) + 1)) or set(sources) != expected:
        raise ValueError("The source must contain all four sections in every chapter")
    targets, bindings = set(), []
    for case in cases:
        match = CASE_ID.fullmatch(case["id"])
        if not match:
            raise ValueError("Unexpected coverage case ID")
        key = (int(match[1]), match[2])
        if key in targets or key not in sources:
            raise ValueError("Duplicate or unknown evidence target")
        if (
            case["expected_status"] != "answered"
            or case["expected_chapters"] != [key[0]]
            or case.get("expected_language") != "fa"
            or case.get("split") != "development"
            or case.get("review_status") != "proposed"
            or case.get("reviewer") != ""
            or case.get("previous_question")
            or case["category"] in {"multi_section", "followup"}
            or not case.get("required_facts")
        ):
            raise ValueError("Expected a proposed, single-chapter, fresh-conversation answer case")
        targets.add(key)
        source = sources[key]
        anchors = list(
            dict.fromkeys(
                case["required_quotes"]
                + [q for fact in case["required_facts"] for q in fact["evidence_quotes"]]
            )
        )
        positions = []
        for anchor in anchors:
            relative = source["text"].find(anchor)
            if relative == -1:
                raise ValueError(
                    f"{case['id']}: an evidence anchor is absent from its target section"
                )
            positions.append(
                {
                    "quote": anchor,
                    "source_start": source["source_start"] + relative,
                    "source_end": source["source_start"] + relative + len(anchor),
                }
            )
        bindings.append(
            {
                "case_id": case["id"],
                **{k: v for k, v in source.items() if k != "text"},
                "anchors": positions,
            }
        )
    if targets != expected:
        raise ValueError("Every chapter/section pair needs exactly one proposed case")
    return sorted(bindings, key=lambda b: (b["chapter_number"], SECTIONS.index(b["section_kind"])))


def render_review(cases: list[dict], audit: dict) -> str:
    lookup = {c["id"]: c for c in cases}
    lines = [
        "# بررسی پرسش‌های تک‌موضوعی همهٔ فصل‌های دوره",
        "",
        "این بسته برای بررسی پوشش محتوای دوره است: یک پرسش از مقدمه، توضیح اصلی، مثال و تمرین هر فصل.",
        "همهٔ پاسخ‌های مورد انتظار پیشنهادی‌اند؛ بررسی خودکارِ وجود شواهد به معنی تأیید معنایی یا تأیید شما نیست.",
        "پرسش‌های چندموضوعی و ترکیب شواهد فصل‌های مختلف به مرحلهٔ بعد موکول شده‌اند.",
        "",
        "## وضعیت و دامنه",
        "",
        f"- تعداد پرسش‌ها: {len(cases)}؛ همه فارسی، تک‌نوبتی و نیازمند پاسخ از متن.",
        "- دامنهٔ فعلی: کل دوره؛ فصل و بخشِ نوشته‌شده زیر هر پرسش فقط محل شواهد است، نه محدودیت دسترسی اجراشده.",
        "- این فایل پاسخ واقعی چت‌بات را نشان نمی‌دهد؛ هنوز هیچ پاسخ جدیدی جمع‌آوری نشده است.",
        "- مجموعهٔ ۲۰ پرسش قبلی و نتایج آن بدون تغییر باقی مانده‌اند. این بسته یک توسعهٔ جداگانه است.",
        "- فهرست‌های تعریف‌شده در یک موضوع، یک درخواست‌اند؛ نیازی به ترکیب موضوعات مستقل نیست.",
        "- این مجموعهٔ توسعه است، نه آزمون ندیده. با توجه به انتشار پرسش‌ها، نباید بعداً آزمون مستقل نامیده شود.",
        "- عنوان و فهرست ابتدای سند، پرسش‌های ساختار کل سند، آزمون‌های منفی و مسائل چندموضوعی در این بسته نیستند.",
        "",
        "## روش بررسی",
        "",
        "برای هر پرسش بررسی کنید که معنا روشن است، پاسخ مورد انتظار دقیقاً به همان درخواست پاسخ می‌دهد،",
        "و شواهد نقل‌شده واقعاً کافی‌اند. بازنویسی وفادار به معنا پذیرفتنی است؛ پاسخ تولیدی لازم نیست عین نقل‌قول باشد.",
        "گزاره‌های ممنوع برای جلوگیری از تحریف‌اند؛ پاسخ لازم نیست همهٔ آن‌ها را صریحاً نفی کند.",
        "تأیید یا اصلاح خود را با شناسهٔ پرسش اعلام کنید. تیک‌های این فایل به‌تنهایی وضعیت JSONL را عوض نمی‌کنند.",
        "",
        "## پوشش",
        "",
        "| فصل | مقدمه | توضیح | مثال | تمرین |",
        "|---|---|---|---|---|",
    ]
    for chapter in sorted({b["chapter_number"] for b in audit["bindings"]}):
        lines.append(f"| {chapter} | ۱ پرسش | ۱ پرسش | ۱ پرسش | ۱ پرسش |")
    for binding in audit["bindings"]:
        case = lookup[binding["case_id"]]
        if binding["section_kind"] == "introduction":
            lines += ["", f"## فصل {binding['chapter_number']} {binding['chapter_title']}"]
        lines += [
            "",
            f"### {case['id']} — {SECTION_LABELS[binding['section_kind']]}",
            "",
            f"**پرسش:** {case['question']}",
            "",
            "**معنای لازم در پاسخ:**",
            "",
            *[f"- {fact['text']}" for fact in case["required_facts"]],
            "",
            f"**منبع:** فصل {binding['chapter_number']}، بخش {SECTION_LABELS[binding['section_kind']]}، "
            f"پاراگراف‌های {binding['paragraph_start']} تا {binding['paragraph_end']} در متن استاندارد استخراج‌شده.",
            "",
            "**شواهد دقیق:**",
            "",
        ]
        for anchor in binding["anchors"]:
            lines += [*[f"> {line}" for line in anchor["quote"].splitlines()], ""]
        if case["forbidden_claims"]:
            lines += ["**نباید ادعا شود:**", ""]
            lines += [f"- {claim}" for claim in case["forbidden_claims"]] + [""]
        lines += ["- [ ] تأیید می‌کنم", "- [ ] نیاز به اصلاح دارد", "", "یادداشت بررسی:", ""]
    source = audit["source"]
    lines += [
        "## مشخصات منبع و محدودیت‌ها",
        "",
        f"- نسخهٔ سند ذخیره‌شده: {source['version']}؛ پاراگراف‌ها: {source['paragraph_count']}؛ قطعات: {source['passage_count']}.",
        f"- SHA256 متن استاندارد: `{source['source_sha256']}`.",
        f"- SHA256 مجموعهٔ پرسش‌ها: `{audit['dataset_sha256']}`.",
        "- شمارهٔ پاراگراف با شمارهٔ صفحهٔ Word یا PDF یکی نیست. فقط متن بدنهٔ استخراج‌شده مبنای این بررسی است.",
        "- تمام نقل‌قول‌ها در محل تعیین‌شده پیدا شدند؛ این بررسی، درستی علمی یا حقوقی همهٔ ادعاهای کتاب را تضمین نمی‌کند.",
        "- پرسش‌های مربوط به توصیه‌ها و مثال‌ها باید در چارچوب همین متن پاسخ داده شوند، نه به صورت دستور واقعی برای شرکت شما.",
        "- بررسی مرز دسترسی، امتناع درست و پرسش‌های مبهم همچنان به آزمون‌های جداگانه نیاز دارد.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(manifest["documents"]) != 1:
        raise ValueError("Expected one source profile")
    profile = manifest["documents"][0]
    dataset = (args.manifest.parent / profile["dataset"]).resolve()
    cases = load_cases(dataset, None)
    source, passages = read_source(args.database, profile["tenant"], profile["course"])
    if source["source_corpus_sha256"] != profile.get("source_corpus_sha256"):
        raise ValueError("Source corpus differs from the manifest pin")
    bindings = audit_cases(cases, passages)
    audit = {
        "format_version": 1,
        "kind": "offline_proposed_gold_source_audit",
        "network_requests_made": 0,
        "provider_calls_made": 0,
        "owner_approved": False,
        "semantic_approval_inferred": False,
        "dataset_sha256": sha(dataset.read_bytes()),
        "source": source,
        "cases": len(cases),
        "chapters": len({b["chapter_number"] for b in bindings}),
        "chapter_section_pairs": len(bindings),
        "anchor_checks": sum(len(b["anchors"]) for b in bindings),
        "bindings": bindings,
    }
    outputs = {
        "source-audit.json": json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        "review.md": render_review(cases, audit),
    }
    if any((args.output_dir / name).exists() for name in outputs):
        raise ValueError("Refusing to overwrite an existing audit or review; use a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        with (args.output_dir / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    print(
        json.dumps(
            {key: audit[key] for key in ("cases", "chapters", "anchor_checks", "owner_approved")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
