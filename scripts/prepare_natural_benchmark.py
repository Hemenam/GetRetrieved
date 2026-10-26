"""Build, validate and plan the natural-student benchmark offline. No API/Settings imports.

Hand-authored questions live in evals/natural-student-v1/spec.py. Generated artifacts
are reproducible; existing output files are never overwritten. This is NOT execution
authorization, a semantic judge, a claim of source completeness, or an approval tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from audit_chapter_coverage import read_source  # noqa: E402

DEFAULT = ROOT / "evals" / "natural-student-v1"
SPEC_PATHS = {
    "natural-student-v1": DEFAULT / "spec.py",
    "natural-student-v2": ROOT / "evals" / "natural-student-v2" / "spec.py",
}
GENERATED = (
    "cases.jsonl",
    "assisted-pairs.jsonl",
    "manifest.json",
    "source-audit.json",
    "review.md",
    "questions.md",
    "grading.md",
)
STATUSES = {"answered", "partial", "refused", "clarification", "conversation"}
KINDS = {"content", "metadata", "control", "conversation"}
ALLOWED_FIELDS = {
    "course_title",
    "chapter_count",
    "chapter_titles",
    "chapter_title",
    "chapter_sections",
    "instructor",
    "duration",
    "prerequisites",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cases(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def encode_cases(cases):
    return "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in cases)


def bind_and_validate(cases, passages, source):
    """Bind every positive fact's anchors to exact scoped source spans; no meaning oracle."""
    sections = {(p["chapter_number"], p["section_kind"]): p for p in passages}
    chapters = sorted({(p["chapter_number"], p["chapter_title"]) for p in passages})
    ids, questions, bindings = set(), set(), []
    for c in cases:
        if (
            c["id"] in ids
            or c["question"] in questions
            or c["expected_status"] not in STATUSES
            or c["expected_language"] not in {"fa", "en"}
            or not c["parts"]
            or not 1 <= len(c["parts"]) <= 8
            or c["review_status"] != "proposed"
            or c["reviewer"]
            or c["split"] != "development"
        ):
            raise ValueError(f"Invalid proposed case: {c['id']}")
        ids.add(c["id"])
        questions.add(c["question"])
        if c["track"] != "course_metadata" and not c["location_assisted"]:
            # Mixed questions may legitimately ask for an actual metadata field.
            content_only = all(p["kind"] != "metadata" for p in c["parts"])
            if content_only and re.search(
                r"\bchapter\s+\w+|فصل\s+(?:اول|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم|یازدهم|[۰-۹0-9]+)",
                c["question"],
                re.I,
            ):
                raise ValueError(f"Unwanted source locator: {c['id']}")
        statuses = [p["status"] for p in c["parts"]]
        expected = (
            statuses[0]
            if len(statuses) == 1
            else "answered"
            if set(statuses) == {"answered"}
            else "partial"
            if "answered" in statuses
            else "refused"
        )
        if c["expected_status"] != expected:
            raise ValueError(f"Inconsistent statuses: {c['id']}")
        expected_kind = c["parts"][0]["kind"] if len(c["parts"]) == 1 else "multi"
        if expected_kind in {"control", "conversation"}:
            expected_kind = "content"  # Public API separates social/clarification via status.
        if c["expected_answer_kind"] != expected_kind:
            raise ValueError("Inconsistent answer shape")
        if bool(c.get("previous_question")) != bool(c.get("setup_reference")):
            raise ValueError("Every follow-up setup requires its own reference")
        cursor = 0
        for part in c["parts"]:
            if "question_text" in part:
                start = c["question"].find(part["question_text"], cursor)
                if start < 0 or not part["question_text"]:
                    raise ValueError("Expected part is not an exact ordered student span")
                cursor = start + len(part["question_text"])
        parts = [(str(i + 1), p) for i, p in enumerate(c["parts"])]
        if c.get("setup_reference"):
            parts.append(("setup", c["setup_reference"]))
        for part_id, part in parts:
            if (
                part["kind"] not in KINDS
                or part["status"] not in STATUSES - {"partial"}
                or not part["facts"]
            ):
                raise ValueError(f"Invalid part: {c['id']}/{part_id}")
            if part["kind"] == "metadata" and part.get("metadata_field") not in ALLOWED_FIELDS:
                raise ValueError("Unbounded metadata operation")
            if part["kind"] == "metadata" and (part["status"] == "answered") != (
                part.get("expected_value") is not None
            ):
                raise ValueError("Inconsistent structural metadata reference")
            if part["kind"] == "metadata":
                field, number = part["metadata_field"], part.get("chapter_number")
                expected_value = {
                    "chapter_count": len(chapters),
                    "chapter_titles": [{"number": n, "title": title} for n, title in chapters],
                    "course_title": [source["title"]],
                    "chapter_title": dict(chapters).get(number),
                    "chapter_sections": [
                        p["section_kind"] for p in passages if p["chapter_number"] == number
                    ]
                    or None,
                }.get(field)
                if part["expected_value"] != expected_value:
                    raise ValueError("Metadata reference differs from authorized source structure")
            for fact_id, fact in enumerate(part["facts"], 1):
                if (
                    part["kind"] == "content"
                    and part["status"] == "answered"
                    and not fact["evidence"]
                ):
                    raise ValueError(f"Unbound positive fact: {c['id']}")
                for anchor in fact["evidence"]:
                    p = sections[(anchor["chapter"], anchor["section"])]
                    offset = p["text"].find(anchor["quote"])
                    if offset < 0:
                        raise ValueError(
                            f"Source quote missing: {c['id']}/{part_id}: {anchor['quote']}"
                        )
                    start = p["source_start"] + offset
                    bindings.append(
                        {
                            "case_id": c["id"],
                            "part": part_id,
                            "fact": fact_id,
                            "chapter": anchor["chapter"],
                            "section": anchor["section"],
                            "quote": anchor["quote"],
                            "source_start": start,
                            "source_end": start + len(anchor["quote"]),
                            "passage_id": p["id"],
                            "document_id": source["document_id"],
                            "version": source["version"],
                        }
                    )
    return bindings


def api_case(case):
    """Adapt to existing capture code; this return value is grading data, NOT an API body.

    evaluate.api_evaluation sends only question/previous_question, requested mode and
    include_evaluation. Metadata/multipart need part-aware review in addition to capture.
    This suite must NOT be passed to the legacy strict JSONL loader unchanged.
    """
    quotes = list(
        dict.fromkeys(a["quote"] for p in case["parts"] for f in p["facts"] for a in f["evidence"])
    )
    result = {
        "id": case["id"],
        "category": case["track"],
        "question": case["question"],
        "expected_status": case["expected_status"],
        "expected_language": case["expected_language"],
        "expected_chapters": sorted(
            {a["chapter"] for p in case["parts"] for f in p["facts"] for a in f["evidence"]}
        ),
        "required_quotes": quotes,
        "required_facts": [],
        "notes": case["notes"],
        "review_status": case["review_status"],
        "reviewer": case["reviewer"],
        "split": case["split"],
    }
    if case.get("previous_question"):
        result["previous_question"] = case["previous_question"]
    return result


def student_payload(case):
    """The only benchmark-derived material allowed in the current-turn chatbot request."""
    return {"question": case["question"], "response_mode": "explained", "include_evaluation": True}


def plan(cases, assisted=(), *, include_assisted=False):
    selected = list(cases) + (list(assisted) if include_assisted else [])
    setups = sum(bool(c.get("previous_question")) for c in selected)
    return {
        "network_requests_made": 0,
        "selected_cases": len(selected),
        "main_cases": len(cases),
        "assisted_cases_included": len(selected) - len(cases),
        "setup_turns_per_mode": setups,
        "scored_turns_per_mode": len(selected),
        "modes": ["full_context", "hybrid", "hybrid_rerank"],
        "repetitions": 1,
        "query_turns_three_modes": (len(selected) + setups) * 3,
        "pending_reference_approvals": sum(c["review_status"] != "approved" for c in selected),
        "execution_authorized": False,
        "note": "Offline workload plan only. An API request can contain several billable model calls. This is not a dollar cap.",
    }


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def reference(part):
    return " ".join(f["text"] for f in part["facts"])


def render_review(cases, assisted, manifest):
    lines = [
        "# Natural student benchmark — reference review",
        "",
        f"{len(cases)} main cases + {len(assisted)} separate location-assisted diagnostics. All references are **proposed**.",
        "Exact evidence anchors were checked locally; presence of a quote does not prove its interpretation.",
        "The student sees only their question and actual prior conversation, not these answers or locations.",
        "",
        "هدف بازبینی: آیا سؤال طبیعی و روشن است؟ آیا پاسخ مورد انتظار کامل، وفادار به متن و بدون ادعای اضافه است؟",
        "",
        "`answered` = پاسخ مستند؛ `partial` = پاسخ به بخش‌های مجاز و اعلام نبود پاسخ برای بقیه؛ `refused` = عدم پاسخ محتوایی؛ `clarification` = نیاز واقعی به روشن‌سازی؛ `conversation` = گفت‌وگوی کوتاه.",
        "",
        "Meaning-equivalent natural answers are acceptable. These references are not scripts the model must copy.",
        "",
    ]
    for track in dict.fromkeys(c["track"] for c in cases):
        lines += [
            f"## {track}",
            "",
            "| ID | سؤال دانشجو / student question | وضعیت | پاسخ/رفتار مورد انتظار | محل شواهد برای داور |",
            "| --- | --- | --- | --- | --- |",
        ]
        for c in (x for x in cases if x["track"] == track):
            question = (
                ("زمینهٔ گفت‌وگو: " + c["previous_question"] + "<br>پرسش بعدی: ")
                if c.get("previous_question")
                else ""
            ) + c["question"]
            expected = "<br>".join(
                f"{i}. [{p['status']}/{p['kind']}] {reference(p)}"
                for i, p in enumerate(c["parts"], 1)
            )
            if c["format_requirement"]:
                expected += "<br>Format: " + c["format_requirement"]
            locations = sorted(
                {
                    (a["chapter"], a["section"])
                    for p in c["parts"]
                    for f in p["facts"]
                    for a in f["evidence"]
                }
            )
            evidence = "; ".join(f"{ch}/{section}" for ch, section in locations)
            if any(p["kind"] == "metadata" for p in c["parts"]):
                evidence += "; imported outline / missing administrative fields"
            lines.append(
                f"| {c['id']} | {cell(question)} | {c['expected_status']} / {c['expected_language']} | {cell(expected)} | {cell(evidence or 'Control/policy reference; human review required')} |"
            )
        lines.append("")
    lines += [
        "## Location-assisted diagnostic pairs (not main-score cases)",
        "",
        "| ID | Natural case | Assisted question |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {c['id']} | {c['paired_natural_id']} | {cell(c['question'])} |" for c in assisted]
    lines += [
        "",
        "## Source and approval boundaries",
        "",
        f"Source corpus SHA-256: `{manifest['source']['source_corpus_sha256']}`.",
        "See source-audit.json for exact scoped source spans, and grading.md for evaluation rules.",
        "Missing-field/negative and policy expectations need human review; quote matching does not prove absence.",
        "Approving this new dataset requires a separate review. Older dataset approvals and scores do not carry over.",
        "",
    ]
    return "\n".join(lines)


def render_questions(cases):
    lines = [
        "# Natural student questions",
        "",
        "Questions only; no answers, source locations or chapter hints added by the harness.",
        "",
    ]
    for c in cases:
        lines += [f"## {c['id']}", ""]
        if c.get("previous_question"):
            lines += ["Previous student turn: " + c["previous_question"], ""]
        lines += [c["question"], ""]
    return "\n".join(lines)


GRADING = """# Grading the natural-student benchmark

The reference facts express meanings, not prescribed answer wording. Exact source
quotes are evidence-audit anchors, NOT phrases that must appear in generated answers.
Automated citation/anchor/status checks are diagnostics, not semantic correctness scores.

## Human/independent semantic review

Score each dimension separately on 0, 1, 2 (wrong/missing; partial; satisfactory):

| Dimension | What satisfactory means |
| --- | --- |
| Intent/status | Answer, partial, refusal, clarification or social response is appropriate. Missing course facts are not unclear questions. |
| Directness | Responds to the student's actual need, not a source dump or an answer to a rewritten search query. |
| Factual support | Every claim follows from its own cited evidence; nearby but distinct concepts are not merged. |
| Completeness | All required facts/parts, ordering and qualifications are present. A correct fragment is not a complete answer. |
| Boundaries | Hypothetical examples stay hypothetical; numerical examples do not become universal policy; no clinical or legal conclusions are invented. |
| Context/grouping | Resolvable follow-ups use history, topic changes stand alone, comparisons remain joint, lists are not split into duplicate questions. |
| Language/presentation | Requested FA/EN and meaningful format constraints are followed; proper names and exact quotes may keep their original language. |
| Evidence/provenance | Citations resolve within the authorized course/revision and support the particular answer part. Metadata uses outline provenance, not fake quotes. |

Mark non-applicable dimensions N/A, never automatic full credit. A social greeting does
not earn eight easy citation/support points. For each score, record a brief reason and
the supporting source span. A source-unsupported claim is a critical failure even if the
answer is fluent, its citations exist and a model verifier approved it.

## Multipart and follow-up safeguards

Grade every expected part separately AND the overall request. Report all-parts-correct
rate and supported-part preservation; do not average away a missing answer. All-refused
requests are not partial. Related missing information and outside-course questions may
have different messages; require truthful scope handling, not one exact refusal phrase.

Count a follow-up only after the setup turn satisfies its own reference. Report failed
setup separately; do not mislabel a context failure as retrieval of the final question.
Use a fresh conversation for each case and each retrieval mode. Setup is an actual
student turn, never an answer injected into history by the test harness.

## Comparison and cost

Report by track and semantic family, and publish a macro-average across tracks alongside
the raw micro-average. Questions are deliberately stratified, NOT sampled from known LMS
traffic, and paraphrases are correlated. Do not claim population accuracy or independent
sample counts. All cases are development; none is a genuine untouched held-out test.
For release, collect independently authored student questions and additional courses.

Compare the same questions/reference version/source snapshot/model configuration in all
three modes. Freeze inputs before collection; no tuning on a scored capture followed by
silently overwriting it. The assisted pairs are a separate diagnostic: compare paired
status, meaning accuracy and required-evidence coverage; never pool them into the main score.

Keep operational failures (401/429/500/503/timeouts/invalid output) separate from valid
refusals. Preserve failed captures. Record latency p50/p95, request-level stage times,
reported input/output tokens, repair/retry attempts and cost with an explicit pricing date.
Count flattened provider calls once. Setup calls and failed/ambiguous billed attempts
belong in spend; missing provider usage is unknown, NOT zero. No live judge, hidden warmup,
semantic retries or benchmark execution is authorized by this preparation step.

## Current policy versus product aspirations

The current product explains supplied material, including source-backed scenarios. It
does not invent tailored crisis plans, personal clinical advice, a company's real rules,
or missing LMS administrative fields. New personalized plans are therefore proposed
refusals, not proof that every possible course assistant should refuse them. If the owner
changes that policy, revise and approve those references BEFORE comparing architectures.

Natural wording does not mean removing necessary context: a question about the example's
10:30 update must still identify the example. Names, topic terms and scenario details can
identify the information need without disclosing a chapter number or passage ID.
"""


def build(output, database, tenant, course, *, refresh_draft=False):
    if any((output / name).exists() for name in GENERATED):
        if not refresh_draft:
            raise FileExistsError("Generated artifacts already exist; choose a new output snapshot")
        old, _, _ = validate_pack(output, check_spec=False)
        if old["review_status"] != "proposed" or old["execution_authorized"]:
            raise ValueError("Never replace an approved or execution-authorized snapshot")
    source, passages = read_source(database, tenant, course)
    if len(passages) != 44 or sorted({p["chapter_number"] for p in passages}) != list(range(1, 12)):
        raise ValueError("This hand-authored reference pack requires the audited 11-chapter source")
    spec_path = DEFAULT / "spec.py"
    module_spec = importlib.util.spec_from_file_location("natural_student_spec", spec_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    base_path = ROOT / "evals/chapter-coverage-v1/approved-20261019/cases.jsonl"
    cases, assisted = module.definitions(read_cases(base_path), source, passages)
    return write_pack(output, cases, assisted, source, passages, spec_path, base_path)


def write_pack(
    output,
    cases,
    assisted,
    source,
    passages,
    spec_path,
    base_path,
    *,
    format_version="natural-student-v1",
    extra_manifest=None,
):
    """Write a previously guarded draft target; callers enforce snapshot preservation."""
    bindings = bind_and_validate(cases + assisted, passages, source)
    counts = Counter(c["track"] for c in cases)
    targets = {
        (c["source_coverage_target"]["chapter"], c["source_coverage_target"]["section"])
        for c in cases
        if c.get("source_coverage_target")
    }
    if targets != {(p["chapter_number"], p["section_kind"]) for p in passages}:
        raise ValueError("Incomplete source teaching-unit coverage")
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.jsonl").write_text(encode_cases(cases), encoding="utf-8")
    (output / "assisted-pairs.jsonl").write_text(encode_cases(assisted), encoding="utf-8")
    manifest = {
        "format_version": format_version,
        "created_date": "2026-10-22",
        "source": source,
        "source_review": "assistant_quote_span_checked_not_human_approved",
        "spec_sha256": digest(spec_path),
        "origin_coverage_sha256": digest(base_path),
        "cases_sha256": digest(output / "cases.jsonl"),
        "assisted_sha256": digest(output / "assisted-pairs.jsonl"),
        "main_case_count": len(cases),
        "assisted_case_count": len(assisted),
        "tracks": dict(counts),
        "statuses": dict(Counter(c["expected_status"] for c in cases)),
        "languages": dict(Counter(c["expected_language"] for c in cases)),
        "semantic_family_count": len({c["question_family_id"] for c in cases}),
        "review_status": "proposed",
        "execution_authorized": False,
        "plan_main_only": plan(cases),
        "plan_with_diagnostics": plan(cases, assisted, include_assisted=True),
        **(extra_manifest or {}),
    }
    audit = {
        "source": source,
        "bindings": bindings,
        "binding_count": len(bindings),
        "coverage_targets": [dict(chapter=ch, section=sec) for ch, sec in sorted(targets)],
        "all_anchors_present": True,
        "limitations": [
            "Quote presence is not proof of semantic entailment.",
            "Missing facts and policy outcomes require human review.",
            "Single course; all visible development cases; not a held-out accuracy estimate.",
        ],
    }
    (output / "source-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "review.md").write_text(render_review(cases, assisted, manifest), encoding="utf-8")
    (output / "questions.md").write_text(render_questions(cases), encoding="utf-8")
    (output / "grading.md").write_text(GRADING, encoding="utf-8")
    manifest["artifact_sha256"] = {
        name: digest(output / name) for name in GENERATED if name != "manifest.json"
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_pack(output, database=None, tenant=None, course=None, *, check_spec=True):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    spec_path = SPEC_PATHS.get(manifest["format_version"])
    if spec_path is None:
        raise ValueError("Unknown benchmark specification version")
    if check_spec and digest(spec_path) != manifest["spec_sha256"]:
        raise ValueError("Question specification changed after generation")
    if manifest.get("parent_manifest_sha256"):
        parent_manifest, _, _ = validate_pack(DEFAULT)
        if (
            digest(DEFAULT / "manifest.json") != manifest["parent_manifest_sha256"]
            or parent_manifest["cases_sha256"] != manifest["parent_cases_sha256"]
        ):
            raise ValueError("Parent benchmark changed after extension")
    for name, value in manifest["artifact_sha256"].items():
        if digest(output / name) != value:
            raise ValueError(f"Changed artifact: {name}")
    cases, assisted = (
        read_cases(output / "cases.jsonl"),
        read_cases(output / "assisted-pairs.jsonl"),
    )
    if (
        len(cases) != manifest["main_case_count"]
        or len(assisted) != manifest["assisted_case_count"]
    ):
        raise ValueError("Case count changed")
    by_id = {c["id"]: c for c in cases}
    for pair in assisted:
        original = by_id[pair["paired_natural_id"]]
        for key in ("parts", "question_family_id", "expected_language", "expected_status"):
            if pair[key] != original[key]:
                raise ValueError("Assisted comparison changed the information need/reference")
    if database:
        source, passages = read_source(
            database, tenant or manifest["source"]["tenant"], course or manifest["source"]["course"]
        )
        if source != manifest["source"]:
            raise ValueError("Source revision changed")
        bindings = bind_and_validate(cases + assisted, passages, source)
        audit = json.loads((output / "source-audit.json").read_text(encoding="utf-8"))
        if audit["bindings"] != bindings:
            raise ValueError("Source evidence audit differs")
    return manifest, cases, assisted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "validate", "plan"])
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--tenant", default="customer-demo")
    parser.add_argument("--course", default="captain-storm")
    parser.add_argument("--include-assisted", action="store_true")
    parser.add_argument(
        "--refresh-draft",
        action="store_true",
        help="Regenerate only an unchanged, unapproved generated draft",
    )
    args = parser.parse_args()
    if args.command == "build":
        if not args.database:
            parser.error("build requires --database (read-only source audit)")
        manifest = build(
            args.output, args.database, args.tenant, args.course, refresh_draft=args.refresh_draft
        )
        result = {
            k: manifest[k]
            for k in ("main_case_count", "assisted_case_count", "tracks", "statuses", "languages")
        }
    else:
        _, cases, assisted = validate_pack(args.output, args.database, args.tenant, args.course)
        result = plan(cases, assisted, include_assisted=args.include_assisted)
        result["artifact_hashes_valid"] = True
        result["source_rechecked"] = bool(args.database)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
