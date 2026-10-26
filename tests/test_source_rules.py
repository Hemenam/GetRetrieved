"""Narrow polarity/provenance checks are deterministic, not an entailment claim."""

import pytest
from test_hosted_gateway import excerpt

from hrlearnium.source_rules import explicit_rule_conflict, source_status_preserved

SOURCE = "بایدها:\nتوقف فوریِ کار فعلی\nچک کردن کانال رسمی\nنبایدها:\nارسال پیام بدون هماهنگی"


@pytest.mark.parametrize(
    "claim,conflict",
    [
        ("باید توقف فوری کار فعلی انجام شود.", False),
        ("توقف فوری کار فعلی نباشد.", True),
        ("توقف فوری کار فعلی نیست.", True),
        ("نباید توقف فوری کار فعلی انجام شود.", True),
        ("نباید ارسال پیام بدون هماهنگی انجام شود.", False),
        ("ارسال پیام بدون هماهنگی باید انجام شود.", True),
        ("در مورد یک موضوع دیگر صحبت می‌کنیم.", False),
        (SOURCE, False),
    ],
)
def test_explicit_checklist_polarity(claim, conflict):
    assert explicit_rule_conflict(claim, SOURCE) is conflict


@pytest.mark.parametrize(
    "kind,claim,valid",
    [
        ("explanation", "کارکنان کانال رسمی را بررسی کنند.", True),
        ("exercise", "گروه رسمی شرکت ساخته شده است.", False),
        ("exercise", "تمرین دوره پیشنهاد می‌کند یک گروه رسمی بسازید.", True),
        ("example", "تارگت شرکت بیست درصد کاهش یافت.", False),
        ("example", "در مثال دوره، تارگت بیست درصد کاهش یافت.", True),
        ("example", "مثلاً مدیر در این بحران چنین کاری می‌کند.", True),
        ("example", "For instance, the example manager lowers the target.", True),
        ("example", "در سناریوی آموزشی چنین اقدامی انجام می‌شود.", True),
    ],
)
def test_examples_and_exercises_are_explicitly_attributed(kind, claim, valid):
    source = excerpt()
    source.citation.section_kind = kind
    assert source_status_preserved(claim, [source]) is valid


@pytest.mark.parametrize("language,kind", [("fa", "example"), ("en", "exercise")])
def test_scope_is_attached_before_review_without_changing_claims_or_citations(language, kind):
    from hrlearnium.schemas import Explanation
    from hrlearnium.source_rules import attribute_source_scope

    source = excerpt()
    source.citation.section_kind = kind
    draft = Explanation(statements=[{"text": "A course statement.", "citation_ids": [source.id]}])
    scoped = attribute_source_scope(draft, [source], language)
    assert scoped.statements[0].text.endswith(draft.statements[0].text)
    assert scoped.statements[0].citation_ids == draft.statements[0].citation_ids
    assert source_status_preserved(scoped.statements[0].text, [source])
    assert draft.statements[0].text == "A course statement."
    assert attribute_source_scope(scoped, [source], language) == scoped
