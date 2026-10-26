"""A list heading scopes all its items, not merely the nearest three tokens."""

import pytest

from hrlearnium.source_rules import explicit_rule_conflict

SOURCE = "بایدها:\nثبت وضعیت اولیه\nنبایدها:\nارسال پرونده مشتری\nپاک کردن گزارش روزانه"


@pytest.mark.parametrize(
    "claim,conflict",
    [
        ("نبایدها شامل ارسال پرونده مشتری و پاک کردن گزارش روزانه است.", False),
        ("بایدها شامل ارسال پرونده مشتری و پاک کردن گزارش روزانه است.", True),
        (
            "بایدها شامل ثبت وضعیت اولیه است؛ نبایدها شامل ارسال پرونده مشتری و پاک کردن گزارش روزانه است.",
            False,
        ),
        ("نبایدها شامل ارسال پرونده مشتری است؛ بایدها شامل پاک کردن گزارش روزانه است.", True),
    ],
)
def test_explicit_heading_polarity_applies_to_each_list_item(claim, conflict):
    assert explicit_rule_conflict(claim, SOURCE) is conflict
