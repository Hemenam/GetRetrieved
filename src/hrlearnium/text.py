"""Search-only Persian normalization. Never apply these helpers to displayed excerpts."""

from __future__ import annotations

import re
import unicodedata

_TRANSLATION = str.maketrans(
    "كيىةۀؤإأٱ٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "کییههوااا01234567890123456789",
)
_ARABIC_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_STOP_WORDS = frozenset(
    """از به با در برای و یا که را این آن یک ما شما من او آنها ها های ای
    است هست هستند بود بوده باشد باشند شود شده می نمی هم تا اما اگر فقط
    چه چیست چگونه چرا کدام آیا چطور چند درباره مورد لطفا لطفاً بگو بده
    کنید کند کنیم کنم شود شوند شدن کردن کردنم دارد دارند دارم داشتن
    باید تواند توانند تواندش بسیار بیشتر کمتر همه هر هیچ طبق براساس بر اساس
    متن دوره سند فصل توضیح بدهید دهید پاسخ سوال پرسش میخواهم میخوام
    the a an is are of to in and or what how why please explain document course
    """.split()
)


def normalize_persian(text: str) -> str:
    """Normalize spelling, marks, joiners, digits and spacing for matching only.

    Joining controls become spaces so ``تصمیم‌گیری`` and ``تصمیم گیری`` match.
    NFC deliberately avoids compatibility rewriting of source-like symbols.
    """
    normalized = unicodedata.normalize("NFC", text).translate(_TRANSLATION)
    normalized = _ARABIC_MARKS.sub("", normalized).replace("ـ", "")
    normalized = re.sub("[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]", " ", normalized)
    return " ".join(normalized.casefold().split())


def tokenize(text: str) -> list[str]:
    """Return normalized Unicode words and numbers, without punctuation."""
    return _TOKEN.findall(normalize_persian(text))


def meaningful_tokens(text: str) -> list[str]:
    """Remove function words while retaining domain terms, digits and acronyms."""
    return [word for word in tokenize(text) if word not in _STOP_WORDS]
