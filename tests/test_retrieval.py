import math

import pytest

from hrlearnium.retrieval import SearchPassage, retrieve
from hrlearnium.text import meaningful_tokens, normalize_persian, tokenize


def passage(identifier, text, *, title="", embedding=None):
    return SearchPassage(identifier, text, 1, title, "explanation", embedding)


def test_persian_normalization_is_search_only():
    original = "  كَـتابِ يک می\u200cماند ۱۲٣۴٥  "
    assert normalize_persian(original) == "کتاب یک می ماند 12345"
    assert original == "  كَـتابِ يک می\u200cماند ۱۲٣۴٥  "
    assert tokenize("مدل ۳۰-۳۰-۳۰؛ HR") == ["مدل", "30", "30", "30", "hr"]
    assert meaningful_tokens("مدل ۳۰-۳۰-۳۰ چیست؟") == ["مدل", "30", "30", "30"]


def test_lexical_retrieval_matches_letter_and_digit_variants():
    source = passage("model", "مدل ۳۰-۳۰-۳۰ سه سبد زمانی دارد؛ دقیقه، ساعت و روز.")
    results = retrieve("مدل ٣٠-٣٠-٣٠ چيست؟", [passage("other", "گوش دادن همدلانه"), source])
    assert results[0].passage is source
    assert results[0].lexical_score > 0
    assert results[0].semantic_score is None
    assert results[0].rank_score == results[0].lexical_score
    assert results[0].passage.text == source.text


def test_bm25_specific_term_beats_generic_overlap():
    results = retrieve(
        "بحران دی‌بریف",
        [
            passage("specific", "دی‌بریف پس از بحران شامل مرور وقایع و احساسات است."),
            passage("generic", "بحران بحران بحران بحران بحران"),
            passage("other", "همدلی و پشتیبانی کارکنان"),
        ],
    )
    assert results[0].passage.id == "specific"


def test_chapter_title_is_searchable_metadata():
    results = retrieve(
        "مدل قطب نما",
        [passage("model", "سه سبد زمانی را رعایت کنید.", title="قطب‌نما در مه مدل تصمیم‌گیری")],
    )
    assert results[0].passage.id == "model"


def test_unknown_and_stopword_only_queries_return_nothing():
    passages = [passage("a", "گوش دادن همدلانه")]
    assert retrieve("پایتخت ژاپن", passages) == []
    assert retrieve("این چیست؟", passages) == []
    assert retrieve("", passages) == []
    assert retrieve("گوش", passages, limit=0) == []


def test_hybrid_search_can_retrieve_semantic_paraphrase():
    lexical = passage("lexical", "آرامش", embedding=[0.0, 1.0])
    semantic = passage("semantic", "پشتیبانی روانی", embedding=[1.0, 0.0])
    results = retrieve("آرامش", [lexical, semantic], query_embedding=[1.0, 0.0])
    assert {result.passage.id for result in results} == {"lexical", "semantic"}
    assert (
        next(result for result in results if result.passage.id == "semantic").semantic_score == 1.0
    )


def test_rrf_rewards_agreement_between_rankers():
    results = retrieve(
        "همدلی",
        [
            passage("both", "همدلی", embedding=[1.0, 0.0]),
            passage("lexical", "همدلی", embedding=[-1.0, 0.0]),
            passage("semantic", "پشتیبانی", embedding=[0.9, 0.1]),
        ],
        query_embedding=[1.0, 0.0],
    )
    assert results[0].passage.id == "both"
    assert results[0].rank_score > results[1].rank_score


def test_missing_passage_embeddings_are_allowed():
    results = retrieve("همدلی", [passage("a", "همدلی", embedding=None)], query_embedding=[1.0, 0.0])
    assert len(results) == 1
    assert results[0].semantic_score is None


@pytest.mark.parametrize(
    "vector", [[], [0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0], [True, 1.0], ["1", 1.0]]
)
def test_invalid_query_embeddings_are_rejected(vector):
    with pytest.raises(ValueError):
        retrieve("متن", [passage("a", "متن", embedding=[1.0, 0.0])], query_embedding=vector)


@pytest.mark.parametrize("vector", [[1.0], [math.nan, 1.0], [0.0, 0.0]])
def test_invalid_passage_embeddings_are_rejected(vector):
    with pytest.raises(ValueError):
        retrieve("متن", [passage("a", "متن", embedding=vector)], query_embedding=[1.0, 0.0])


def test_large_finite_vectors_do_not_overflow():
    result = retrieve(
        "همدلی", [passage("a", "همدلی", embedding=[1e300, 1e300])], query_embedding=[1e300, 1e300]
    )[0]
    assert math.isfinite(result.semantic_score)
    assert result.semantic_score == pytest.approx(1.0)


def test_ties_and_limits_are_deterministic_independent_of_input_order():
    passages = [passage("b", "همدلی"), passage("a", "همدلی"), passage("c", "همدلی")]
    assert [result.passage.id for result in retrieve("همدلی", passages, limit=2)] == ["a", "b"]
    assert retrieve("همدلی", passages) == retrieve("همدلی", list(reversed(passages)))


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        retrieve("همدلی", [passage("a", "همدلی"), passage("a", "متن دیگر")])
