"""Ranking boundaries and service integration; no model downloads or paid calls."""

import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hrlearnium.config import Settings
from hrlearnium.models import ModelUnavailable
from hrlearnium.reranking import CrossEncoderReranker, rerank
from hrlearnium.retrieval import Candidate, SearchPassage


def settings(**overrides):
    return Settings(
        _env_file=None, jwt_secret="test-secret-32-characters-at-least-long", **overrides
    )


def candidates():
    return [
        Candidate(SearchPassage(str(i), f"متن {i}", i, "فصل", "explanation"), 1, 0.9, 0.1)
        for i in range(1, 4)
    ]


class Ranker:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def score(self, question, items):
        self.calls.append((question, items))
        return self.scores

    def identity(self):
        return {"model": "test-ranker"}


def test_ranking_promotes_relevant_evidence_preserves_passages_and_original_scores():
    pool = candidates()
    ranked = rerank("سوال", pool, Ranker([-1.0, 2.0, 0.0]), limit=2)
    assert [c.passage.id for c in ranked] == ["2", "3"]
    assert ranked[0].passage is pool[1].passage
    assert ranked[0].rank_score == pool[1].rank_score
    assert ranked[0].rerank_score == 2.0
    assert all(c.rerank_score is None for c in pool)


@pytest.mark.parametrize(
    "scores", [[1.0], [0.0, float("nan"), 1.0], [True, 1.0, 2.0], [[1.0], [2.0], [3.0]]]
)
def test_invalid_score_shape_count_or_value_is_an_infrastructure_failure(scores):
    with pytest.raises(ModelUnavailable):
        rerank("سوال", candidates(), Ranker(scores), limit=2)


def test_ties_are_stable_and_empty_pool_invokes_no_model():
    ranker = Ranker([1.0, 1.0, 1.0])
    assert [c.passage.id for c in rerank("سوال", candidates()[::-1], ranker, limit=3)] == [
        "1",
        "2",
        "3",
    ]
    ranker.calls.clear()
    assert rerank("سوال", [], ranker, limit=2) == []
    assert ranker.calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_backend": "literal"},
        {"reranker_model": ""},
        {"candidate_limit": 8, "rerank_pool_limit": 4},
    ],
)
def test_bad_reranking_configuration_is_rejected(overrides):
    with pytest.raises(ValidationError):
        settings(retrieval_mode="hybrid_rerank", **overrides)


def test_loading_is_lazy_local_only_and_remote_code_is_disabled(monkeypatch):
    constructors, predictions = [], []

    class Tokenizer:
        model_max_length = 8192

        def __call__(self, pairs, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": [[1, 2] for _ in pairs]}

    class Encoder:
        tokenizer = Tokenizer()
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash="pinned-commit"))

        def __init__(self, model, **kwargs):
            constructors.append((model, kwargs))

        def predict(self, pairs, **kwargs):
            predictions.append(pairs)
            return [float(i) for i in range(len(pairs))]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Encoder))
    ranker = CrossEncoderReranker(settings(reranker_revision=""))
    assert constructors == []
    ranker.score("سوال", candidates())
    ranker.score("سوال دوم", candidates())
    assert len(constructors) == 1
    assert constructors[0][1]["local_files_only"] is True
    assert constructors[0][1]["trust_remote_code"] is False
    assert constructors[0][1]["revision"] is None
    assert predictions[0][0][1] == "فصل\nمتن 1"
    assert ranker.identity()["resolved_revision"] == "pinned-commit"


def test_missing_dependency_and_overlong_evidence_fail_without_truncation(monkeypatch):
    ranker = CrossEncoderReranker(settings())
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ModelUnavailable, match="optional"):
        ranker.score("سوال", candidates())

    class Tokenizer:
        model_max_length = 3

        def __call__(self, pairs, **kwargs):
            return {"input_ids": [[1, 2, 3, 4] for _ in pairs]}

    ranker._encoder = SimpleNamespace(tokenizer=Tokenizer())
    with pytest.raises(ModelUnavailable, match="not truncated"):
        ranker.score("سوال", candidates())
