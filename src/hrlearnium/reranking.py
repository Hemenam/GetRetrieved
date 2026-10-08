"""Optional cross-encoder ranking of an already authorized candidate pool."""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from importlib.util import find_spec
from numbers import Real
from typing import Protocol

from hrlearnium.config import Settings
from hrlearnium.models import ModelUnavailable
from hrlearnium.retrieval import Candidate


class Reranker(Protocol):
    def score(self, question: str, candidates: list[Candidate]) -> list[float]: ...
    def identity(self) -> dict: ...


def rerank(
    question: str,
    candidates: list[Candidate],
    ranker: Reranker,
    *,
    limit: int,
    min_score: float | None = None,
) -> list[Candidate]:
    """Never add or rewrite evidence. Scores indicate relevance, not answerability."""
    if min_score is not None and (
        isinstance(min_score, bool)
        or not isinstance(min_score, Real)
        or not math.isfinite(min_score)
    ):
        raise ValueError("Reranker threshold must be a finite scalar")
    if not candidates:
        return []
    scores = ranker.score(question, candidates)
    if len(scores) != len(candidates) or any(
        isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score)
        for score in scores
    ):
        raise ModelUnavailable("Reranker returned invalid scores")
    ranked = [
        replace(candidate, rerank_score=float(score))
        for candidate, score in zip(candidates, scores, strict=True)
        if min_score is None or score >= min_score
    ]
    return sorted(ranked, key=lambda item: (-item.rerank_score, -item.rank_score, item.passage.id))[
        :limit
    ]


class CrossEncoderReranker:
    """Load once, on first use. Normal/full-context queries import no ML runtime.

    Local files are required by default; model downloads need explicit configuration.
    Overlong query/passage pairs are rejected instead of silently truncating evidence.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._encoder = None
        self._lock = threading.Lock()
        self._resolved_revision = None

    def identity(self) -> dict:
        return {
            "model": self.settings.reranker_model,
            "revision": self.settings.reranker_revision,
            "resolved_revision": self._resolved_revision,
            "device": self.settings.reranker_device,
            "max_tokens": self.settings.reranker_max_tokens,
            "batch_size": self.settings.reranker_batch_size,
            "local_files_only": self.settings.reranker_local_files_only,
        }

    def readiness(self) -> dict:
        installed = find_spec("sentence_transformers") is not None
        return {
            "ready": installed,
            "check": "configuration_only",
            "model_loaded": self._encoder is not None,
            "inference_verified": False,
            "missing_dependencies": [] if installed else ["hrlearnium-chatbot[rerank]"],
        }

    def _load(self):
        if self._encoder is None:
            try:
                from sentence_transformers import CrossEncoder

                encoder = CrossEncoder(
                    self.settings.reranker_model,
                    revision=self.settings.reranker_revision,
                    device=self.settings.reranker_device,
                    max_length=self.settings.reranker_max_tokens,
                    local_files_only=self.settings.reranker_local_files_only,
                    trust_remote_code=False,
                )
                self._resolved_revision = getattr(encoder.model.config, "_commit_hash", None)
                self._encoder = encoder
            except ImportError:
                raise ModelUnavailable(
                    "Install the optional hrlearnium-chatbot[rerank] extra"
                ) from None
            except Exception:
                raise ModelUnavailable("Reranker model could not be loaded") from None
        return self._encoder

    def score(self, question: str, candidates: list[Candidate]) -> list[float]:
        pairs = [
            (question, item.passage.chapter_title + "\n" + item.passage.text) for item in candidates
        ]
        with self._lock:
            encoder = self._load()
            try:
                limit = self.settings.reranker_max_tokens
                model_limit = encoder.tokenizer.model_max_length
                if isinstance(model_limit, int) and 0 < model_limit < limit:
                    limit = model_limit
                tokens = encoder.tokenizer(pairs, padding=False, truncation=False, verbose=False)[
                    "input_ids"
                ]
                if any(len(ids) > limit for ids in tokens):
                    raise ModelUnavailable(
                        "Reranker token limit exceeded; evidence was not truncated"
                    )
                values = encoder.predict(
                    pairs,
                    batch_size=self.settings.reranker_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                # Classifier vectors cannot be interpreted as scalar relevance scores.
                if getattr(values, "ndim", 1) != 1:
                    raise ModelUnavailable("Use a reranker with one relevance score per pair")
                return values.tolist() if hasattr(values, "tolist") else list(values)
            except ModelUnavailable:
                raise
            except Exception:
                raise ModelUnavailable("Reranker inference failed") from None
