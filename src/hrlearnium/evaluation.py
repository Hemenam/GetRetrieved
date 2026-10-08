"""Request-local diagnostics, explicitly enabled by the evaluation runner."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from hrlearnium.policy import (
    EXPLANATION_INSTRUCTIONS,
    SELECTOR_INSTRUCTIONS,
    VERIFICATION_INSTRUCTIONS,
)
from hrlearnium.schemas import POLICY_VERSION

_trace: ContextVar[dict | None] = ContextVar("course_evaluation", default=None)
_stage: ContextVar[str | None] = ContextVar("course_evaluation_stage", default=None)


def configuration(settings, reranker=None) -> dict:
    prompts = [SELECTOR_INSTRUCTIONS, EXPLANATION_INSTRUCTIONS, VERIFICATION_INSTRUCTIONS]
    code = hashlib.sha256()
    for name in ("service", "models", "policy", "schemas", "ingestion", "retrieval", "reranking"):
        code.update(name.encode())
        code.update(Path(__file__).with_name(name + ".py").read_bytes())
    hosted = settings.model_backend == "openai_compatible"
    endpoint = settings.api_base_url if hosted else settings.ollama_base_url
    return {
        "backend": settings.model_backend,
        "chat_model": settings.api_model if hosted else settings.selector_model,
        "embedding_model": settings.api_embedding_model if hosted else settings.embedding_model,
        "embedding_revision": settings.api_embedding_revision if hosted else None,
        "provider_fingerprint": hashlib.sha256(endpoint.rstrip("/").encode()).hexdigest(),
        "structured_output": settings.api_structured_output if hosted else "ollama_json_schema",
        "max_completion_tokens": settings.api_max_completion_tokens if hosted else 4096,
        "candidate_limit": settings.candidate_limit,
        "relevance_thresholds": {
            "min_cosine": settings.retrieval_min_cosine,
            "min_bm25": settings.retrieval_min_bm25,
            "search_gate": "cosine OR bm25; positive score required",
            "min_reranker_score": settings.reranker_min_score,
            "applies_to": ["hybrid", "hybrid_rerank"],
            "calibrated_confidence": False,
        },
        "max_excerpts": settings.max_excerpts,
        "max_context_characters": settings.max_context_characters,
        "rerank_pool_limit": settings.rerank_pool_limit,
        "policy_version": POLICY_VERSION,
        "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "pipeline_sha256": code.hexdigest(),
        "reranker": (
            {key: value for key, value in reranker.identity().items() if key != "resolved_revision"}
            if reranker
            else None
        ),
    }


@contextmanager
def collect(settings, enabled: bool, reranker=None):
    value = (
        {"configuration": configuration(settings, reranker), "stages_ms": {}, "provider_calls": []}
        if enabled
        else None
    )
    token = _trace.set(value)
    try:
        yield value
    finally:
        _trace.reset(token)


@contextmanager
def stage(name: str):
    token = _stage.set(name)
    started = time.perf_counter()
    try:
        yield
    finally:
        trace = _trace.get()
        if trace is not None:
            trace["stages_ms"][name] = round((time.perf_counter() - started) * 1000, 3)
        _stage.reset(token)


def candidates(name: str, items):
    trace = _trace.get()
    if trace is not None:
        trace[name] = [
            {
                "id": item.passage.id,
                "chapter": item.passage.chapter_number,
                "lexical_score": item.lexical_score,
                "semantic_score": item.semantic_score,
                "rank_score": item.rank_score,
                "rerank_score": item.rerank_score,
            }
            for item in items
        ]


def corpus(passages):
    trace = _trace.get()
    if trace is not None:
        # Stable across reimports: IDs/embeddings are excluded; all source passages are included.
        values = sorted(
            (item.chapter_number or 0, item.chapter_title, item.section_kind, item.text)
            for item in passages
        )
        trace["source_corpus_sha256"] = hashlib.sha256(
            json.dumps(values, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        trace["source_passage_count"] = len(passages)


def relevance_filter(before: int, after: int):
    trace = _trace.get()
    if trace is not None:
        trace["relevance_filter"] = {
            "candidates_before": before,
            "candidates_after": after,
            "rejected_count": before - after,
        }


def provider_usage(result: dict, model: str, kind: str):
    trace = _trace.get()
    if trace is None:
        return
    raw = result.get("usage", {})
    usage = {
        key: raw[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(raw, dict) and type(raw.get(key)) is int and raw[key] >= 0
    }
    trace["provider_calls"].append(
        {
            "stage": _stage.get(),
            "kind": kind,
            "model": model,
            "usage": usage or None,
            "reported_model": result.get("model") if isinstance(result.get("model"), str) else None,
        }
    )
