"""Request-local diagnostics, explicitly enabled by the evaluation runner."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from hrlearnium.policy import (
    ANSWER_REPAIR_INSTRUCTIONS,
    ANSWER_REVIEW_INSTRUCTIONS,
    BRIEF_WRITING_INSTRUCTIONS,
    COMPLETE_SELECTION_INSTRUCTIONS,
    COMPLETE_VERIFICATION_INSTRUCTIONS,
    EVIDENCE_BRIEF_INSTRUCTIONS,
    EXPLANATION_INSTRUCTIONS,
    LANGUAGE_REPAIR_INSTRUCTIONS,
    PLAN_VERIFICATION_INSTRUCTIONS,
    PLANNING_INSTRUCTIONS,
    REQUEST_PLAN_REPAIR_INSTRUCTIONS,
    REQUEST_PLAN_VERIFICATION_INSTRUCTIONS,
    REQUEST_PLANNING_INSTRUCTIONS,
    ROUTE_VERIFICATION_INSTRUCTIONS,
    ROUTING_INSTRUCTIONS,
    SELECTOR_INSTRUCTIONS,
    SPAN_REVIEW_INSTRUCTIONS,
    UNIT_ROUTING_INSTRUCTIONS,
    VERIFICATION_INSTRUCTIONS,
)
from hrlearnium.schemas import POLICY_VERSION

_trace: ContextVar[dict | None] = ContextVar("course_evaluation", default=None)
_stage: ContextVar[str | None] = ContextVar("course_evaluation_stage", default=None)
logger = logging.getLogger("hrlearnium.pipeline")
ERROR_CODES = frozenset(
    {
        "model_unavailable",
        "provider_http_error",
        "provider_timeout",
        "provider_transport_error",
        "provider_invalid_response",
        "structured_response_invalid",
        "invalid_question_plan",
        "invalid_search_translation",
        "invalid_answer_language",
        "answer_verification_failed",
        "invalid_evidence_brief",
        "invalid_answer_review",
        "answer_source_budget_exceeded",
        "invalid_answer_citations",
    }
)


def configuration(settings, reranker=None) -> dict:
    prompts = [
        PLANNING_INSTRUCTIONS,
        PLAN_VERIFICATION_INSTRUCTIONS,
        SELECTOR_INSTRUCTIONS,
        COMPLETE_SELECTION_INSTRUCTIONS,
        EXPLANATION_INSTRUCTIONS,
        VERIFICATION_INSTRUCTIONS,
        COMPLETE_VERIFICATION_INSTRUCTIONS,
        REQUEST_PLANNING_INSTRUCTIONS,
        REQUEST_PLAN_REPAIR_INSTRUCTIONS,
        REQUEST_PLAN_VERIFICATION_INSTRUCTIONS,
        LANGUAGE_REPAIR_INSTRUCTIONS,
        ROUTING_INSTRUCTIONS,
        ROUTE_VERIFICATION_INSTRUCTIONS,
        ANSWER_REPAIR_INSTRUCTIONS,
        ANSWER_REVIEW_INSTRUCTIONS,
        UNIT_ROUTING_INSTRUCTIONS,
        EVIDENCE_BRIEF_INSTRUCTIONS,
        BRIEF_WRITING_INSTRUCTIONS,
        SPAN_REVIEW_INSTRUCTIONS,
    ]
    code = hashlib.sha256()
    for name in (
        "service",
        "models",
        "policy",
        "schemas",
        "ingestion",
        "retrieval",
        "reranking",
        "grounding",
        "selection_contract",
        "course_metadata",
        "language",
        "request_planning",
        "source_spans",
        "source_rules",
        "request_units",
        "answer_contracts",
        "fact_contracts",
        "request_modifiers",
    ):
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
        "provider_retry_policy": {
            "max_retries": settings.api_max_retries if hosted else 0,
            "backoff_seconds": settings.api_retry_backoff_seconds if hosted else 0,
            "scope": "transient transport only; no HTTP, schema or semantic retries",
        },
        "candidate_limit": settings.candidate_limit,
        "question_planning": {
            "plan_repair_limit": 1,
            "routing_protocol": "v12 indexed segments with bounded source-literal presentation attachments",
            "pre_retrieval_clarification": False,
            "routing_fallback": "whole original request, normal evidence gates, no invented execution plan",
            "answer_repair_limit_per_group": 1,
            "answer_review": "server-owned span IDs; explicit intent/actor/stance/cause checks; bounded eligible contrast context",
            "answer_recovery_pool_limit": settings.rerank_pool_limit,
            "answer_recovery_pool_scope": "hybrid: same relevance-gated ranking before top-k; reranked/full-context: existing admitted pool",
            "answer_verification_failure_is_source_absence": False,
            "pre_draft_evidence_brief": True,
            "source_offsets_visible_to_models": False,
            "internal_failures_are_clarifications": False,
            "enabled": settings.model_backend != "literal",
            "translate_search_queries": settings.translate_search_queries,
            "max_requirements": 8,
            "completeness_checks": "per-requirement evidence and answer coverage",
            "independent_execution": True,
            "partial_answers": True,
            "max_independent_parts": 8,
            "evidence_budget": "max_excerpts per part, at most 8 parts",
            "typed_intent_routing": True,
            "original_text_groups": True,
            "global_independent_switch": False,
            "response_language": "request-scoped FA/EN, explicit preference before original-script fallback",
            "language_repair_limit_per_group": 1,
        },
        "relevance_thresholds": {
            "min_cosine": settings.retrieval_min_cosine,
            "min_bm25": settings.retrieval_min_bm25,
            "search_gate": "cosine OR bm25; positive score required",
            "min_reranker_score": settings.reranker_min_score,
            "applies_to": ["hybrid"],
            "rerank_admission": "positive ranked matches, bounded pool; no embedding floor",
            "calibrated_confidence": False,
        },
        "max_excerpts": settings.max_excerpts,
        "overview_max_excerpts": settings.overview_max_excerpts,
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
    except Exception as error:
        if value is not None:
            child = getattr(error, "evaluation_trace", None)
            if isinstance(child, dict) and child is not value:
                value["failed_part_trace"] = child
                for field in ("provider_calls", "provider_attempts"):
                    value.setdefault(field, []).extend(child.get(field, []))
            # Internal diagnostics only. HTTP error handlers still return a fixed,
            # redacted message; the authorized benchmark can account for failed spend.
            error.evaluation_trace = value
        raise
    finally:
        _trace.reset(token)


@contextmanager
def stage(name: str):
    token = _stage.set(name)
    started = time.perf_counter()
    try:
        yield
    except Exception as error:
        # Never log exception strings, provider bodies, questions, source text or URLs.
        code = getattr(error, "diagnostic_code", None)
        code = code if isinstance(code, str) and code in ERROR_CODES else "unclassified"
        status = getattr(error, "provider_status", None)
        status = status if type(status) is int and 400 <= status <= 599 else None
        logger.warning(
            "pipeline_stage_failed stage=%s code=%s provider_status=%s", name, code, status
        )
        raise
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


def request_plan(plan):
    trace = _trace.get()
    if trace is not None:
        value = plan.model_dump()
        for index, group in enumerate(value["groups"]):
            group["question_id"] = f"q{index + 1}"
        trace["request_plan"] = value


def request_plan_verification(check):
    trace = _trace.get()
    if trace is not None and hasattr(check, "model_dump"):
        trace["request_plan_verification"] = check.model_dump()


def request_plan_attempt(attempt, plan, failure, check):
    # Log codes only even when evaluation capture is disabled: no learner/source/provider text.
    if failure:
        logger.warning("request_plan_attempt=%d failure=%s", attempt + 1, failure)
    trace = _trace.get()
    if trace is not None:
        trace["routing_protocol"] = "legacy_v7"
        trace.setdefault("request_plan_attempts", []).append(
            {
                "attempt": attempt + 1,
                "repair": bool(attempt),
                "failure": failure,
                "plan": plan.model_dump() if hasattr(plan, "model_dump") else None,
                "verification": check.model_dump() if hasattr(check, "model_dump") else None,
            }
        )


def language_result(target, initially_valid):
    trace = _trace.get()
    if trace is not None:
        trace["answer_language_check"] = {
            "target": target,
            "initial_script_check_passed": initially_valid,
            "repair_attempts": 0 if initially_valid else 1,
            "final_script_check_passed": True,
            "limitation": "Conservative FA/EN script check, not a fluency/semantic guarantee",
        }


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


def question_plan(plan, queries):
    trace = _trace.get()
    if trace is not None:
        trace["question_plan"] = plan.model_dump() | {"retrieval_queries": queries}


def part_result(index, question, part_trace, *, single=False):
    trace = _trace.get()
    if trace is None or part_trace is None:
        return
    if single:
        for field, value in part_trace.items():
            if field == "configuration":
                continue
            if field in {"provider_calls", "provider_attempts"}:
                trace.setdefault(field, []).extend(value)
            elif field == "stages_ms":
                trace[field].update(value)
            else:
                trace[field] = value
        return
    trace.setdefault("parts", []).append(
        {"index": index, "question": question, "trace": part_trace}
    )
    for field in ("provider_calls", "provider_attempts"):
        trace.setdefault(field, []).extend(
            dict(call, part_index=index) for call in part_trace.get(field, [])
        )
    for name, value in part_trace.get("stages_ms", {}).items():
        trace["stages_ms"][f"part_{index}.{name}"] = value


def requirement_selection(selection, status):
    trace = _trace.get()
    if trace is not None:
        trace["requirement_selection"] = selection.model_dump() | {"validated_status": status}


def unit_routing(units, proposed, plan):
    trace = _trace.get()
    if trace is not None:
        trace["routing_protocol"] = "v12_indexed_modifiers"
        trace["original_request_units"] = units
        trace["proposed_unit_routes"] = (
            proposed.model_dump() if hasattr(proposed, "model_dump") else None
        )


def evidence_brief(brief):
    trace = _trace.get()
    if trace is not None:
        trace["evidence_brief"] = brief.model_dump()
        trace.setdefault("evidence_briefs", []).append(brief.model_dump())


def requirement_verification(verification, accepted):
    trace = _trace.get()
    if trace is not None:
        trace["requirement_verification"] = verification.model_dump() | {"accepted": accepted}


def routing_attempt(attempt, route, failure, check):
    trace = _trace.get()
    if trace is not None:
        trace["routing_protocol"] = "v9"
        trace.setdefault("routing_attempts", []).append(
            {
                "attempt": attempt + 1,
                "repair": bool(attempt),
                "failure": failure,
                "route": route.model_dump() if hasattr(route, "model_dump") else None,
                "verification": check.model_dump() if hasattr(check, "model_dump") else None,
            }
        )


def routing_fallback(failure):
    trace = _trace.get()
    if trace is not None:
        trace["routing_fallback"] = {"reason": failure, "strategy": "whole_original_request"}


def answer_attempt(
    attempt, explanation, verification, accepted, *, review=None, local_failures=None
):
    trace = _trace.get()
    if trace is not None:
        trace.setdefault("answer_attempts", []).append(
            {
                "attempt": attempt + 1,
                "repair": bool(attempt),
                "accepted": accepted,
                "local_contract_failures": list(local_failures or []),
                "explanation": explanation.model_dump(),
                "verification": verification.model_dump(),
                "review_protocol": "consistent_server_span_ids_v11"
                if hasattr(review, "intent")
                else "legacy_quote_or_boolean_review",
                "review": review.model_dump() if review is not None else None,
            }
        )


def review_contract_check(attempt, review, issues, *, conservative_rejection=False):
    trace = _trace.get()
    if trace is not None:
        trace.setdefault("review_contract_attempts", []).append(
            {
                "attempt": attempt,
                "issues": list(issues),
                "review": review.model_dump(),
                "server_conservative_rejection": conservative_rejection,
            }
        )


def writer_fact_references(mappings):
    trace = _trace.get()
    if trace is not None:
        trace.setdefault("writer_fact_bindings", []).append(mappings)


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


def provider_attempt(model, kind, attempt, outcome, *, provider_status=None, will_retry=False):
    """Failed/ambiguous attempts may incur charges, even when no usage is returned."""
    code = outcome if outcome in ERROR_CODES or outcome == "success" else "model_unavailable"
    value = {
        "stage": _stage.get(),
        "kind": kind,
        "model": model,
        "attempt": attempt,
        "outcome": code,
        "provider_status": provider_status,
        "will_retry": will_retry,
    }
    trace = _trace.get()
    if trace is not None:
        trace.setdefault("provider_attempts", []).append(value)
    if code != "success":
        logger.warning(
            "provider_attempt_failed stage=%s attempt=%s code=%s will_retry=%s",
            _stage.get(),
            attempt,
            code,
            will_retry,
        )
