"""Precise fact evidence and writer references; ownership is not semantic proof."""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import Field, create_model

from hrlearnium.schemas import (
    ClarificationEvidenceDecision,
    EvidenceBrief,
    EvidenceDecisionEnvelopeBase,
    Explanation,
    RefusedEvidenceDecision,
    StrictModel,
)
from hrlearnium.source_spans import source_spans


class FactExplanationBase(StrictModel):
    """Marker for bounded writer schemas referencing approved facts, not citations."""


def precise_brief_model(candidate_ids, spans, max_sources):
    ids = list(dict.fromkeys(candidate_ids))
    if not ids or not spans or not 1 <= max_sources <= 8 or len(spans) > 450:
        raise ValueError("Invalid fact evidence catalog")
    source_type = Enum("FactSourceId", {f"S{i}": value for i, value in enumerate(ids)}, type=str)
    span_type = Enum("FactSpanId", {f"P{i}": s["id"] for i, s in enumerate(spans)}, type=str)
    fact = create_model(
        "PreciselySupportedFact",
        __base__=StrictModel,
        text=(str, Field(min_length=1, max_length=1000)),
        span_ids=(list[span_type], Field(min_length=1, max_length=16)),
    )
    branches = []
    for count in range(1, min(max_sources, len(ids)) + 1):
        value = create_model(
            f"LiteralValue{count}",
            __base__=StrictModel,
            value=(str, Field(min_length=1, max_length=200)),
            source_index=(Annotated[int, Field(strict=True, ge=0, le=count - 1)], ...),
        )
        branches.append(
            create_model(
                f"PreciseBrief{count}",
                __base__=StrictModel,
                status=(Literal["answered"], ...),
                reason_code=(Literal["supported"], ...),
                source_ids=(list[source_type], Field(min_length=count, max_length=count)),
                facts=(list[fact], Field(min_length=1, max_length=8)),
                requested_values=(list[value], Field(max_length=8)),
            )
        )
    branches.extend((RefusedEvidenceDecision, ClarificationEvidenceDecision))
    return create_model(
        "PreciseEvidenceDecision",
        __base__=EvidenceDecisionEnvelopeBase,
        decision=(Union[tuple(branches)], ...),
    )


def normalize_precise_brief(decision, spans):
    from hrlearnium.models import ModelUnavailable

    if decision["status"] != "answered":
        return EvidenceBrief(
            status=decision["status"], reason_code=decision["reason_code"], complete=False, facts=[]
        )
    ids = decision["source_ids"]
    catalog = {s["id"]: s for s in spans}
    used = set()
    if len(ids) != len(set(ids)):
        raise ModelUnavailable("Duplicate brief sources", code="invalid_evidence_brief")
    for fact in decision["facts"]:
        refs = fact["span_ids"]
        if len(refs) != len(set(refs)) or any(
            s not in catalog or catalog[s]["citation_id"] not in ids for s in refs
        ):
            raise ModelUnavailable(
                "Fact spans do not belong to selected sources", code="invalid_evidence_brief"
            )
        used.update(catalog[s]["citation_id"] for s in refs)
    if used != set(ids):
        raise ModelUnavailable("Brief contains unused sources", code="invalid_evidence_brief")
    return EvidenceBrief(
        status="answered",
        reason_code="supported",
        complete=True,
        facts=decision["facts"],
        requested_values=[
            {"value": v["value"], "source_id": ids[v["source_index"]]}
            for v in decision["requested_values"]
        ],
    )


def fact_explanation_model(fact_count):
    if not 1 <= fact_count <= 8:
        raise ValueError("Invalid approved fact count")
    index = Annotated[int, Field(strict=True, ge=0, le=fact_count - 1)]
    statement = create_model(
        "FactBoundStatement",
        __base__=StrictModel,
        text=(str, Field(min_length=1, max_length=1200)),
        fact_indices=(list[index], Field(min_length=1, max_length=fact_count)),
    )
    return create_model(
        "FactBoundExplanation",
        __base__=FactExplanationBase,
        statements=(list[statement], Field(min_length=1, max_length=8)),
    )


def bind_writer_facts(wire, brief, excerpts):
    from hrlearnium.models import ModelUnavailable

    spans = {s["id"]: s for s in source_spans(excerpts)}
    statements, mappings = [], []
    for row in wire.model_dump()["statements"]:
        refs = row["fact_indices"]
        if len(refs) != len(set(refs)) or any(
            type(i) is not int or not 0 <= i < len(brief.facts) for i in refs
        ):
            raise ModelUnavailable(
                "Invalid writer fact references", code="invalid_answer_citations"
            )
        support = list(dict.fromkeys(s for i in refs for s in brief.facts[i].span_ids))
        if not support or any(s not in spans for s in support):
            raise ModelUnavailable("Writer fact evidence changed", code="invalid_answer_citations")
        citations = list(dict.fromkeys(spans[s]["citation_id"] for s in support))
        statements.append({"text": row["text"], "citation_ids": citations})
        mappings.append({"fact_indices": refs, "span_ids": support})
    return Explanation(statements=statements), mappings
