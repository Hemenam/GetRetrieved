"""Strict per-request source budgets and evidence references, not semantic answerability."""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import Field, create_model

from hrlearnium.schemas import (
    ClarificationEvidenceDecision,
    EvidenceDecisionEnvelopeBase,
    RefusedEvidenceDecision,
    StrictModel,
)


def decision_model(candidate_ids: list[str], requirement_count: int, max_sources: int):
    """Each answered branch has exactly N sources and valid 0..N-1 references.

    Fixed requirement keys prevent duplicate/missing requirement indices. One shared
    enum avoids multiplying the provider's enum budget by the number of branches.
    All source facts remain in the separately supplied authorized passages.
    """
    ids = list(dict.fromkeys(candidate_ids))
    if not ids or not 1 <= requirement_count <= 8 or not 1 <= max_sources <= 8:
        raise ValueError("Invalid evidence contract dimensions")
    source_type = Enum(
        "CandidateSourceId", {f"S{i}": value for i, value in enumerate(ids)}, type=str
    )
    branches = []
    for count in range(1, min(max_sources, len(ids)) + 1):
        index_type = Annotated[int, Field(strict=True, ge=0, le=count - 1)]
        coverage = create_model(
            f"EvidenceMap{count}Sources{requirement_count}Parts",
            __base__=StrictModel,
            **{
                f"r{i}": (list[index_type], Field(min_length=1, max_length=count))
                for i in range(requirement_count)
            },
        )
        branch = create_model(
            f"AnswerWith{count}Sources",
            __base__=StrictModel,
            status=(Literal["answered"], ...),
            reason_code=(Literal["supported"], ...),
            source_ids=(list[source_type], Field(min_length=count, max_length=count)),
            coverage=(coverage, ...),
        )
        branches.append(branch)
    branches.extend((RefusedEvidenceDecision, ClarificationEvidenceDecision))
    return create_model(
        "EvidenceDecisionEnvelope",
        __base__=EvidenceDecisionEnvelopeBase,
        decision=(Union[tuple(branches)], ...),
    )


def brief_decision_model(candidate_ids, max_sources, max_facts=8):
    """Fixed-size source branches make the global fact-source budget enforceable."""
    ids = list(dict.fromkeys(candidate_ids))
    if not ids or not 1 <= max_sources <= 8:
        raise ValueError("Invalid brief budget")
    source_type = Enum("BriefSourceId", {f"S{i}": v for i, v in enumerate(ids)}, type=str)
    branches = []
    for count in range(1, min(max_sources, len(ids)) + 1):
        index = Annotated[int, Field(strict=True, ge=0, le=count - 1)]
        fact = create_model(
            f"FactFrom{count}Sources",
            __base__=StrictModel,
            text=(str, Field(min_length=1, max_length=1000)),
            source_indices=(list[index], Field(min_length=1, max_length=count)),
        )
        value = create_model(
            f"ValueFrom{count}Sources",
            __base__=StrictModel,
            value=(str, Field(min_length=1, max_length=200)),
            source_index=(index, ...),
        )
        branch = create_model(
            f"BriefWith{count}Sources",
            __base__=StrictModel,
            status=(Literal["answered"], ...),
            reason_code=(Literal["supported"], ...),
            source_ids=(list[source_type], Field(min_length=count, max_length=count)),
            facts=(list[fact], Field(min_length=1, max_length=max_facts)),
            requested_values=(list[value], Field(max_length=8)),
        )
        branches.append(branch)
    branches.extend((RefusedEvidenceDecision, ClarificationEvidenceDecision))
    return create_model(
        "BoundedEvidenceBrief",
        __base__=EvidenceDecisionEnvelopeBase,
        decision=(Union[tuple(branches)], ...),
    )
