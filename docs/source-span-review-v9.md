# V9: server-owned support spans and bounded evidence recovery

Implemented after the stopped v8 failed-question rerun. This implementation has **not** had a paid benchmark rerun. Prior captures, references, results and the three-consecutive-error stop guard are unchanged. Offline tests establish contracts, not live model accuracy.

## Changes

1. The shipped OpenAI-compatible and Ollama gateways now implement `review_grounded_answer`. The server makes deterministic contiguous source slices (at most 600 characters each) with text-bound IDs and exact offsets. The model selects IDs; it cannot introduce a quote field. Its dynamic schema constrains choices to the provided catalog. Local checks also reject unknown, duplicate, missing and non-cited-source support, including in JSON-object mode.
2. The reviewer returns independently checked booleans for original-question relevance, actor/action direction, source stance and causal support. All must pass, alongside per-statement evidence and complete original-question coverage. A real source ID alone does not establish that a claim follows from it.
3. In plain hybrid mode, recovery context comes from the same positive, relevance-filtered ranking before the initial `candidate_limit` cutoff, bounded by `rerank_pool_limit` and `max_context_characters`. The original selector still receives its normal top-k pool. The reviewer can see the bounded recovery context to notice a more relevant or contradictory topic; the existing one-shot repair can replace selected excerpts from this pool. No new retrieval, embedding call, cross-course source, threshold bypass or unbounded loop is introduced. Full-context mode uses its existing bounded corpus; reranked mode uses only its admitted post-rerank pool.
4. Explicit first/last chapter-title requests resolve from the current authorized outline rather than a model-generated chapter number. The same full-match structural guard applies inside multipart routes. Returned metadata is canonical `chapter_title` with its actual chapter number. Multiple source documents have no assumed global order and return metadata unavailable. Content questions that merely mention the last chapter remain content.
5. Repair and verification failures remain distinct from evidence-based refusals. They cannot silently publish a rejected draft or erase successfully answered independent siblings. Source version/access and statement citation-budget checks are retained.

The ordinary successful content path still has five chat calls (routing, route verification, selection, writing, review), plus hybrid embeddings. Larger review/recovery context can increase token costs; one repair remains the maximum. The model and provider settings, DB, chunking and search floors were not changed.

## Compatibility and diagnostics

The public answer structure is unchanged; `policy_version` becomes `grounded-course-v9`. The two relative metadata operations are internal planning fields. `answer_attempts` records `review_protocol=server_span_ids_v9` for the new reviewer and labels old adapters as legacy. Evaluation configuration and code/prompt hashes include these changes, including the source-span builder. `answer_recovery_candidates` is recorded in opt-in hybrid evaluation traces.

Old custom gateway adapters without `review_grounded_answer` retain their existing quote/boolean reviewer until migrated. The legacy `review_answer` method remains compatible but is not the new production gateway path. Review/draft diagnostics stay opt-in and authorized, not public HTTP error details.

## Validation and remaining uncertainty

Offline regression tests exercise real/literal newlines, long source lines, span identity/version binding, invented and non-cited spans, independent intent failures, provider schemas for all three gateway formats, one-shot repair/withholding in every mode, rank-14 recovery, threshold exclusion, multipart metadata routing, multiple documents and source revisions. HTTP integration tests drive the shipped gateway through mocked provider responses.

The micromanagement failure had its relevant explanation at rank 14, outside the initial eight candidates. This change makes bounded eligible evidence available to review/repair; it does not prove the model will always interpret it correctly. Role/stance checks and entailment remain probabilistic. Broad course summaries also need semantic breadth review and a rubric that allows equivalent summaries; no expected answers or exact-anchor checks were loosened to obtain a higher score.

Restart running LMS workers/standalone servers to load the new code. Next validation: a fresh, separately captured run on the unchanged failed-question set; do not merge old successes into a new full-suite accuracy number.

The constrained ID-based response schema follows the official [Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs). Schema adherence enforces output shape, not semantic truth.
