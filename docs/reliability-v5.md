# Reliability and evidence alignment — policy v5

## Changes

1. **Bounded transport recovery.** The hosted gateway retries a transient connect/read/write/remote
   protocol failure once by default, with a 0.5-second backoff. `HR_API_MAX_RETRIES` accepts 0–2;
   `HR_API_RETRY_BACKOFF_SECONDS` accepts 0–5. A retry is not started after the configured call budget
   has elapsed; each attempt's HTTP timeout uses the remaining budget. HTTPX timeouts are per I/O phase,
   not a hard whole-request wall-clock deadline. Authentication/HTTP errors, rate limits, invalid JSON,
   schema errors and semantic refusals are not retried. The benchmark runner and frontend still never
   retry failed HTTP queries. Transport failures cannot be made impossible by application code.
2. **Selection made consistent by construction.** The native selector returns an object containing a
   nested answered/refused/clarification decision union. Refusals and clarifications have no evidence
   fields. Answered decisions provide one globally bounded source list and per-requirement indices
   into that list; Python derives the canonical per-part ID map. Each answered schema branch has
   exactly N sources (1–4 with the default limit), index references constrained to 0..N-1, and fixed
   required keys r0, r1, etc. The old independently generated lists could disagree or exceed the
   **combined** source budget even when every part's own list satisfied its individual limit.
   Candidate authorization, source checks, per-part coverage, source-count limits and fail-closed
   validation remain in place. The old pizza-case invariant was not retained and did not recur in
   two diagnostic queries. A separate multi-part failure in the intermediate v5 run was reproduced
   twice: the selector supplied six and five unique sources against a limit of four. The final
   schema prevents this before generation; two post-fix selector checks each supplied four valid sources.
3. **Shared evidence contract.** The writer and verifier now receive the same requirement-to-source map
   already enforced by the server. Prompts require each part's answering statement to cite its assigned
   evidence. Previously the server could reject citation alignment the models had never been told to
   follow. The original-question and semantic-support checks remain mandatory. Planning instructions
   also discourage invented subquestions and changing who a question concerns. Semantic correctness
   remains model-dependent, not guaranteed by these structural checks.

These changes use the [official structured-output contract](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat)
and bounded recovery informed by [official API error/retry guidance](https://developers.openai.com/api/reference/python).
We retain the existing configured model and custom HTTPX transport rather than migrate SDKs or models.

## Diagnostics and billing

`configuration.provider_retry_policy` records the retry settings. Request-local `provider_attempts`
records every observed attempt and its stage, outcome and retry decision. Successful completions still
appear once in `provider_calls`; failures without usage do not become zero-token calls. Runtime summaries
add observed attempt, failed-attempt and retry counts. Recovered ambiguous failures prevent a complete
USD estimate, even if the successful call has reported usage. Earlier stages of an entirely failed query
can still be absent from returned diagnostics. Neither a query cap nor a token subtotal is a spend cap.

No exception text, provider body, key, URL or course passage is written into error logs. Client error
responses remain generic. There are no hidden schema-repair or answer-regeneration loops.

## Compatibility and scope

- Native OpenAI-compatible and Ollama gateways support the new wire decision shape. Public query/response
  fields are unchanged apart from the `grounded-course-v5` policy label.
- Custom gateways must accept `evidence_map=` on `explain_complete` and `verify_complete`. Direct legacy
  selection/explanation methods remain available. The local `.env` was not changed.
- No document, DB architecture, chunking, embedding identity, retrieval floor, source cap, reranker model,
  pool size, approved gold label or previous benchmark capture was changed.
- An offline CPU probe with four rather than fourteen Torch threads preserved the same top-eight IDs
  (maximum score difference about 0.0000025), but cold scoring still took 145.3 seconds versus a captured
  baseline rerank stage of 158.3 seconds. These were not controlled repetitions. That is insufficient
  evidence for a general performance fix, so the production CPU/thread configuration is unchanged.
  Reranker latency remains a measured architecture tradeoff, not a solved problem.

Use `evals/pilot/manifest-v5-pilot.json` for a fresh 20-case, three-mode, one-repetition comparison.
Preserve prior reports and keep v4/v5 experiments separate because prompts/code/retry policy differ.
The intermediate v5 capture (`pilot-v5-20261018-213110`) predates the global-budget fix and is also
separate from the final v5 capture: use their pipeline/prompt hashes, not only the policy label.
Restart an existing server to activate v5; no reindex is needed.

## Final pilot and version boundary

The final capture is `evals/results/pilot-v5-final-20261018-215934`; see the
[comparison and limitations](../evals/results/pilot-v5-final-summary-20261018-215934/observations.md)
and [all actual responses](../evals/results/pilot-v5-final-summary-20261018-215934/responses.md).
It returned no HTTP 503, but retained one interrupted client timeout and a shared multi-part
semantic weakness. Automated passes were 18/20 full context, 19/20 hybrid and 19/20 reranked;
these are not human semantic grades or evidence that all problems are solved.

After that collection, an additional pre-attempt deadline check was added to handle a host pause
during retry backoff. It prevents an expired retry from starting. This timing-only change was
verified offline (620 Python tests and 29 frontend tests pass), not by another full paid cohort.
The final capture's recorded pipeline hash therefore precedes that additional guard.
