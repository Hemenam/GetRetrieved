# Planner translation contract: 503 diagnosis and fix

## Confirmed cause

The unchanged v4 pilot was rerun into
`evals/results/pilot-v4-recheck-20261017-105828`: **22 HTTP 503 errors**
(full context 8/20, hybrid 7/20, reranked hybrid 7/20), with no client timeout.
The previous captured run and approved gold were not overwritten.

Three subsequent scoped diagnostic queries reproduced the same local failure:
pilot-003 in hybrid, pilot-005 in full context, and pilot-010 in hybrid.
All three provider completions finished normally, but supplied a non-null
`persian_query` while `translate_to_persian=false`. `validate_plan()` then raised
`ModelUnavailable("Unexpected search translation")`, mapped to HTTP 503.
This establishes a concrete application-contract defect, not a provider outage.
Old error-only reports do not establish that every historical 503 had this cause.

The old schema allowed `persian_query` to be either a string or null on every request.
The prompt requested null when translation was disabled, but a string was still
schema-valid. Strict generation therefore did not enforce the request's setting.

## Implementation

- `schemas.py`: `UntranslatedQuestionPlan` requires JSON null;
  `TranslatedQuestionPlan` requires a nonempty string. Both inherit the unchanged
  internal `QuestionPlan` interface.
- `models.py`: select the matching type **before generation** and validate locally
  with that same type. The actual schema, not just a prose instruction, now enforces
  the translation setting for hosted and Ollama gateways. Ollama retains its
  existing 2048-token planning allowance.
- `grounding.py`: keep the translation and evidence safety checks intact. There
  is no silent fallback, weakening of the relevance gate, or automatic paid retry.
- `evaluation.py` and `models.py`: add sanitized stage/error-category telemetry,
  distinguishing provider HTTP status, timeout, transport, invalid response,
  malformed structured output, and planner invariants. Exception text, provider
  bodies, URLs, questions, source content and API credentials are not logged.
  Client-facing error bodies remain generic.

This follows the [official structured-output contract](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat):
required nullable fields allow both values; schema adherence alone is not semantic correctness.

No model migration, course edits, reindexing, `.env` changes, threshold adjustment,
reranker-pool changes or benchmark-label changes were made for this fix.
Policy remains v4; the captured pipeline hash changes with the code. Do not merge
pre-fix and post-fix measurements into a fixed-configuration experiment.

## Verification

- **596 Python tests passed**, including the private complete DOCX ingestion checks.
- Added 12 planner boundary regressions covering translation on/off, valid/invalid
  values, hosted strict schema, hosted JSON-object mode, and Ollama.
- Added 8 diagnostics tests covering redaction and provider-error classification.
- Ruff checks and changed-file formatting checks passed.
- **29 frontend tests passed** (Node test isolation disabled for this sandbox).
- A targeted live HTTP smoke test uses pilot-003, pilot-005 and pilot-010 in all
  three modes (nine requests maximum), saved separately under
  `evals/results/planner-fix-smoke-20261017-111459`.

All **nine live API requests completed without HTTP errors or timeouts**: eight
returned answers and one returned a refusal. Six passed automated benchmark checks;
all three pilot-010 outcomes still failed. Full-context and reranked answers omitted
the manager-fear evidence, while hybrid refused the answerable question. These are
not successful semantic answers merely because HTTP 200 was returned. The cold
reranked pilot-003 request took 214.3 seconds; the other two took 55.3 and 30.2 seconds.
The temporary test server was stopped after collection.

The live smoke test checks this error regression, not the complete 20-case cohort
or a final architecture ranking. Multi-part answer completeness, planner intent
expansion, semantic verifier errors and CPU reranker latency remain separate quality
issues; eliminating HTTP errors does not demonstrate they are solved.

Restart any already-running application server to load the fix; reindexing is not needed.
