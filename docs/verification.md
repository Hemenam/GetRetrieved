# Local verification

## Current v3 checks — 8 October 2026

- Python: **485 passed, 1 skipped** (private-DOCX regression requires its external fixture).
  Two existing test-client deprecation warnings remain. Ruff and `git diff --check` pass.
- Browser client: **20 Node.js tests passed**, including conversational responses and rejection
  of a missing explanation instead of silently displaying source chunks.
- Local HTTP: API 0.3.0 runs on port 8000; OpenAPI defaults to explained and includes conversation status.
- Configured hosted `gpt-4.1-mini`, full-context, explained mode: **7/7 smoke cases passed**
  status, response-mode, rendering, evidence-coverage and citation checks. Greetings made zero
  model calls; supported answers made three calls; unrelated and missing-information questions
  were refused; an unresolved reference asked for clarification without model inference.
- The first smoke run passed 6/7 and exposed an incorrect first-passage summary for
  `این را توضیح بده.` with no history. A narrow reference guard and stricter prompts were added;
  the repeat passed 7/7. Both local reports remain under ignored `evals/results/`:
  `chat-behavior-full-context.json` (initial), `chat-behavior-full-context-v3.json` (repeat).
- The two generated repeat answers were inspected against their cited chapter-one passages:
  they name reality, organizational action and the members' next step without a whole-chunk prefix.
  This is a small development smoke check, not instructor approval or broad semantic validation.
- The runtime remains full-context with no embedding model configured. Hybrid and optional raw
  reranker floors have controlled integration tests, **not** live-model calibration. Database,
  document import and chunking were not changed for this upgrade.

See [relevance gating](relevance-gating.md) for settings and the reproducible smoke command.
Use the full reviewed course/held-out evaluations before learner rollout.

## Earlier v2 baseline — 13 September 2026

The following results and configuration describe that earlier run, not the current runtime.

| Check | Result |
| --- | --- |
| DOCX import | 11 chapters, 242 nonempty paragraphs, 44 passages |
| Source integrity | All 44 stored source spans and source-text hashes verified |
| Infrastructure test suite | 370 passing tests, including the private-source regression and v2 response modes |
| Code checks | Ruff passes; installed dependencies are consistent |
| Actual HTTP smoke | Passed against FastAPI on localhost in literal lookup mode |
| Exact HTTP response | 611-character chapter 4 excerpt matched original source and separately resolved citation |
| Authorization smoke | Missing credentials returned 401; unauthorized course returned 403 |
| Lexical retrieval evaluation | 60/60 answerable cases had all required evidence within the top 8 candidates |
| Evidence coverage | 66/66 expected chapter checks and 102/102 exact quote anchors found |
| V2 localhost Swagger | API 0.2.0 and `response_mode: verbatim / explained` served successfully |
| Explanation boundaries | Controlled tests cover citation validation, negative support checks, provider failures and source changes during generation |
| Hosted provider adapter | Mocked Chat Completions/embeddings tests pass; no live API call performed |
| Hosted embedding revision | Regression verifies explicit revision changes save new vectors instead of deduplicating them |

The full retrieval report is generated locally at `evals/results/retrieval_baseline.json`; runtime reports are excluded from version control. The 100 evaluation cases are proposed and have not been approved by the instructor.

The 60-case result measures lexical candidate coverage. It does **not** measure whether a live model selects the correct excerpts. The remaining 40 negative/ambiguous cases have not been scored with a live semantic selector. Ollama and hosted adapter behavior is tested with controlled HTTP responses, including failures and malformed selections. Model explanations and their verification use mocked responses in automated tests; generated semantic correctness remains unverified with a real model. No local model has been downloaded or activated here.

The local `.env` is still configured for `literal` diagnostic lookup. Blank hosted configuration entries have been added, but no provider URL, model or key is configured. Natural-language Q&A can use a configured hosted API or local Ollama. Full-context mode can use the existing 44 passages directly; hybrid mode requires embedding reindexing. See the [main README](../README.md) for API-key configuration and the [setup guide](setup-and-deployment.md) for local models. Run API evaluation in both response modes with the actual model and review generated explanations against their citations before learner rollout.

Docker configuration and CI files are included. The Docker engine was not available for a container build/run check here. The native Python service was run and checked directly.
