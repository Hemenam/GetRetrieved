# Local verification

Verification performed against the provided course on 13 September 2026.

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
