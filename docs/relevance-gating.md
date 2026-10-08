# Relevance gates and natural answers

Policy version `grounded-course-v3` separates **search relevance**, **answerability** and
**answer writing**. A passage scoring well in search does not necessarily contain the answer.
The database, imported source text and chunking strategy are unchanged.

## Current pipeline

1. Authenticate the tenant/user/course and check ownership of any conversation ID.
2. Recognize a narrow set of whole-message greetings, thanks and help requests. Reply briefly
   without retrieval, an LLM call or citations. Do not add social turns to the answered-question
   history. A greeting plus a substantive question is not swallowed by this shortcut.
3. Collect evidence using the selected retrieval mode:
   - `full_context`: provide all authorized current passages within the context-size limit.
     There is no retriever score in this mode. The LLM must select only passages that answer
     the actual question, or refuse/ask for clarification.
   - `hybrid`: rank using BM25 and embedding cosine similarity with reciprocal-rank fusion
     (RRF), apply the raw-score floors below, then take the top eligible candidates.
   - `hybrid_rerank`: apply the same search floors, build the larger reranking pool, then
     optionally filter by a minimum raw cross-encoder score before taking its top candidates.
4. If filtering leaves no evidence, return an insufficient-evidence refusal without asking
   the answer writer to fill the gap. Otherwise the LLM evidence selector checks semantic
   answerability: topical overlap or a high search score alone is not enough.
5. Validate selected IDs, course authorization, active source revisions and exact source spans.
6. In the default `explained` mode, the LLM writes a direct, natural answer using those excerpts.
   Every statement cites evidence. A separate LLM call checks responsiveness and grounding;
   unsupported output is withheld, not replaced with raw chunks.
7. Return generated statements with citation markers as `answer`, and exact quotations separately
   in `excerpts`. The chat UI opens evidence on demand. Explicit `verbatim`/Source text still
   returns quotations instead of a generated answer.

An unrelated question receives an outside-course refusal when the selector identifies it.
A related question with missing information receives an insufficient-evidence refusal; the
system should not claim a topic is unrelated merely because retrieval found nothing. Provider
errors remain HTTP 503, not content refusals. Source-support checks are model judgments, not
a mathematical guarantee of correctness.

## Adjustable minimum scores

Add or change these settings in the server's `.env`, then restart the server:

```dotenv
HR_RETRIEVAL_MIN_COSINE=0.30
HR_RETRIEVAL_MIN_BM25=1.0
```

These are **provisional defaults, not calibrated confidence thresholds**. A hybrid candidate
is eligible when its positive cosine score meets the cosine floor **OR** its positive BM25
score meets the BM25 floor. The OR rule preserves exact-name/term matches that embeddings
might miss, and semantic matches with different wording. The LLM still checks the surviving
evidence against the question. Zero-score candidates are excluded even if the floors are zero.

Cosine values depend on the embedding model, language and course; BM25 depends on query terms,
document frequencies and passage lengths. `0.30` does **not** mean 30% confidence. RRF is used
only for ordering eligible results: its rank-dependent score is not an absolute relevance gate.

`HR_RERANKER_MIN_SCORE` is optional and unset by default. Set it only after measuring the raw
score distribution of the exact cross-encoder you use. Different rerankers have different scales,
and their outputs are not automatically probabilities. No reranker download or model change is
performed by this upgrade. The existing optional runtime is still required for `hybrid_rerank`.

Full-context mode ignores search floors because it does not run a retriever. It continues to
use LLM selection and explanation verification. Switching to hybrid still requires an embedding
model and reindexing; thresholds do not create embeddings or enable that mode automatically.

## Calibrate on the course

1. Assemble instructor-reviewed questions with relevant passage IDs: direct answers,
   paraphrases, numbers/names, multi-passage answers, related-but-unanswered questions and
   unrelated questions. Keep a held-out set separate from tuning cases.
2. Record raw cosine/BM25/reranker scores and compare gate settings. Measure relevant-passage
   recall, rejected necessary evidence, candidate noise and the end-to-end false-answer and
   false-refusal rates. Do not optimize only for a shorter candidate list.
3. Pick settings meeting agreed course-quality targets, then run the unchanged held-out set.
   Review generated answers against their cited passages; status/citation tests cannot grade meaning.
4. Repeat after changing the embedding model, reranker, course or chunking. Store model identities,
   settings, source hashes and application revision with each report.

Enable **Test details** in chat to inspect `configuration.relevance_thresholds`, raw
`retrieved_candidates`, the `relevance_filter` before/after counts, `eligible_candidates`, and
the final `selector_candidates`. Sensitive evidence stays behind the existing authenticated API.

## Compatibility and checks

Version 3 changes the API default from `verbatim` to `explained`, removes quote prefixes from
explained `answer`, and adds `status="conversation"`. The included browser UI and Django connector
handle the new contract. Older integrations must update their render/validation logic or explicitly
request verbatim while adapting. Exact citation evidence remains unchanged.

`tests/test_chat_behavior.py` covers score boundaries, lexical rescue, weak-match refusals,
greetings, mixed greeting/questions, ownership and separate natural-answer/evidence rendering.
`evals/chat_behavior.jsonl` supplies a small proposed real-model smoke set; it is not instructor
approved and is not a replacement for the full course evaluation.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --mode api --dataset evals/chat_behavior.jsonl --local-auth --tenant customer-demo --course captain-storm --subject chat-behavior-smoke --response-mode explained --expect-retrieval-mode full_context --output evals/results/chat-behavior-full-context-v3.json
```

The separation of retrieval failures from model-response failures follows the
[OpenAI accuracy guide](https://developers.openai.com/api/docs/guides/optimizing-llm-accuracy).
The selector/writer/verifier prompts now include explicit request-boundary examples and relevant
context, consistent with the [prompt engineering guide](https://developers.openai.com/api/docs/guides/prompt-engineering).

See [evaluation guidance](../evals/README.md) and [the retrieval experiment guide](retrieval-experiments.md)
for broader comparisons. Passing a small smoke test does not prove general semantic accuracy.
