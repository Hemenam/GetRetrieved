# Retrieval and completeness upgrade

Policy `grounded-course-v4` addresses premature cross-language filtering and incomplete multi-part
answers found in the development pilot. Stored documents, chunking, embeddings, approved labels,
captured benchmark responses and human-review files are unchanged. The chapter-outline route is
not part of this upgrade. Live answer-quality improvements have not yet been measured.

## What changes

Every substantive LLM-backed question gets one structured planning call identifying up to eight
requirements. Hybrid English questions additionally receive a faithful Persian search translation
by default. Original, translated and component searches are embedded in one bounded batch, fused
by rank, and deduplicated by source ID. Presentation instructions such as “Please answer in English”
are excluded from search hints; the original question still controls selection and answer writing.
No translation or requirement text becomes evidence. This translation targets the current Persian
course contract; it is not automatic multilingual source-language detection.

Plain hybrid uses a provisional cosine floor of `0.24` or BM25 floor `1.0`, before its eight-candidate
shortlist. Reranked mode instead retains positive ranked matches up to `HR_RERANK_POOL_LIMIT`
(default 32), then scores them with the configured local reranker. Optional `HR_RERANKER_MIN_SCORE`
still applies after scoring. Similarity and raw reranker scores are not answer-confidence percentages.
Broader pools can increase CPU latency; no model files or hardware configuration are changed.

The selector returns evidence IDs for every requirement. The server refuses incomplete coverage
instead of writing an answer to just one part. Explained answers pass a separate structured check
of original-question coverage, every requirement's statement coverage, valid statement indices,
citation overlap with that part's evidence, and semantic support. A global `supported=true` flag
alone is insufficient. Model interpretation can still be wrong: these checks do not prove truth.
There are no hidden repair retries. Invalid model output remains HTTP 503 rather than a course refusal.

## Configuration and cost

Restart the server to activate the changed pipeline. No reindex is needed while embedding identity
and source content stay the same. The local `.env` was not rewritten; existing explicit overrides
still take precedence over the new defaults:

```dotenv
HR_RETRIEVAL_MIN_COSINE=0.24
HR_RETRIEVAL_MIN_BM25=1.0
HR_TRANSLATE_SEARCH_QUERIES=true
```

Answerable explained requests now use four chat calls: planning, selection, explanation, verification.
Verbatim requests use two. Hybrid adds batched search embeddings; full context does not. Whole-message
greetings, cheap policy rejections and unresolved bare follow-ups still avoid model inference.
New `question_plan`, `requirement_selection` and `requirement_verification` diagnostics are opt-in
through the existing authenticated evaluation response. They may contain question text.

Strict schema fields and reference enums follow the [official OpenAI structured-output contract](https://developers.openai.com/api/docs/guides/structured-outputs).
Custom gateway integrations must implement the new plan/complete-selection/complete-explanation/
complete-verification methods; legacy direct gateway methods remain available for diagnostics.

## Reproduce the offline score sweep

From the repository root, with the existing source database and immutable baseline snapshots:

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_retrieval.py --report evals/results/pilot-run-20261016-1630/captain-storm--hybrid--r1.json --dataset evals/results/pilot-run-20261016-1630/inputs/captain-storm.jsonl
```

This command prints JSON, makes zero provider calls and opens SQLite read-only. It checks the
dataset hash, source fingerprint and current passage IDs instead of silently reusing stale scores.
It replays the original query scores, not new translations or reranker inference. In that baseline,
source coverage is 10/12 answerable cases at `0.30` and `0.25`, and 12/12 at `0.24` and `0.20`.
Four of eight negative/control cases receive candidates at all four settings. Candidate exposure is
not a false answer, and this small development sweep does not establish safe or optimal thresholds.

Run new paid tests only into a fresh benchmark output directory. Review all generated statements,
both English false-refusal cases, the manager-fear/control multi-part case, and negative cases.
Keep an untouched held-out set for architecture decisions. Neither cached-score replay nor mocked
regression tests should be presented as a new response-quality score or instructor approval.
