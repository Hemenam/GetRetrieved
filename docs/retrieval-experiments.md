# Evaluation and retrieval experiments

Three experiments share the existing answering prompts and source passages:

| Mode | Candidate preparation | Default selector input |
| --- | --- | --- |
| `full_context` | Entire authorized course within the existing context limit | All passages |
| `hybrid` | BM25 and embedding similarity combined using RRF | Top 8 |
| `hybrid_rerank` | Hybrid top 32, then cross-encoder query/passage scoring | Top 8 |

The LLM evidence selector still decides whether the passages completely support the question. Reranker scores measure relevance, not confidence that the answer is supported. Section enforcement, chunking and explanation verification retain their existing behavior.

## Prepare instructor annotations

Run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/review_evaluation.py prepare-gold --dataset evals/course_qa.jsonl --output evals/course_qa.gold.jsonl
```

This copies the proposed cases into editable JSONL. Each positive case gets draft `required_facts` tied to source quote anchors, with blank meaning descriptions. Have the instructor fill each fact's `text`, confirm `evidence_quotes`, add `forbidden_claims`, check the expected status, and set `review_status="approved"` and `reviewer` to their name. Nothing is automatically approved.

Example additional fields (merge into a complete case, retaining question, status, chapters, quotes and notes):

```json
{
  "required_facts": [
    {
      "id": "time-horizons",
      "text": "The three horizons are 30 minutes, 30 hours, and 30 days.",
      "evidence_quotes": ["EXACT SUPPORTING SOURCE TEXT"]
    }
  ],
  "forbidden_claims": ["The framework guarantees that every crisis is solved in 30 days."],
  "review_status": "approved",
  "reviewer": "Instructor name",
  "split": "development",
  "scope_notes": "Explain the framework as taught in this course."
}
```

Approved answerable cases require nonblank fact descriptions and a reviewer. Define facts as meanings the answer must communicate; quote-anchor matches alone cannot measure those meanings.

Use `development` for tuning questions. Add independently written questions with `split="held_out"` for a final test. Include Persian paraphrases/number variants, incomplete lists, hypothetical examples, false premises, missing explanations, contradictory statements and distracting related topics. Current expected statuses should follow the existing whole-course contract; section enforcement is deferred.

The second course is not required now. When it arrives, create a separate JSONL dataset with its own questions, source anchors and instructor annotations. Run the same evaluator with that dataset and course ID. Keep held-out questions from it too. No placeholder material is scored as a real second course.

## Capture the full-context baseline

Configure your available embedding model before capturing the baseline, even though full-context mode will not use it, so the experiment settings remain consistent. Start the server with `HR_RETRIEVAL_MODE=full_context`; in another terminal run:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --mode api --course captain-storm --tenant customer-demo --local-auth --response-mode explained --expect-retrieval-mode full_context --dataset evals/course_qa.gold.jsonl --approved-only --output evals/results/full_context.json
```

While annotations are pending, use `evals/course_qa.jsonl` and omit `--approved-only` for development runs. Such reports remain proposed. `--limit 5` is useful for a smoke run; compare identical limits across experiments. Add `--split held_out` to run independent questions later.

The API course must already be imported. `--local-auth` uses the private local settings to mint short-lived query tokens for a loopback server; remote servers use `HR_EVAL_TOKEN`. Each case starts a fresh conversation, including any setup turn for follow-ups. Provider/rate-limit errors count as operational errors. Live runs consume provider tokens; these tools do not run them automatically.

Reports capture actual answers, exact excerpts and explanations for review. The evaluator requests `include_evaluation=true` to capture candidate IDs/scores, stage latency, prompt/code fingerprints, source-corpus fingerprint, model IDs and provider token usage when supplied. Candidate evidence coverage is checked through independently resolved excerpts. Signing secrets and API keys are excluded. Generated reports contain course material and are Git-ignored.

`--expect-retrieval-mode` detects a server that was not restarted after changing settings. The evaluator does not change the running server configuration.

## Run hybrid retrieval

Keep the same chat model, prompts, candidate limit and context limit. Configure the available embedding model and switch to `HR_RETRIEVAL_MODE=hybrid`. Existing full-context imports need embeddings:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium reindex --tenant customer-demo --course captain-storm
```

Restart the server. Repeat the same evaluation with `--expect-retrieval-mode hybrid` and `--output evals/results/hybrid.json`. Reindexing may call the embedding provider. All three runs should record the same embedding model setting, even though full context does not use embeddings.

## Run hybrid with a cross-encoder

Install the optional runtime in the experiment environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[rerank]"
```

Configure and restart the server:

```dotenv
HR_RETRIEVAL_MODE=hybrid_rerank
HR_CANDIDATE_LIMIT=8
HR_RERANK_POOL_LIMIT=32
HR_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
HR_RERANKER_DEVICE=cpu
HR_RERANKER_LOCAL_FILES_ONLY=true
HR_RERANKER_BATCH_SIZE=4
HR_RERANKER_MAX_TOKENS=8192
```

The default is a multilingual candidate to evaluate, not a demonstrated winner on this Persian course. Use pre-cached files or a local model folder. To allow an initial model download explicitly, set `HR_RERANKER_LOCAL_FILES_ONLY=false`; loading occurs on the first reranked query. Pin `HR_RERANKER_REVISION` to a model commit for reproducibility. CPU latency depends on the machine; loading/prediction are serialized per process.

The adapter uses [Sentence Transformers CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder/model.html); see the candidate's [model card](https://huggingface.co/BAAI/bge-reranker-v2-m3). Hub-provided custom model code is disabled. Overlong query/passage pairs return an infrastructure error rather than silently truncating evidence. Set the token limit within the chosen model's supported range.

Hybrid and hybrid-rerank share embeddings. Switching between them needs no reindex if source and embedding identity are unchanged. Run the same evaluation with `--expect-retrieval-mode hybrid_rerank` and `--output evals/results/hybrid_rerank.json`.

Readiness reports missing embedding settings and optional reranker dependencies. Reranker readiness checks configuration, not model download or inference. A successful query exercises inference.

## Review actual meanings

Prepare one review per report:

```powershell
.\.venv\Scripts\python.exe scripts/review_evaluation.py prepare-review --report evals/results/full_context.json --output evals/results/full_context.review.json
```

The instructor reads the original course and fills: status correctness, relevance, completeness, preservation of example context, coverage of each required fact, support for every statement and its citations, and presence of forbidden claims. A statement is supported only if every claim in it is supported. Set `reviewed=true` for completed cases and identify the reviewer. Non-answers require a status judgment. Pending cases remain ungraded. Do not edit captured questions, responses, source excerpts or gold annotations.

```powershell
.\.venv\Scripts\python.exe scripts/review_evaluation.py score --report evals/results/full_context.json --review evals/results/full_context.review.json --output evals/results/full_context.semantic.json
```

Repeat for the other modes. Human scores report review coverage, fact coverage, statement support, citation support and semantic passes. Reviews are bound to the exact report hash and cannot grade a different run. A successful scoring command means the review file was valid, not that every answer passed.

## Compare the same questions

```powershell
.\.venv\Scripts\python.exe scripts/compare_evaluations.py evals/results/full_context.json evals/results/hybrid.json evals/results/hybrid_rerank.json --output evals/results/comparison.json
```

The comparison requires the same dataset, ordered cases, course, response mode, source passages, chat/embedding settings, prompts, code and limits. It reports paired status differences, evidence coverage, operational failures, latency and available token usage. Compare the separate human scores alongside it; structural checks cannot establish a semantic winner.

To estimate provider cost, add `--pricing path/to/pricing.json` to each evaluation. Map your actual model IDs to numeric `input_per_million` and `output_per_million` prices in USD. If a reported call lacks usage or pricing, the total estimate remains null; available priced calls contribute to a subtotal. Estimates cover reported query/setup calls, excluding indexing, hosting and local compute. Unknown usage is not zero. Query timing excludes citation/candidate lookups; end-to-end evaluation timing includes them.

Repeat runs to measure variability. Choose using instructor-reviewed support and completeness first, then cost and latency. Model IDs can be aliases; the reports do not infer immutable provider snapshots.
