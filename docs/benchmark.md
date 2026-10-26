# Course-chatbot benchmark: first milestone

The benchmark compares `full_context`, `hybrid`, and `hybrid_rerank` on the **same questions and document**. It also summarizes multiple document profiles without pretending their questions are equally difficult. It does not modify chunking, database architecture, prompts, models, or course material.

This milestone includes an offline plan, bounded API runner, per-run human-review files, paired structural comparisons, and JSON/CSV summaries. It does **not** include a dashboard, an LLM judge, or statistical confidence intervals. Synthetic infrastructure tests are not measurements of chatbot quality.

## 1. Inspect the pilot without calling a model

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py plan
```

The checked-in [manifest](../evals/pilot/manifest.json) selects 20 [owner-approved development cases](../evals/pilot/cases.jsonl), three modes, and three repetitions: **180 main turns + up to 18 setup turns = 198 query requests**. There are additional prerequisite checks and evidence GETs. One query can invoke several hosted calls (planning, embeddings, evidence selection, explanation, verification); 198 is **not** the number of billable LLM calls. The existing policy-v3 collection stays unchanged; policy-v4 improvements need a new run directory and fresh response review.

`run` without `--execute` also only prints a plan. Planning uses neither credentials, settings, the database, nor the network. The request cap is **not a dollar budget**; configure a provider-side spend limit before execution. Missing usage/pricing stays unknown rather than being reported as zero.

## 2. Review the gold dataset

Every pilot case starts as `review_status: "proposed"`. Execution rejects the entire selected cohort until **all** cases are approved; it never silently filters out difficult pending cases.

An instructor must check each case against the exact document currently imported:

1. Confirm the question and `expected_status`: `answered`, `refused`, `clarification`, or `conversation`.
2. Confirm each `required_facts[].text` describes a necessary meaning, and its exact `evidence_quotes` really support it. These are draft annotations, not certified ground truth. Correct `required_quotes` and chapter numbers too.
3. Confirm absence for negative cases; a related paragraph is not enough to answer an unsupported price, author identity, or prescription.
4. Check forbidden claims and `expected_language`. For English questions, judge the generated answer's language, not the Persian source excerpts.
5. Set `reviewer` to the instructor's identity and `review_status` to `approved` only after reviewing. Keep this pilot in `development`.

Cases share `question_family_id` when they are translations, paraphrases, or related follow-ups. A family cannot cross development/held-out splits within a dataset. Newly authored held-out questions must stay separate from tuning; the existing public development pilot is not a final test.

All document profiles in one experiment must select the same split. Run development and held-out experiments separately; their macro scores are never blended.

Draft case origins are recorded in `notes`. The pilot covers greetings, exact and paraphrased questions, numeric variants, English answers, multi-chapter evidence, list completeness, follow-ups, examples versus real policy, missing information, unrelated questions, injection, and ambiguity. It is not comprehensive. Section-only authorization is deferred; expectations follow the current whole-course contract.

## 3. Prepare the existing server

The runner queries already-imported courses. It does not upload, reindex, install model files, or download a reranker. Use the [hybrid testing guide](hybrid-testing.md) to prepare embeddings and the optional reranker. One up-to-date server can serve all three per-request modes; restarting between runs is unnecessary.

For the configured pilot, confirm tenant `customer-demo` and course `captain-storm`. Prefer isolated benchmark course copies when preparing new material; never replace live content as part of a benchmark. Querying creates normal conversation state, but each case begins a new conversation. Only that case's optional `previous_question` is used for its follow-up.

Authentication is either `HR_EVAL_TOKEN`, or explicit `--local-auth` for loopback only (same local JWT settings as the server). Tokens and signing/provider keys are not written into reports. Prerequisite GETs confirm availability of requested modes, **not** provider health or answer quality.

## 4. Execute only after review and spend approval

Example for the testing server on port 8002; adjust to your running server:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py run --execute --base-url http://127.0.0.1:8002 --local-auth --output evals/results/pilot-run-001
```

The output directory must be new. The runner:

- Snapshots the manifest and datasets, records hashes and local runtime/package versions, and checks mode prerequisites before any query.
- Shuffles document/mode order within each repetition with the manifest seed. Case order stays fixed for paired comparisons. The seed does **not** make hosted model output deterministic. No paid warm-up calls are hidden; first-use/cold-load latency is included.
- Sends `retrieval_mode` and `response_mode: "explained"` on both setup and main requests, requiring returned modes to match.
- Saves each completed report and an unfilled `.review.json`; checkpoints progress in `index.json`.
- Checks that source, model identities, prompt/code hashes, thresholds, and other fixed settings stay comparable. A repeated mode must also keep its reranker configuration. Drift stops further runs after preserving the report that exposed it.

The hard `max_query_turns` checks the entire planned main+setup cohort before execution. The runner performs no HTTP retries or automatic paid resume. Policy v5's hosted gateway can retry a transient transport failure once within a logical provider call; attempts, retry settings and recovered failures are recorded in diagnostics. This query-turn cap does not cap provider attempts or USD spend. A failed/interrupted experiment keeps completed reports; use a new directory for another experiment. An interrupted in-flight batch or failed provider attempt may incur charges not captured in usage data. Operational errors are never counted as successful refusals.

Optional `--pricing path/to/pricing.json` uses the existing explicit per-model USD-per-million-token schema. Prices are snapshotted too. This estimates reported query/setup provider usage, excluding ingestion/indexing, hosting, and local reranker computation. Unreported or unpriced usage makes the full estimate unknown.

## 5. Review answers and produce comparisons

Each completed mode/repetition has a `.review.json`. Read the original course and cited excerpts, fill boolean judgments and `reviewer`, then mark completed cases `reviewed: true`. Do not edit captured responses or gold annotations. Reviews are bound to exact report hashes; pending fields do not become passes.

For answers, judge relevance, completeness, example context, every required fact, support for **every claim** in every statement, citation support, and forbidden claims. A relevant source chunk alone does not establish that the chatbot answered the question. All statuses also require the language judgment when specified. A refusal to an answerable question fails status correctness; do not mark it correct merely because it is cautious.

In explained mode, a passage dump that never directly answers the student's question fails relevance even if it cites authentic, related chunks. A concise, supported direct answer is the target; do not require word-for-word agreement with a draft fact's wording.

These review files are **not blinded**: architecture can be inferred from filenames/responses. Where practical, have a second reviewer audit disagreements. A randomized blinded-review interface and an independently calibrated LLM judge are later milestones. Separating structural checks from semantic human review follows [OpenAI's evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices).

You can summarize before finishing reviews, but quality scores/ranks remain unavailable:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py summarize --run-dir evals/results/pilot-run-001 --output evals/results/pilot-summary-001
```

After completing reviews, use a **new** summary output directory. It contains:

- `summary.json`: denominators, per-document/mode scores, candidate evidence coverage, paired status disagreements, token usage, stage latency, comparability checks, and macro leaderboard.
- `leaderboard.csv`: compact per-document/mode quality, error, language, latency, and cost columns; blank quality/cost fields mean unknown. Full denominators are in JSON, with answerable/non-answer counts also in CSV.

For an offline, question-by-question comparison of captured answers in all three modes:

```powershell
.\.venv\Scripts\python.exe scripts/compare_responses.py --run-dir evals/results/YOUR-RUN --output evals/results/YOUR-SUMMARY/responses.md
```

This validates report hashes and matching question cohorts, shows actual responses (not regenerated
examples), and compares latency on common answered questions. It never calls a model or fills human reviews.

For a stopped experiment with all three captured cohorts, `--allow-stopped` permits an explicitly
marked **inspection-only** response view. It does not change the experiment status, erase failures
or enable an architecture ranking.

Primary quality is fully correct, complete, grounded, correctly cited, language-compliant answers divided by **all answerable main turns**, including errors and false refusals. It requires human judgments, not quote overlap. Setup failures cannot pass a follow-up. Review coverage is over evaluable responses; operational errors remain in quality denominators.

Observed ranks require a complete comparable experiment, approved gold, and complete human reviews. Eligibility conservatively requires zero observed false answers on non-answer cases, no operational errors, and all automated checks passed. Ties share a rank. This is descriptive ordering, **not a statistically proven winner**, and `winner` remains null. Zero errors in a 20-case pilot does not establish zero production risk.

## 6. Add another initial document

Create a separate case JSONL, review it against that document, import the document into an isolated course using existing ingestion, and add another document profile to a copied manifest. Dataset and optional DOCX `document` paths are relative to the manifest. Increase `max_query_turns` deliberately after checking the new plan. Document IDs and tenant/course pairs must be unique.

Optional `source_corpus_sha256` pins a previously verified corpus fingerprint. It is reported in evaluation diagnostics and is **not the DOCX file hash**. Without a pin, the first observed corpus becomes the within-profile baseline. Optional `document` enables local DOCX source-span verification through the existing evaluator. Without it, stored authorized excerpt checks still run, but local file provenance is not independently established.

Paired comparisons stay within one document/course. Cross-document quality is an equal-weight mean of per-document rates; it does not mix question-level cohorts or declare one document easier/better. Semantic question IDs can be retained for content variants, but evidence mappings and answerability must be reviewed again. Keep source difficulty and held-out/development status visible.

Before a final architecture decision, expand the dataset, add untouched held-out courses/questions, define deployment-specific safety thresholds, and add paired uncertainty estimates clustered by question family (not by treating three repeats as independent questions).

## Explicit recovery after an operational interruption

`scripts/recover_benchmark_errors.py` is a separate, opt-in collection tool for a stopped,
single-document, one-repetition experiment. Use it only after the cause is addressed and the user
authorizes continuing. It selects **operational errors only**: evaluated answers, false refusals and
semantic failures are never replayed to obtain a better result. Planning is offline, for example:

```powershell
.\.venv\Scripts\python.exe scripts/recover_benchmark_errors.py --source-run evals/results/ORIGINAL-RUN --manifest evals/chapter-coverage-v1/approved-20261019/manifest.json --max-query-turns 57
```

Execution additionally requires `--execute --output NEW-DIRECTORY` and normal authentication/server
arguments. The cap must explicitly cover the selected failed requests; it is not a USD limit.
The first retry is a real recovery request, not an additional hidden paid health probe. A new
operational error stops the tool immediately, including failures during evidence checking. There
are no automatic outer HTTP retries or automatic resume after another interruption.

The tool checks the original report hashes, approved dataset, pinned source and current fixed
pipeline configuration before collection. Every request is checkpointed individually, with its
original error and source-report hash. A pending in-flight request can still incur unreported usage
if execution is interrupted. The application gateway's separately recorded bounded transport retry
policy remains unchanged.

Inspect both phases offline using a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts/report_benchmark_recovery.py --source-run evals/results/ORIGINAL-RUN --recovery-run evals/results/RECOVERY-RUN --output evals/results/RECOVERED-VIEW
```

The resulting view labels each response **original** or **recovery**, and keeps original errors in
their original files. It does not overwrite reports, mark a stopped run complete, fill owner reviews,
infer a winner or turn two collection phases into a clean controlled experiment. Its outcome counts
are descriptive recovered-view counts, not first-attempt reliability or proof of generalization.

## Separately authorized semantic diagnostic repeats

`scripts/rerun_benchmark_case.py` is for a **new explicit request** to repeat one already-evaluated
case, not for filling a benchmark with better retries. Without `--execute` it is offline. Execution
requires a new output directory, verifies the approved question/source/fixed configuration, sends
one fresh-conversation query, and saves a separately labeled diagnostic response and checkpoint.
It never changes a previous answer or score, never resumes automatically, and never sends a second
outer query on failure. The existing bounded provider transport-retry policy remains visible.

`scripts/inspect_chapter_answers.py --chapter N` prints source passages and the three saved answers
for offline assistant review. With `--cost-output NEW.json` instead, it records the token-cost
comparison using its explicit standard list-price assumptions. Those estimates are not an account
bill: gateway pricing, cached-input usage, unreported failed attempts and local compute are excluded.
Neither inspection command calls a model or fills human-review files.
