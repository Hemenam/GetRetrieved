# Running the natural-student benchmark

The rich `natural-student-v2` pack uses a dedicated runner, not the legacy strict
JSONL loader. It has 188 main cases plus ten real setup turns: **198 chatbot query
turns for hybrid**, once each. The twelve location-assisted diagnostic pairs are
excluded and must not be pooled with the main benchmark.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/run_natural_benchmark.py
```

This validates the pack and current stored source and prints the workload without
calling the model. After the owner authorizes a paid run:

```powershell
.\.venv\Scripts\python.exe scripts/run_natural_benchmark.py --execute --mode hybrid --output evals/results/my-new-hybrid-run --max-query-turns 198
```

Use a new output directory each time. The 198-turn cap is **not a dollar cap**:
one chatbot request can invoke several billable model calls, including bounded
internal planning/language repairs and configured transport retries. There are
no hidden warmups, outer question retries, reranker calls or separate paid judges.

## Integration and isolation

The runner calls the actual embedded `CourseService` via `create_assistant`, using
trusted benchmark tenant/course identity. It does not need JWTs or an HTTP server.
This evaluates the LMS component, not HTTP authentication, server rate limits,
browser behavior or production LMS identity/enrollment integration.

It backs up the configured SQLite database to the run directory before collection.
All conversation writes occur in that copy; the original LMS database is not
reindexed or changed by query collection. The copy includes private course material
and existing database contents: keep the entire run directory private.

Only the student's question, mode, evaluation flag and actual returned conversation
ID enter a request. Expected answers, chapter locations and reference anchors are
never sent to the chatbot. Each independent case starts a fresh conversation.
Setup turns are judged against their own reference, including the greeting setup.

## Captures and interpretation

- `index.json`: authorization scope, immutable input/configuration fingerprints,
  attempted turns, completed captures, pending request and completion state.
- `inputs/`: exact benchmark/reference snapshots. Proposed references stay proposed;
  permission to collect responses does not invent reference approval.
- `natural-NNN--scored.json` and `--setup.json`: raw responses, diagnostics and
  sanitized error codes, saved before grading assertions.
- `results.json`: accumulated outcomes, per-track checks, latency, token usage,
  operational errors and cases requiring review.
- `answers.md`: every collected question and answer, with provisional automatic flags.
- `benchmark.sqlite3`: private isolated database copy.

Automatic checks cover status, routing, part counts, reference evidence anchors,
source revision, metadata values, output script, rendering and citation links.
They **do not prove factual correctness, completeness, appropriate grouping or
directness**. Missing an exact evidence anchor can also be a grading diagnostic,
not necessarily a wrong answer. Follow `evals/natural-student-v2/grading.md` for
meaning-based review. Never call the automatic pass rate semantic accuracy.

Reported token usage includes captured setup calls. No prices are assumed by this
runner. Failed requests can have unknown usage and nonzero charges, so token totals
and later public-list-price estimates are not provider invoices.

The runner stops immediately for common provider configuration/authentication,
billing or rate-limit HTTP errors, or after three consecutive operational failures.
Isolated question-level protocol failures remain recorded and collection proceeds.
It never retries a whole question or resumes a pending paid request automatically.

After collection is complete, generate the hash-checked offline summary:

```powershell
.\.venv\Scripts\python.exe scripts/report_natural_benchmark.py evals/results/my-new-hybrid-run
```

An optional `--pricing-reference PATH` accepts a separately dated model-rate record.
This produces a public list-price equivalent for observed calls, not a verified
provider invoice. The reporter refuses running checkpoints and changed captures,
and creates `summary.md`/`summary.json` without replacing prior summaries.
