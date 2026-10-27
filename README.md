# HRLearnium course assistant

A Python component for answering student questions from LMS course documentation,
with cited evidence, course metadata lookup and multipart answers. The LMS owns
authentication and enrollment; the recommended integration requires **no second JWT**.
Current policy: `grounded-course-v12.2`. This is a development implementation;
reliable operation across arbitrary courses has not been established.

## Current capabilities

| Area | Implemented |
| --- | --- |
| Input | Text-bearing DOCX body paragraphs and table cells; exact extracted-text citations |
| Retrieval | `full_context`, `hybrid` (BM25 + embeddings + RRF), `hybrid_rerank` (optional local cross-encoder) |
| Answers | Generated cited explanation by default; optional `verbatim` source passages |
| Structure | Imported chapter titles/counts/sections; unavailable administrative fields are not invented |
| Conversation | Persian/English responses, greetings, independent question parts and limited follow-up context |
| Integration | Direct Python component, LMS-authenticated HTTP adapter, local testing UI |

Content questions go through **routing → retrieval → evidence brief → LLM drafting →
support/coverage review → cited answer**. Python validates source revisions, exact
spans and reference ownership; one bounded answer repair is available. Unsupported
questions are refused. Processing failures are reported separately, and supported
independent parts can survive a failed sibling. Semantic correctness still depends
on model judgments.

## Local setup

From the repository root, with Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m hrlearnium init --backend openai_compatible
```

`init` creates `.env` only if absent. Configure it locally with your provider's
actual endpoint, model name and key:

```dotenv
HR_MODEL_BACKEND=openai_compatible
HR_API_BASE_URL=https://YOUR_PROVIDER_HOST/v1
HR_API_KEY=YOUR_SERVER_SIDE_KEY
HR_API_MODEL=YOUR_CHAT_MODEL
HR_ALLOW_REMOTE_MODELS=true
HR_RETRIEVAL_MODE=full_context
```

The provider must support the configured Chat Completions/structured-output contract.
Keys stay server-side. Local Ollama is also supported; `literal` is a no-LLM
diagnostic backend, not a natural-question chatbot.

Supply your own `course.docx`; course files and databases are excluded from Git.
Inspect extraction and warnings before importing:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium inspect .\course.docx
.\.venv\Scripts\python.exe -m hrlearnium ingest .\course.docx --tenant demo --course demo-course
.\.venv\Scripts\python.exe -m hrlearnium dev-chat --tenant demo --course demo-course --port 8000
```

Open <http://127.0.0.1:8000/chat>. This loopback-only testing UI needs no token and
automatically connects to the configured course. Restart after code or `.env` changes.
The separate `serve` command is the retained standalone **JWT** adapter.

For hybrid, set `HR_API_EMBEDDING_MODEL` and `HR_RETRIEVAL_MODE=hybrid`, then run:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium reindex --tenant demo --course demo-course
```

Reindexing calls the embedding provider. Reranked mode additionally needs the
`[rerank]` package extra and model files; it does not silently fall back when unavailable.
Full context needs no embeddings but has a default 48,000-character corpus limit.
Retrieval scores and their thresholds are heuristics, not confidence probabilities.

## Integrate with the LMS

Create `create_assistant(CoreSettings())` once per worker. After the LMS verifies
the user and course access, construct a trusted `Principal` and call
`assistant.query(principal, course_id, QueryRequest(...))`; close the assistant
at worker shutdown. Calls are synchronous. Never obtain identity or permissions
from learner-supplied JSON.

See the [component guide](docs/lms-component.md) and
[service-layer example](examples/lms_component.py). For HTTP, use
`create_lms_app(principal_resolver, settings=...)` with the LMS's authentication.
The query endpoint is `POST /v1/courses/{course_id}/query`:

```json
{"question":"What does this course cover?","response_mode":"explained","retrieval_mode":"hybrid"}
```

For follow-ups, send the returned `conversation_id`. Current history contains the
last three answered questions, not a full transcript. The host LMS owns transport
limits, enrollment checks and which diagnostics/options learners may request.

## Limits and validation

- Import is DOCX only. PDF, HTML, OCR and multimedia ingestion are not implemented.
- Chapter/section detection uses Persian text heuristics. Automatic list labels,
  headers, footers and images are excluded; review extracted boundaries for each course.
- Access is tenant/course/action-level. Section-only or locked-lesson permissions
  require an explicit integration before using this component for that scope.
- SQLite and in-process search target small corpora. Generation, review scope,
  modifier routing and context budgets have known issues; see the
  [project review and priorities](docs/project-review.md).

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/chat_ui.test.mjs
.\.venv\Scripts\ruff.exe check src tests examples scripts
```

Tests use controlled provider responses; they do not establish answer quality.
Tests needing undistributed private benchmark packs skip explicitly when absent.
Live evaluation uses the configured paid API and requires a deliberate run;
see [benchmark execution](docs/natural-benchmark-execution.md).

## Further reading

- [Project review](docs/project-review.md): findings, proposed changes and acceptance checks.
- [Refinement history](docs/refinement-increments.md): v12 evidence, routing and reviewer contracts.
- [Setup/deployment](docs/setup-and-deployment.md) and [chat UI](docs/chat-ui.md).
- [API schema](docs/openapi.json); [core entry point](src/hrlearnium/component.py);
  [pipeline](src/hrlearnium/service.py); [model adapters](src/hrlearnium/models.py).
