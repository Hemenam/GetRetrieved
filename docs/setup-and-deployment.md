# Setup and deployment

> This guide covers local Ollama setup, diagnostic literal lookup, and deployment. For hosted model API keys and provider configuration, follow [the main README](../README.md). Run all commands from the project root.

A standalone FastAPI backend for an LMS chatbot with **natural cited answers by default**, exact source evidence available separately, and an optional verbatim source-text mode. Django can call it over an authenticated HTTP API, or use the built-in `/chat` testing page. Configure a local Ollama backend or a compatible hosted model API for natural-language questions.

The supplied Persian course was inspected as source data: 11 chapters, 242 nonempty paragraphs, and 44 coherent passages. The original customer file and the local database are excluded from Git and Docker images.

## What is enforced

- `response_mode="explained"` is the default. The LLM writes a natural answer; every generated statement cites included excerpt IDs, and a separate model call checks responsiveness and support before display. Exact source excerpts remain separate evidence.
- `response_mode="verbatim"` explicitly requests only stored excerpt text. The LLM selects passage IDs; the backend retrieves the exact text.
- Whole-message greetings, thanks and help requests receive short conversational replies without retrieval or model calls. Hybrid retrieval has configurable minimum search scores; see [relevance gating](relevance-gating.md).
- Each selected ID must be an authorized retrieved candidate from an active document revision. The backend checks the source span and checksum again before responding.
- Questions, previous questions, and imported document text are untrusted data. They cannot change the source policy.
- Tenant, course, user, and action permissions come from a signed, short-lived Django service token, not from browser-supplied identity fields.
- Unsupported questions produce a fixed Persian/English refusal. Unresolvable follow-ups produce a fixed clarification. A model outage is HTTP **503**, not a misleading content refusal.
- Course methods may be quoted and, in explained mode, explained in the student's language using the selected evidence. New advice, new examples, personalized plans and outside factual additions remain outside scope.

Exact quotation provenance is enforced in code. **Selecting the right passage, deciding whether to refuse, and grounding generated explanations still require evaluation.** An exact quote can be irrelevant or misleading when taken from a hypothetical example. The explanation verifier is another model call, not a guarantee of factual support. Do not interpret passing infrastructure tests as an instructor-approved quality result.

## Quick start on Windows

Python 3.11 or newer is required. Run commands from the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m hrlearnium init --backend ollama
```

`init` creates `.env` with a random signing secret and leaves an existing `.env` untouched. Never commit that file or put its secret in browser code.

### Normal question answering with local Ollama

Install [Ollama](https://ollama.com/download) on the model host, then pull the configured models:

```powershell
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:8b
```

This Ollama configuration keeps model requests on the local Ollama host. These models are starting points, not a measured recommendation for this course. Model size, available RAM/VRAM, context length, and latency need checking on the deployment host. Set `HR_SELECTOR_MODEL` to the evaluated model you choose. The Ollama quick start uses hybrid retrieval, so changing the embedding model or its digest requires reindexing. For `HR_RETRIEVAL_MODE=full_context`, the embedding model is not used for ingestion or queries.

If you previously used literal mode, change `HR_MODEL_BACKEND=ollama` in `.env`. For hybrid retrieval, reindex existing course documents. For full-context selection, existing passages can be used immediately without reindexing.

```powershell
.\.venv\Scripts\python.exe -m hrlearnium inspect 'F:\Downloads\4361832582818373377_8046908015302784 (1).docx'
.\.venv\Scripts\python.exe -m hrlearnium ingest 'F:\Downloads\4361832582818373377_8046908015302784 (1).docx' --tenant customer-demo --course captain-storm
.\.venv\Scripts\python.exe -m hrlearnium serve
```

Open `http://127.0.0.1:8000/docs` for the OpenAPI explorer. In Ollama mode, `/health/ready` checks model inventory in addition to database access. The service does not automatically pull models. With a hosted backend, authorized course context and learner questions are sent to the configured provider; use [the hosted setup instructions](../README.md) to configure that connection.

Hosted `/health/ready` reports configuration readiness with `provider_verified=false`. It does not make a live provider request or establish that the key, model, quota, or provider schema support works. Validate those with an authenticated test query after configuration.

### Offline literal lookup

To inspect and exercise the entire source-to-API path without installing a model, set `HR_MODEL_BACKEND=literal` in `.env`. On a fresh checkout, use `hrlearnium init --backend literal` instead. Then ingest and serve as above.

This diagnostic backend accepts only the explicit grammar `متن دقیق «عبارت موجود در متن»` (also `عبارت دقیق`, `نقل قول`, or `exact quote "..."`). It returns a matching original passage; ordinary natural-language questions are refused. It is not a production substitute for semantic question answering. `response_mode="explained"` requires an LLM backend and returns HTTP 503 under the literal backend.

Do not confuse the **literal backend** with the **verbatim response mode**: verbatim responses accept natural-language questions when Ollama or a hosted LLM is configured. The response mode controls the wording shown to the learner; the backend controls how questions are understood.

```powershell
$token = & .\.venv\Scripts\python.exe -m hrlearnium token --tenant customer-demo --course captain-storm
$headers = @{ Authorization = "Bearer $token" }
$body = @{ question = 'متن دقیق «سبد اول (۳۰ دقیقه آینده)»'; response_mode = 'verbatim' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/courses/captain-storm/query -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Tokens expire after 60 seconds by default. Generate a fresh token for later requests. Include a returned `conversation_id` only when continuing the same user's conversation in the same course.

## API contract

All `/v1` endpoints require `Authorization: Bearer <JWT>`. The course must be explicitly authorized by the token. A learner's query token cannot upload content.

| Method and path | Scope | Purpose |
| --- | --- | --- |
| `POST /v1/courses/{course_id}/query` | `query` | Answer or decline a question |
| `GET /v1/courses/{course_id}/excerpts/{id}` | `query` | Open an authorized citation |
| `POST /v1/courses/{course_id}/documents` | `content:write` | Import multipart DOCX field `file` |
| `GET /v1/courses/{course_id}/documents` | `content:read` | List active course documents |
| `PUT /v1/courses/{course_id}/documents/{id}` | `content:write` | Replace atomically with a new revision |
| `DELETE /v1/courses/{course_id}/documents/{id}` | `content:write` | Retire a document from retrieval and citation access |
| `GET /health/live` | None | Check API process |
| `GET /health/ready` | None | Check database and backend readiness; hosted readiness is configuration-only |

Natural-language query with exact quotation output:

```json
{"question": "مدل ۳۰-۳۰-۳۰ چیست؟", "response_mode": "verbatim"}
```

The same question with a natural cited answer and separate source evidence:

```json
{"question": "مدل ۳۰-۳۰-۳۰ چیست؟", "response_mode": "explained"}
```

Both course-answer requests require an LLM backend. Omitting `response_mode` defaults to `explained`. The mode can be selected independently on each request, including follow-ups. Short supported greetings do not require model inference.

Response fields:

```text
request_id, status, answer, excerpts[], conversation_id,
reason_code, retrieval_mode, response_mode, explanation, policy_version
```

`status` is `answered`, `refused`, `clarification`, or `conversation`. For an answered `verbatim` response, `answer` equals `"\n\n".join(excerpt.text for excerpt in excerpts)` exactly and `explanation` is `null`. For an answered `explained` response, `answer` contains only generated statements with numbered citations, separated by blank lines. `explanation.statements` contains each statement's `text` and `citation_ids`, which must resolve to included excerpt IDs. Exact source text remains in `excerpts`. The returned `response_mode` matches the request. Nonanswered statuses contain neither excerpts nor explanations.

Each excerpt has its immutable ID, original text, and citation including document ID/version/title, chapter number/title, section type, paragraph range, source offsets, and source checksum. Citation labels are metadata, not additions to the course quotation. Keep explanation text visually separate from quotations; see the complete rendering contract in [Django integration](django-integration.md).

The default refusal is:

> برای این پرسش، توضیح مرتبط و کافی در محتوای این دوره پیدا نکردم. لطفاً پرسشی مرتبط با محتوای دوره بپرسید.

HTTP errors remain separate: `401` invalid token, `403` unauthorized scope/course, `404` missing or inaccessible resource, `409` revision/content conflict, `413` oversized input, `415` unsupported file type, `422` invalid input/document, `429` rate limit, `503` unavailable/invalid evidence service. Never render a 503 as “the course does not contain the answer.”

See [Django integration](django-integration.md), the usable [Python connector](../examples/django_connector.py), and [OpenAPI](openapi.json).

## Retrieval and content lifecycle

All backends use SQLite with tenant/course filtering. Retrieval is configured separately from response wording:

- `HR_RETRIEVAL_MODE=hybrid` combines BM25 keyword search and embedding similarity using reciprocal-rank fusion. Ingestion creates embeddings; queries require current embeddings for all active course passages. Vector similarity is an exact scan over the authorized course only.
- `HR_RETRIEVAL_MODE=full_context` sends all authorized course passages to the ID selector, bounded by the configured context limit. It does not create or require embeddings, including during ingestion. The existing 44 passages can be used with an LLM in this mode without reindexing.
- The diagnostic literal backend uses lexical lookup regardless of the retrieval setting.

At this corpus size a separate vector database is not required. The default course limit is 2,000 passages; load-test before expanding that limit or migrate the repository layer to PostgreSQL/pgvector for larger deployments.

For a supported LLM request, verbatim mode uses one chat call to select evidence. Explained mode uses three chat calls: select evidence, generate cited statements, and verify those statements. Hybrid retrieval adds one query-embedding call. Requests refused by an earlier check may use fewer calls. Full-context selection skips embeddings but sends more source text to the selector; response wording still follows the requested `response_mode`.

DOCX extraction preserves the original text characters; normalization of Persian/Arabic letters, digits, diacritics, and spacing is used only for retrieval. Lists and wrong/correct examples stay with their context. Citations reference chapters and extracted paragraph offsets; **Word page numbers are not invented**. Automatic Word list numbering, headers/footers, or images may require source cleanup; ingestion returns explicit warnings. This importer does not perform OCR or import arbitrary PDFs, videos, URLs, or web content.

Uploading identical bytes with the same embedding identity is idempotent. A replacement becomes active only after parsing and, for hybrid retrieval, embedding succeeds. Old revisions remain available for existing citations while the document is active; retirement hides all its citations. Physical deletion/retention policy is an operator decision; retired source data remains in SQLite and backups until purged.

After switching previously unembedded content to hybrid retrieval or changing the configured embedding identity:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium reindex --tenant customer-demo --course captain-storm
.\.venv\Scripts\python.exe -m hrlearnium verify-source --tenant customer-demo --course captain-storm
```

Reindexing requires an LLM backend with `HR_RETRIEVAL_MODE=hybrid` and creates a new revision if the embedding identity changes. It is atomic per document, not across the whole course; hybrid course queries fail closed while mixed embedding identities exist. A failed reindex leaves the last stored revision intact. Switching to full-context selection does not require reindexing; its ID-selection stage can be followed by explanation generation and verification when requested.

## Verification and evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests examples scripts
$env:HRLEARNIUM_TEST_DOCX = 'F:\Downloads\4361832582818373377_8046908015302784 (1).docx'
.\.venv\Scripts\python.exe -m pytest tests/test_ingestion.py -q
.\.venv\Scripts\python.exe scripts/evaluate.py --mode retrieval --document $env:HRLEARNIUM_TEST_DOCX --output evals/results/retrieval.json
```

The [100-case Persian evaluation set](../evals/course_qa.jsonl) is **proposed and not instructor-approved**. It includes answerable questions, related-but-unsupported questions, new-advice requests, injections, ambiguous follow-ups, and hypothetical-example boundaries. Each expected answer has short exact source anchors.

After models are configured, source ingested/reindexed, and the API running:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --mode api --local-auth --tenant customer-demo --course captain-storm --output evals/results/api.json
.\.venv\Scripts\python.exe scripts/evaluate.py --mode api --response-mode explained --local-auth --tenant customer-demo --course captain-storm --output evals/results/api-explained.json
```

The runner explicitly defaults to verbatim for the existing quotation baseline; this differs from the API's new explained default. The second command evaluates the explained response contract. `--local-auth` is for the local operator and mints fresh short-lived tokens using `.env`; it never prints them. For remote deployments, supply a permitted token through `HR_EVAL_TOKEN` instead. On a fast model, use `--delay-seconds 3` to stay within the default query rate limit, or set an appropriate limit on a separate evaluation instance. The runner distinguishes HTTP/model failures, unsupported answers, unnecessary refusals, quote coverage, mode matching, and citation integrity. Generated explanations still require human review against their cited excerpts; passing structure and link checks does not establish semantic correctness. Retrieval-only scores and mocked model tests do not establish real-model answerability accuracy.

The instructor should review the cases and wrong/correct example boundaries, add held-out learner questions, and agree to release criteria before learner rollout. Repeat full evaluations after model, prompt, retrieval, or course changes. Keep exactness violations and cross-tenant disclosure at zero; measure and review unsupported-answer and unnecessary-refusal rates separately.

## Deployment

Runtime dependencies are locked in `requirements.lock`; test dependencies are in `requirements-dev.lock`. Ollama runs without an external model API key. The hosted backend uses the provider settings described in [the main README](../README.md).

```powershell
docker compose up -d --build
docker compose exec ollama ollama pull qwen3-embedding:0.6b
docker compose exec ollama ollama pull qwen3:8b
```

The provided Compose stack explicitly selects the Ollama backend. Hosted deployments should use the API service with provider configuration from [the main README](../README.md), adjusting the Compose backend override if reusing this stack. The Compose API binds to localhost and the Ollama service has no published port. It uses a separate persistent course database volume. Upload content to the running API, or copy the DOCX into the container and run the ingestion CLI there; an earlier host database is not automatically copied into the volume. Pin the tested Ollama image digest in your deployment after choosing the model runtime. The provided baseline does not assume an NVIDIA Docker runtime.

Use HTTPS at the reverse proxy, keep the service private to Django, turn off interactive docs with `HR_ENABLE_DOCS=false`, and store JWT and provider secrets in your deployment's secret manager. No permissive browser CORS policy is enabled. The SQL-backed limits survive process restarts; model concurrency is bounded per worker. Start with one API worker for a single model host and measure latency under expected load. Allow for the three sequential chat stages of explained answers when setting connector, worker, and reverse-proxy timeouts; hybrid queries also need an embedding call.

The database contains course source files, revisions, and up to three recent **answered** user questions per conversation. Conversations expire after one hour; expired entries are physically removed on subsequent successful conversation writes. Application logs contain request IDs, methods, statuses, and timings, not questions or course text. Back up the database consistently, including SQLite WAL state via SQLite's backup API rather than copying a live database file alone.

Implementation references: [FastAPI security](https://fastapi.tiangolo.com/tutorial/security/), [Ollama structured output](https://docs.ollama.com/capabilities/structured-outputs), [Ollama embeddings](https://docs.ollama.com/api/embed), and [hybrid retrieval](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview).
