# HRLearnium course assistant

A course-chatbot component that answers student questions from authorized course documents. The LMS owns login and enrollment permissions. Use the [token-free LMS component integration](docs/lms-component.md) for direct Python calls or LMS-authenticated HTTP; the independent FastAPI/JWT server and testing UI remain development/legacy adapters. With an LLM configured, students receive a **natural, cited answer by default**, with exact source excerpts available separately as evidence. An explicit source-text option returns quotations instead.

The project supports API keys through an OpenAI-compatible Chat Completions adapter, as well as local Ollama. **An API key alone does not activate a model:** the backend, provider URL, model name and key must be configured together. Use the evaluation runner to check your configured model; infrastructure tests alone do not establish answer quality.

## Run the local chatbot without JWT

With the current `.env` and imported course:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium dev-chat --tenant customer-demo --course captain-storm --port 8000
```

Open <http://127.0.0.1:8000/chat>. The page automatically connects to the
server-configured course; no token or browser API key is needed. This explicit
development adapter binds only to loopback, rejects foreign hosts/origins, and
grants query-only access to that one course. Do not deploy it as LMS authentication.
The LMS integration remains token-free and uses the main backend's trusted user
context. The old `serve` command is the retained standalone **JWT** adapter.

The supplied source is a Persian **DOCX**, with 11 chapters, 242 nonempty body paragraphs and 44 stored passages. The importer accepts DOCX only. PDF import, OCR and page-number citations are not implemented.

## The student's response option

Both options use the same LLM answerability check. `verbatim` describes the **output**, not the wording the student must use.

| `response_mode` | Successful response |
| --- | --- |
| `explained` (default) | A generated answer to the actual question, with citations; exact passages remain separately in `excerpts` |
| `verbatim` | Original supporting passages, copied exactly, with source locations |

Both options accept direct questions, paraphrases, colloquial Persian, and questions about relationships described in the course. A related topic is insufficient: the model must judge that the passages support the actual answer. Unsupported questions receive a fixed refusal; unresolved references can receive a clarification.

Whole-message greetings, thanks, and help requests receive a short `conversation` response without retrieval, citations, or model calls. A greeting followed by a real question still goes through the evidence pipeline.

Examples for `POST /v1/courses/captain-storm/query`:

```json
{
  "question": "توی بحران چطور بفهمم کدوم کار رو باید اول انجام بدم؟",
  "response_mode": "verbatim"
}
```

```json
{
  "question": "توی بحران چطور بفهمم کدوم کار رو باید اول انجام بدم؟",
  "response_mode": "explained"
}
```

Explained mode permits a clearer explanation of the source meaning, without dumping whole chunks into the answer. It does not permit extra facts, invented examples, personal opinions, internet research or new personalized advice. The model has no web-search tools in this application.

## How a question is answered

```mermaid
flowchart TD
    A[Question and response mode] --> B[Trusted LMS identity and course access]
    B --> S{Whole-message greeting or help?}
    S -->|Yes| T[Short conversational reply]
    S -->|No| C[Load authorized current passages]
    C --> P[Route topics using course outline and original text]
    P --> D[Full document context or hybrid retrieval]
    D --> E[LLM selects evidence for every requested part]
    E -->|Unsupported or ambiguous| F[Fixed refusal or clarification]
    E -->|Supported| G[Validate IDs and exact source spans]
    G -->|verbatim| H[Return exact excerpts and citations]
    G -->|explained| I[LLM drafts statements with excerpt IDs]
    I --> J[Validate citations and check support and part coverage]
    J -->|Supported| K[Return natural answer and separate source evidence]
    J -->|Rejected draft| R[One evidence-bound answer repair and recheck]
    R -->|Verified| K
    R -->|Still unverified| X[Operational error, not a course refusal]
```

1. **Authorize first.** The LMS supplies trusted tenant/user/course/action context; the standalone adapter can obtain that context from a service JWT. The core checks course/action access and filters content by tenant/course before retrieval or model calls. The request body cannot override identity.
2. **Resolve conversation context.** A conversation ID must belong to the same user, tenant and course. Up to three previous answered questions help resolve follow-ups; they are not factual evidence. Omit the ID for a first question. Invented or expired IDs return 404. Conversations have a one-hour sliding expiry; refused questions do not enter history.
3. **Apply narrow Python rules.** Prechecks reject some explicit requests to override the source restriction or invent personalized advice. These supplement the model; they are not a semantic classifier.
4. **Keep original request units and route their indices.** The server conservatively preserves independent questions, comparisons, quoted questions and dependent clauses. The shipped gateways classify numeric unit indices; they cannot retype or silently merge the original requests. Missing bare referents are clarified before speculative retrieval. Exact structural lookups, including first/last chapter titles, resolve against the authorized outline. English hybrid questions can get a Persian search hint. Full-context uses the bounded authorized corpus; hybrid uses relevance-admitted candidates and reranked mode retains its cross-encoder admission floor.
5. **Establish evidence before drafting.** The chat model extracts a concise fact brief directly answering each unit. Broad overviews can instead use literal, chapter-spread introduction text from eligible candidates, avoiding generated topic/source mismatches. A strict response schema bounds the shared source list and every fact's references to it. Actual names, numbers or medications require source-stated values, not role labels or related guidance. Source offsets and citation bookkeeping are not exposed as factual context. Python validates the selected sources, required values and completeness contract. This is semantic fact extraction, not a guarantee that every interpretation is correct.
6. **Verify the source in Python.** Every selected ID must be a retrieved candidate, unique, authorized and from a current active revision. The backend reloads stored text, verifies its canonical source slice and source checksum, then sorts excerpts by source position.
7. **Write from the brief and review.** In `verbatim`, `answer` is exactly the selected excerpt texts joined by two newlines. In `explained`, the writer produces up to eight natural statements from the fact brief. The reviewer selects server-owned support IDs and checks intent, actor/action direction, stance, causality and complete coverage. Each supporting span must belong to that statement's own citations. One rejected draft can trigger bounded evidence reselection, rewrite and fresh review, using only the original eligible pool. A still-unverified answer is an operational failure, not proof of source absence. Supported independent siblings are retained. Source versions are checked again before publication.

All chat stages use `HR_API_MODEL` for the hosted adapter or `HR_SELECTOR_MODEL` for Ollama. The optional embedding model only produces search vectors; it does not write answers or decide answerability.

For one successful ordinary content group, the new gateway path uses **two chat calls for verbatim** or **four for explained**. The source-literal overview brief avoids its briefing model call. Independent units share classification but execute their own evidence/writing/review stages. Hybrid adds batched embeddings. Focused answers retain `HR_MAX_EXCERPTS` (default 4); broad course summaries have `HR_OVERVIEW_MAX_EXCERPTS` (default 8). Repairs and larger contexts can increase cost; exact social/metadata prechecks can avoid calls. See [v10 contracts and limits](docs/request-contracts-v10.md). Old custom gateway adapters retain their compatibility paths until migrated.

Policy v5 adds strict answered/refused selection branches, supplies the per-part evidence map to
both writer and verifier, and permits **one recorded retry for transient transport failures** by
default (`HR_API_MAX_RETRIES=0` disables it). It does not retry semantic refusals, schema failures,
or HTTP errors. Extra attempts can incur charges even without usage data; see the
[reliability changes](docs/reliability-v5.md). Documents, embeddings and retrieval thresholds are unchanged.
These transport retries are separate from the bounded routing and answer-draft repairs.

## Literal and semantic evidence matching

Current execution details: [v10 request contracts and evidence briefs](docs/request-contracts-v10.md).

| Retrieval setting | What happens | Requirements |
| --- | --- | --- |
| `full_context` | All current authorized passages go to the LLM, which evaluates literal and semantic support directly | Chat model only; existing imported passages need no reindex |
| `hybrid` | BM25 keyword ranking and embedding cosine similarity are combined with reciprocal-rank fusion; the top candidates go to the LLM | Chat and embedding models; passages indexed with the configured embedding identity |
| `hybrid_rerank` | Hybrid retrieval builds a larger pool; a local cross-encoder ranks query/passage pairs before the existing LLM evidence selector | Hybrid prerequisites plus the optional reranker runtime and model files |

For this 44-passage document, full context is a useful starting configuration: the LLM can inspect all passages without relying on a shortlist. It is bounded by `HR_MAX_CONTEXT_CHARACTERS` (48,000 by default, counting passage text and chapter titles). An oversized course returns 503 and requires a suitable retrieval configuration. This is a character limit, not a provider token limit; the chosen model must accommodate prompts, JSON metadata, evidence and output.

Hybrid defaults to eight candidates and up to four selected excerpts. Keyword search normalizes Persian/Arabic letter variants, digits, diacritics and spacing, removes fixed stopwords and uses BM25. Vectors allow candidates without shared query words. Returned source text is never normalized. Search scores are rankings, not answerability confidence scores.

The provisional plain-hybrid floors are cosine similarity >= **0.24 OR BM25 >= 1.0**, with a positive score required. They are adjustable development heuristics, not calibrated probabilities. Reranked mode admits a broader bounded pool before scoring; full-context mode has no retriever score. Both still require evidence selection and, for explained answers, semantic verification. See [relevance gating and calibration](docs/relevance-gating.md).

Hybrid retrieval can miss evidence outside the shortlist. Full-context inference can also make semantic mistakes. Both need evaluation with the actual model and course.

## Configure your API key

The hosted adapter implements the [Chat Completions API format](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) with [structured JSON outputs](https://developers.openai.com/api/docs/guides/structured-outputs). Other providers must support this contract; native APIs with different authentication or payload formats need their own adapter. Model names and account access are provider-specific.

Edit `.env` locally. `HR_JWT_SECRET` is only for the retained standalone service-token adapter, not for embedded/LMS authentication. It is separate from the model provider's API key; [CoreSettings](docs/lms-component.md) needs no signing secret.

```dotenv
HR_MODEL_BACKEND=openai_compatible
HR_API_BASE_URL=https://YOUR_PROVIDER_HOST/v1
HR_API_KEY=YOUR_PRIVATE_KEY
HR_API_MODEL=YOUR_CHAT_MODEL_ID
HR_API_STRUCTURED_OUTPUT=json_schema
HR_API_MAX_COMPLETION_TOKENS=4096
HR_RETRIEVAL_MODE=full_context
HR_ALLOW_REMOTE_MODELS=true
```

Replace the placeholders. The base URL must include the provider's API prefix: the adapter appends `/chat/completions`. For OpenAI itself the base URL is `https://api.openai.com/v1`; a compatible provider may use a different prefix. Remote connections require HTTPS. The service does not guess a provider from a key, follow redirects or silently fall back to literal lookup.

`json_schema` requests strict structured output. If the provider supports only JSON-object mode, explicitly set `HR_API_STRUCTURED_OUTPUT=json_object`; application validation still applies. The adapter also sends `store: false` and `max_completion_tokens`, and expects a normal completed Chat Completions response. Incompatible endpoints or models return 503; JSON mode is not a general compatibility switch for unrelated APIs.

The key is loaded as a masked secret and used only in the server-to-provider authorization header. Do not put it in Swagger's Authorize box, browser code or chat. `.env` is excluded from source control. Hosted calls send course evidence, the question and relevant previous questions to the configured provider. Model payloads exclude the LMS signing secret and tenant/student identity fields. Review your provider's data-handling settings for the deployment.

On a fresh checkout, follow the [installation guide](docs/setup-and-deployment.md), using `hrlearnium init --backend openai_compatible`. This creates `.env` without overwriting an existing file.

For hybrid retrieval, additionally configure an [embedding API](https://developers.openai.com/api/reference/resources/embeddings/methods/create) on the same base URL:

```dotenv
HR_RETRIEVAL_MODE=hybrid
HR_API_EMBEDDING_MODEL=YOUR_EMBEDDING_MODEL_ID
HR_API_EMBEDDING_REVISION=1
```

Then reindex the imported course before restarting:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium reindex --tenant customer-demo --course captain-storm
```

Hosted embedding identity includes the provider URL, model name and explicit revision. Changing any of these requires reindexing. Provider aliases can change without their names changing: increment `HR_API_EMBEDDING_REVISION` first, then reindex to save fresh vectors as a new document revision. Without an identity change, identical source uploads are deduplicated. Prefer pinned model identifiers where available.

## Use the browser chat

A lightweight, same-origin chat UI is now served by the Python app. No frontend build or Node.js
installation is needed to use it. Start the existing server from this project folder:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium serve --port 8000
```

Open [the local chat](http://127.0.0.1:8000/chat). Click **Connect a course**, enter
`captain-storm`, and paste a short-lived **service JWT**, generated in another terminal:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium token --tenant customer-demo --course captain-storm --scope query --ttl 300
```

Do not paste your model provider API key. That stays in the server's `.env`.
Choose **Cited explanation** or **Source text**, then send a question. Click a citation to read
the exact source, or enable **Test details** to inspect the existing evaluation diagnostics.

Use **Retrieval mode** to compare **Full context**, **Hybrid**, and **Hybrid + reranker** per
request without changing the server default. **Mode setup and checks** explains missing prerequisites;
**Refresh status** checks again after operator setup. Missing setup blocks sending in that mode,
never silently falls back. Switching retrieval starts a fresh conversation but keeps earlier results
visible. This does not generate missing embeddings or download reranker files automatically.
Follow-ups automatically send the returned conversation ID. Refreshing a token for the same
user/course keeps the conversation; **New chat** resets it. Reloading the page clears its token
and transcript. Export saves the transcript and evidence as JSON, not the access token.

See [the chat UI guide](docs/chat-ui.md) for testing, security boundaries and file locations.
Set `HR_ENABLE_CHAT_UI=false` to disable the page in a deployment; API authentication is unchanged.

## Use localhost Swagger

Restart after changing `.env` or code:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium serve --port 8000
```

Open `http://127.0.0.1:8000/docs`. Hosted `/health/ready` reports missing configuration by setting name, never by secret value. It explicitly performs a **configuration-only check** (`provider_verified: false`), not a test of key validity, model access, quota or connectivity. A successful query is needed to exercise the provider.

Generate an LMS service token in another terminal:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium token --tenant customer-demo --course captain-storm --scope query --scope content:read --ttl 300
```

1. In **Authorize → Value**, paste the printed token without the word `Bearer`.
2. Expand **POST /v1/courses/{course_id}/query**, select **Try it out**, and enter `captain-storm`.
3. Replace the entire request body with either example above. Set `response_mode` to choose the output. Omit `conversation_id` for the first request.
4. Execute. Inspect `excerpts` for original text and locations, and `explanation` for generated statements in explained mode.

The source is already imported in this development database. For a fresh database, upload the DOCX using a `content:write` token or use the ingestion CLI. The example token expires in five minutes; generate another when you get 401.

## Response and citation fields

| Field | Meaning |
| --- | --- |
| `status` | `answered`, `partial`, `refused`, `clarification` or `conversation` |
| `answer_kind` | `content`, `metadata` or `multi`; selects the rendering/evidence contract |
| `metadata` | Imported document-outline provenance for a structural answer; otherwise `null` |
| `parts[]` | Independent questions with individual answers, statuses and evidence; otherwise empty |
| `response_mode` | Echoes the selected option, including on refusals |
| `answer` | Content answer, deterministic metadata answer, or numbered independent-question results; see `answer_kind` |
| `excerpts[]` | Exact original text, immutable passage IDs and source locations |
| `explanation` | Content statements in single explained answers; multi-answer explanations live inside `parts[]`; metadata has none |
| `statements[].text` | A generated explanation statement, at most 1,200 characters |
| `statements[].citation_ids` | One to eight unique included excerpt IDs supporting that statement |
| `citation.document_id`, `version`, `document_title` | Source document and stored revision |
| `citation.chapter_number`, `chapter_title`, `section_kind` | Extracted chapter and section metadata |
| `citation.paragraph_start`, `paragraph_end` | One-based inclusive range of nonempty extracted body paragraphs |
| `citation.source_start`, `source_end` | Zero-based character offsets in canonical source; end is exclusive |
| `citation.source_sha256` | SHA-256 of the complete extracted source, distinct from the original file hash |
| `reason_code` | Such as `supported`, `insufficient_evidence`, `ambiguous` or `outside_scope` |
| `retrieval_mode` | Actual `full_context`, `hybrid`, `hybrid_rerank` or diagnostic `lexical` path |
| `conversation_id`, `request_id` | Conversation continuity and request tracing |
| `policy_version` | `grounded-course-v4`, the application policy version |

In explained `answer`, the application adds numbered citations such as `[1]` to generated statements, separated by blank lines. Numbers refer to the one-based order of `excerpts`; whole quotations are not prepended. A frontend can render `explanation.statements` with clickable citations and open source evidence on demand. Render all text escaped and avoid displaying both answer representations redundantly. Version 3 changes the default mode and explained rendering contract; update existing consumers using the included connector.

Your earlier chapter 4 passage covers paragraphs 92–98 and contains the whole 30-30-30 explanation. Quotations are whole stored passages, not model-authored sentence fragments.

Exactness is checked against extracted DOCX body text, not Word page layout. Headers, footers, footnotes/endnotes, images and automatic list labels are excluded; table cells are read as body paragraphs. Chapter boundaries use heuristics and need review for new documents. Ingestion returns warnings. See [ingestion.py](src/hrlearnium/ingestion.py).

## Failure behavior and limits

Refusals use a fixed Persian or English message saying no supported course answer was found, or that the question is outside the course, and invite a related question. Clarifications ask the student to identify the course topic. A refusal is a system judgment, not proof that no answer exists anywhere in the document.

HTTP 401 means invalid/expired service authentication; 403 means missing permission; conversation-related 404 means an unavailable ID; 409 means content changed; 422 means invalid input; 429 means a request limit; and **503 means unavailable, misconfigured or invalid model/evidence processing**. Never display 503 as a content refusal.

In multipart responses, a failed independent answer can have `status=error` while supported
siblings remain visible. If nothing answered and a part errored, the overall status is `error`;
otherwise supported-plus-error is `partial`. Error parts contain no draft or source claims.

Python enforces quote exactness, authorized evidence IDs, source integrity and citation links. **LLM answerability and explanation-support judgments remain probabilistic.** The additional check reduces unsupported explanations but is not a mathematical guarantee, particularly because it uses the same model. Evaluate actual model behavior and review interpretation of hypothetical examples.

Changed documents activate a new revision after processing succeeds. Identical bytes with the same embedding identity are deduplicated. New queries use current revisions; old citations remain readable while their document is active. Retirement excludes the document and hides its citations. Historical content remains stored until separately purged.

## Diagnostic and local backends

`HR_MODEL_BACKEND=literal` is the earlier **no-LLM diagnostic backend**. It performs BM25 search followed by normalized substring lookup for requests such as `متن دقیق «سبد اول (۳۰ دقیقه آینده)»`. It refuses ordinary natural questions and cannot generate explanations (`explained` returns 503). This differs from `response_mode=verbatim` on an LLM backend.

`HR_MODEL_BACKEND=ollama` uses local models for the same selection, explanation and verification stages. See the [setup guide](docs/setup-and-deployment.md) for local installation, ingestion, deployment and operational limits.

## Verification and code map

Automated tests cover authentication, isolation, exact source spans, provider protocol handling, mode switching, explanation references, verification failures and source updates during generation. They use controlled model responses and do not establish live semantic accuracy. See the [verification record](docs/verification.md).

The [100-case Persian evaluation set](evals/course_qa.jsonl) includes paraphrases, direct questions, unsupported questions, injections and follow-ups. It is proposed and not instructor-approved. Run API evaluation in **both response modes** with the actual provider. Explained-mode structural checks cannot judge meaning: review statements against cited sources. See the [evaluation guide](evals/README.md).

Start the new [20-case benchmark pilot](docs/benchmark.md) with `.\.venv\Scripts\python.exe scripts/benchmark.py plan`. It compares all three retrieval modes with repeated runs, bound human reviews, and JSON/CSV summaries. The pilot is pending instructor review; planning makes no API calls and execution requires explicit approval flags and reviewed gold.

| File | Responsibility |
| --- | --- |
| [component.py](src/hrlearnium/component.py), [core_config.py](src/hrlearnium/core_config.py) | Token-free LMS entry point and provider/storage settings |
| [access.py](src/hrlearnium/access.py), [adapters](src/hrlearnium/adapters) | Core course/action boundaries and optional HTTP/JWT adapters |
| [models.py](src/hrlearnium/models.py) | Provider clients and structured calls |
| [policy.py](src/hrlearnium/policy.py), [service.py](src/hrlearnium/service.py) | Source rules, answerability, explanation checks and assembly |
| [ingestion.py](src/hrlearnium/ingestion.py), [storage.py](src/hrlearnium/storage.py) | Extraction, citations, revisions and scoped storage |
| [text.py](src/hrlearnium/text.py), [retrieval.py](src/hrlearnium/retrieval.py) | Normalization, keyword and semantic ranking |
| [schemas.py](src/hrlearnium/schemas.py), [main.py](src/hrlearnium/main.py) | Contract, Swagger and endpoints |
| [LMS component example](examples/lms_component.py) | Direct calls after host authentication/enrollment checks |
| [Legacy Django HTTP connector](examples/django_connector.py) | Retained standalone JWT integration and wire validation |

Further instructions: [setup and deployment](docs/setup-and-deployment.md), [Django integration](docs/django-integration.md), [OpenAPI](docs/openapi.json).

### Course metadata and multiple questions

Policy v10 currently has **16/18 strict passes in the latest targeted hybrid run**, not a full 188-question result. Manual review found an additional automatically passed answer with an unsupported example attribution. The 18 failures are **not all resolved**. See the [answer-by-answer review, remaining issues and development costs](evals/results/natural-student-v2-hybrid-v10-failures-20261024-08/manual-review.md) and [implementation contracts](docs/request-contracts-v10.md). Restart workers to load new code; the paid benchmark uses the embedded component, not a browser server.

Historically, policy v8 introduced [interpretation-only routing, complete metadata operations and bounded answer
repair](docs/routing-and-answer-repair-v8.md). It addresses the natural-question benchmark's
premature clarifications, generated extra requirements, brittle plans and false source-absence
messages. These historical notes do not describe the current v10 execution path.

The v7 hybrid baseline collected 188 questions plus ten setup turns: 125/188 passed strict
automatic diagnostics. This is not semantic accuracy; the references remain proposed. See the
[benchmark runner](docs/natural-benchmark-execution.md). The historical v7 typed-planning
implementation is documented [here](docs/request-planning-v7.md).

Policy v6 adds scoped document-outline lookup and independent-question execution with per-part
answers, citations and partial responses. Structural answers are labeled as imported document
metadata, not an official LMS syllabus. See [behavior, API fields, limits and testing](docs/metadata-and-multipart-v6.md).
