# Browser chat for pipeline testing

This is a development testing page, not the LMS student interface. Use `dev-chat`
for token-free loopback testing; `serve` retains the older JWT connection workflow.
[Token-free LMS integration](lms-component.md) uses the host's authentication and
does not mount this development screen.

The UI is deliberately small: static HTML, CSS and a JavaScript module served by FastAPI.
It talks to the same `/v1/courses/{course_id}/query` endpoint as the CLI and Django connector.
There is no separate frontend server, framework build, database, model or retrieval pipeline.
Document import, model choice, storage and chunking are unchanged. The shared backend now has
[relevance gates and natural answers](relevance-gating.md); the UI does not implement separate RAG logic.

Policy v6 also displays [course metadata and independent-question results](metadata-and-multipart-v6.md).
The current backend uses [v7 typed planning and language preservation](request-planning-v7.md)
with the same UI response contract; no separate browser-side classifier is used.
Partial responses show an individual status for every question. Metadata source details are distinct
from passage citations. Restart the backend and refresh this page to load the new response contract.

## Start and connect

For local testing with the existing `.env` and imported course:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium dev-chat --tenant customer-demo --course captain-storm --port 8000
```

Open <http://127.0.0.1:8000/chat>. It auto-connects, hides the JWT dialog, and sends
no Authorization header. Tenant, user, course and query-only permission are fixed
by the server, not accepted from browser claims. Retrieval controls, conversations,
citations and exports are unchanged. No signing secret is required.

This launcher cannot bind a public interface and does not trust forwarded client
headers. The adapter also checks loopback peers, Host, Origin and cross-site fetch
metadata, and denies document writes/other courses. Any local operator can access
the selected course: this is **not production authentication**. Model questions
still use the configured paid provider and require working billing.

### Legacy standalone server (JWT)

From the project folder in PowerShell, with the existing `.env` and imported course:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium serve --port 8000
```

Open <http://127.0.0.1:8000/chat>. In a second terminal:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium token --tenant customer-demo --course captain-storm --scope query --ttl 300
```

In **Connect a course**, enter `captain-storm` and paste the printed token. Other tenants/courses
need corresponding CLI arguments and a course with imported content. The frontend does not import
documents or reindex anything. Restart the server after changes to Python settings.

The model provider API key and JWT signing secret stay on the Python server. The browser receives
only the service JWT you paste. This is a developer testing UI, not an LMS login screen. For student
deployment, use the existing LMS authentication/integration to supply authorized short-lived access;
do not expose the token-generation command or signing secret to students.

## What to test

For the embedding/reindex and pinned local-model setup, see [real hybrid testing](hybrid-testing.md).

- Choose **Retrieval mode** independently from the answer format: **Full context**, **Hybrid**,
  **Hybrid + reranker**, or **Server default**. The choice is sent as optional `retrieval_mode`
  on each query; it never changes `.env` or another user's requests. Every reply displays the
  actual mode used, and exports include the selected mode. A mismatched explicit mode is an error,
  not an accepted fallback.
- Connecting or choosing **Refresh status** calls the authenticated, course-scoped
  `GET /v1/courses/{course_id}/retrieval-options`. It reports missing embedding configuration,
  missing/stale course vectors, context limits and reranker dependencies/local model files.
  These are prerequisite checks, not proof of provider access or inference quality. No embedding
  generation, reindexing, LLM inference or model downloads happen during this check. Ollama may
  inspect its model inventory. Unavailable modes can be selected to inspect their requirements,
  but sending is blocked until setup is complete and status is refreshed.
- Switching retrieval mode starts a fresh backend conversation for a clean comparison, while
  keeping earlier results visible. Repeat the same self-contained question in each mode; follow-up
  references will not carry across a switch. **Cited explanation** remains the default answer format.
- Hybrid requires an operator-configured embedding model and a course imported/reindexed in hybrid
  mode. Reranking additionally requires the optional runtime and the configured model's local
  files (unless the operator explicitly enables downloads). The UI does not install dependencies,
  download models, or modify/reindex your course. After setup changes, restart the server if needed
  and use **Refresh status**.
- Send `سلام` or `Hi`. Expect a short greeting, no course excerpts and no Answered badge.
  Greetings do not replace the recent answered questions used to resolve follow-ups.
- Ask a supported question and open its citations. Explained mode displays the structured
  explanation without repeating the long source text in every answer; the source drawer contains
  the exact excerpts. Source-text mode shows the exact excerpts directly in chat.
- Ask a follow-up. The UI reuses the conversation ID, but the backend still remembers only the last
  three answered questions, not previous answers or the entire visible transcript.
- Ask an unsupported or ambiguous question. Refusals and clarification are labelled separately
  from HTTP/provider errors. The UI does not make new answerability judgments.
- Enable **Test details** before sending to include the existing server evaluation metadata.
  Request IDs, reason codes, retrieval mode, browser duration and conversation ID are always
  available under each response. Browser duration includes network time, not just inference.
  In hybrid mode, inspect raw scores, the configured floors and before/after filter counts.
  These scores are not calibrated answer-confidence probabilities.
- Use **Export chat** to save questions, responses, sources and available diagnostics as JSON.
  Exports can contain private course content and student questions; handle them accordingly.

Token expiry is independent of conversation expiry. Generate a fresh token and update the connection
to continue as the same user and course. A different user, tenant or course clears local conversation
context. **New chat** starts without a conversation ID; it does not delete stored server conversations.
The browser transcript/token are not persisted across reloads. Backend conversations retain their
existing expiry behavior.

**Stop waiting** aborts the browser fetch, not necessarily inference already running on the server.
The browser restores the failed question and does not automatically retry. You can also type a next
draft while waiting; a failed request will not overwrite it. A normal supported full-context request
still uses one chat call in source-text mode or three in explained mode, so normal provider charges
apply. Readiness is not proof that a provider key or model works.

## Security boundaries

- The public `/chat` shell contains no course evidence, provider key, or signing secret.
  The query API still checks signatures, scopes, identity, tenant and course access.
- The pasted JWT stays in JavaScript memory. It is not placed in a URL, cookies, browser storage,
  logs or exported session metadata. It is briefly in the password input while being pasted;
  closing the settings clears that input. Reload/disconnect clears the active token.
- Client-decoded JWT claims are only hints for expiry and local context reset, not authentication.
  Python remains the authority.
- Model/source strings are rendered as text, never executable HTML or Markdown. Source citations
  open already-returned excerpts and do not send another model query.
- A restrictive same-origin content-security policy blocks external scripts, styles, connections
  and framing. It does not replace HTTPS, a trustworthy browser, or the existing deployment controls.
- `HR_ENABLE_CHAT_UI=false` disables the UI and its assets independently of Swagger/docs.

The UI intentionally does not add a public token-generation endpoint, loosen CORS, change access
rules, implement section scoping, or alter conversation memory.

## Implementation and checks

- `src/hrlearnium/web/index.html`: page, connection dialog and source drawer.
- `src/hrlearnium/web/assets/styles.css`: responsive layout, Persian-friendly typography and focus states.
- `src/hrlearnium/web/assets/app.mjs`: API client, transient session, structured answers and citations.
- `src/hrlearnium/main.py`: static routing and UI security headers.
- `tests/test_chat_ui.py`: serving, configuration switch and authentication boundaries.
- `tests/chat_ui.test.mjs`: request/response, token refresh, context, cancellation and error contracts.

Run the Python suite as usual. Node.js is needed only for JavaScript development tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/chat_ui.test.mjs
```

For an optional synthetic browser smoke test, run:

```powershell
.\.venv\Scripts\python.exe tests/chat_ui_preview.py
```

This **test-only** server binds to `127.0.0.1:8001`, uses a temporary database and deterministic fake
model, and prints a test token for `course-1`. Open `/chat` on port 8001 and use that course/token.
Synthetic embeddings and a fake reranker let you exercise all three retrieval choices without
model calls. The same test token also permits `course-unindexed`, which has no embeddings and
exercises the missing-setup state. Neither course exists in the real course database.
`unsupported` gives a refusal and `service-error` gives an operational error. It never loads `.env`,
calls a provider, or modifies the real course database. Stop it with Ctrl+C when finished.
These checks validate integration, not the real model's semantic quality.
