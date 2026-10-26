# LMS component integration (2026-10-22)

The LMS owns authentication, enrollment/tenant authorization and the student UI. The
chatbot core does not require a JWT secret, parse learner credentials or provide a login.
Its trusted `Principal` is an internal identity/permission value, not a token or request body.
JWT remains an optional application adapter for the existing standalone testing workflow.

## Recommended: call the Python component from the LMS backend

```python
from hrlearnium.component import create_assistant
from hrlearnium.core_config import CoreSettings
from hrlearnium.schemas import Principal, QueryRequest

# Once per worker; model settings/API keys remain server-side.
assistant = create_assistant(CoreSettings())

# Per request, AFTER the LMS authenticates the user and authorizes course access:
principal = Principal(
    sub=str(authenticated_user.id),
    tenant_id=str(authorized_course.tenant.assistant_tenant_id),
    course_ids=[str(authorized_course.assistant_course_id)],
    scopes=["query"],
)
result = assistant.query(
    principal,
    str(authorized_course.assistant_course_id),
    QueryRequest(question=student_question, conversation_id=student_conversation_id),
)
# Serialize result.model_dump(mode="json") through the existing LMS endpoint.
# At worker shutdown, after requests finish:
# assistant.close()
```

`authenticated_user`, `authorized_course`, `student_question` and
`student_conversation_id` are integration placeholders, not supplied LMS models. The
real backend must implement the enrollment lookup; it has not been provided in this repo.
[A framework-neutral service-layer example](../examples/lms_component.py) allowlists
learner fields and deliberately does not accept browser identity/permission claims.

Use `CoreSettings(_env_file=None, ...)` for explicit host-owned configuration. Otherwise
CoreSettings reads model/storage settings from the existing `.env`, ignoring standalone
JWT fields. It accepts no JWT settings of its own and validates no signing secret.
The core component imports neither JWT nor FastAPI; the installed package still includes
the retained standalone adapter dependencies for backwards-compatible development.

`create_assistant()` returns the same `CourseService` that the HTTP adapter uses. It owns
its created or supplied model gateway: close it once at worker shutdown, not after each
query and not while work is in flight. SQLite opens/closes its own connections per operation.
Methods are synchronous; an async LMS must run blocking work in its normal worker/thread
mechanism. Deploy multiple workers only with an appropriate model capacity/rate plan.

## Operations and retained boundaries

| Operation | Required context scope |
| --- | --- |
| `query`, `retrieval_options`, `course_outline`, `get_excerpt` | `query` |
| `list_documents` | `content:read` |
| `ingest` (new document or replacement), `retire_document` | `content:write` |

Every public operation checks the requested course against `principal.course_ids` and
checks its action scope before storage/model work. Queries, metadata, excerpts and
documents continue to use the principal's tenant. Conversation IDs remain bound to
tenant, course and user; knowing another conversation UUID grants no access.
Retirement still hides a document from retrieval/citation access rather than deleting it.

`Principal.sub` is simply the stable LMS user ID retained under its existing field name;
it does not imply JWT authentication. Construct a fresh Principal from current LMS checks
for each request. Do not cache a broad administrator context for all learners. Never map
unverified browser JSON/headers directly into `Principal`.

Access granularity is currently **tenant/course/action**, not per-document or per-section.
Do not pass document/section-only permissions and assume they are enforced. Such fields
are not supported by Principal. A course-wide outline/retrieval request must only be used
when the learner can access the whole imported course scope. Finer LMS permissions need
an explicit future scoped-document integration before that deployment can be considered safe.

For direct embedding, the LMS owns HTTP authentication/session expiry, CSRF, enrollment
revocation, request/upload size limits, rate limits and public error rendering. The core
retains bounded model concurrency and source/provenance checks, but is not a replacement
for host transport controls. Keep model keys and evaluation diagnostics off learner clients.

## Alternative: retain the HTTP API with LMS-owned authentication

```python
from fastapi import HTTPException, Request
from hrlearnium.core_config import CoreSettings
from hrlearnium.main import create_lms_app
from hrlearnium.schemas import Principal

def require_lms_principal(request: Request) -> Principal:
    # ONLY authenticated LMS middleware may set this server-side state.
    # That middleware must check current tenant membership/enrollment and action scopes.
    principal = getattr(request.state, "lms_principal", None)
    if not isinstance(principal, Principal):
        raise HTTPException(status_code=401, detail="LMS session required")
    return principal

app = create_lms_app(require_lms_principal, settings=CoreSettings())
```

The resolver is an ordinary FastAPI dependency and can be synchronous or asynchronous.
The host must actually mount/wire authenticated middleware; the example alone authenticates
nobody. A missing/invalid context fails closed. There is no fallback to a service token,
anonymous identity, trusted-looking headers or request-body claims. Scope/course denial
is 403, and an inaccessible conversation is 404. The testing `/chat` page/assets are not
mounted in this factory, even if `enable_chat_ui` is present in supplied settings.

The existing API body and response contract are unchanged. A protected shared-process
LMS integration can use this adapter without a second learner token. A separately hosted
Python service still needs a real authenticated server-to-server boundary supplied by the
LMS/deployment (for example verified middleware behind a protected proxy); do not expose
an unsigned identity-header endpoint. This change does not configure that external boundary.
Keep the host's cookie/session CSRF protections enabled; removing JWT does not replace CSRF.

## Standalone/development compatibility

For token-free local UI testing, use:

```powershell
.\.venv\Scripts\python.exe -m hrlearnium dev-chat --tenant customer-demo --course captain-storm
```

It uses CoreSettings (no JWT dependency/secret), fixes a query-only Principal on
the server, and mounts the shared testing UI only on loopback. Foreign host/origin
and remote-peer requests are rejected; other courses and document writes remain
forbidden. This developer-only context is not a replacement for LMS login and
must not be used for production. The real LMS factory remains unchanged and does
not expose this UI or a fallback anonymous identity.

`python -m hrlearnium serve`, `hrlearnium.main:create_app`, token generation and the current
manual-token testing UI continue using the standalone JWT adapter. Its configuration is
`hrlearnium.adapters.standalone.StandaloneSettings`; `hrlearnium.config.Settings` is a
compatibility alias. JWT decoding lives in `hrlearnium.adapters.jwt`; `hrlearnium.auth`
is an old-import compatibility wrapper, not a core dependency. Existing service-token
verification and HTTP scope checks remain active; there is no auth-disable environment flag.

The operator's ingest/reindex/verify-source commands now use CoreSettings and need no JWT
secret. `serve`, `token` and standalone OpenAPI export still require standalone settings.
Old HTTP/JWT Django connector examples remain supported as legacy adapters, not the
recommended token-free path. No API key, signing secret or existing `.env` was changed.

Once the real LMS integration is connected and tested, the optional standalone JWT
adapter/UI and its dependencies can be retired separately. They have not been deleted or
silently disabled here. Reuse existing benchmark references; authentication is a separate
integration/security concern, not a change to content quality or a paid benchmark result.

## Verification and errors

Offline tests cover direct calls and LMS-resolved HTTP calls without credentials, sync/
async resolvers, missing context, forged browser claims, wrong course/action, cross-tenant
storage boundaries and conversation ownership. Fresh-process tests block JWT imports
for the LMS HTTP factory and block both JWT/FastAPI imports for the embedded entry point.
Existing standalone API/UI tests remain in place. All models/sources in these tests are
controlled fixtures; no paid provider calls or LMS production integration is claimed.

For embedding, map `AccessDenied` to your LMS permission response, `ConversationNotFound`
to an inaccessible/expired conversation response, invalid request input to your validation
response, and `ModelUnavailable`/`ServiceBusy` to temporary service failure. Never render
an operational exception as "the course does not contain the answer". Do not expose raw
provider errors. See existing API exception handling for the standalone adapter's mappings.
