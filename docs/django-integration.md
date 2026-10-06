# Connecting Django to the FastAPI course assistant

Explained requests can make three sequential chat calls, plus an embedding call in hybrid mode. Set the connector's `timeout_seconds` and reverse proxy timeout to accommodate measured total query latency. `HR_MODEL_TIMEOUT_SECONDS` applies to individual provider HTTP operations, not the whole FastAPI query; the connector's default 120 seconds may need increasing for slower models.

The FastAPI service runs independently of the LMS. Django authorizes the learner,
mints a short-lived credential, and sends the request over a private connection or
HTTPS. The browser talks only to Django. Course text and learner questions do not
need to be exposed to a third-party model provider when using the local backend.

Copy `examples/django_connector.py` into your Django application's service layer.
It requires `httpx` and `PyJWT`; it does not import Django. Use the same high-entropy
HS256 secret on both servers (at least 32 characters). Generate one locally with
`python -c "import secrets; print(secrets.token_urlsafe(48))"` and store it in a
server-side secret store or deployment environment. Do not commit it or send it to
the browser. Use a production secret distinct from local development credentials.

## Authentication and authorization

The connector issues a new token for each request with these claims:

| Claim | Meaning |
| --- | --- |
| `iss` | `django-lms` by default; must match the API configuration |
| `aud` | `hrlearnium-api` by default; must match the API configuration |
| `sub` | Stable LMS user ID as a string |
| `tenant_id` | Tenant selected and authorized by Django |
| `course_ids` | Explicit list of authorized course IDs; connector sends just the requested course |
| `scopes` | `query`, `content:read`, or `content:write`, selected for the endpoint |
| `iat`, `exp` | Issue and expiry timestamps; 60-second validity by default |
| `jti` | Unique token identifier |

FastAPI enforces the tenant, course, scope, and conversation-owner boundaries.
Django must still check current enrollment on every query: signing a claim is an
authorization decision. Do not derive tenant IDs, authorized courses, or content
editing permissions from browser JSON. Do not use a single broad administrative
token for all learners. Clock synchronization matters for short-lived tokens.

## Example Django query view

This is an integration template, not a claim that the LMS has these models. Replace
`get_authorized_course_for_learner` with your real enrollment and tenant lookup.
It should return a course object only when the authenticated user currently has
access, otherwise raise Django's `Http404` or `PermissionDenied`. The object's
`assistant_course_id` and `tenant.assistant_tenant_id` must be trusted mappings
maintained by your server. Never implement the helper as a lookup by course ID
alone. Mount the route under your existing authenticated LMS application.

```python
import json
from uuid import UUID

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .services.django_connector import (
    HRLearniumAPIError,
    HRLearniumClient,
    HRLearniumProtocolError,
    HRLearniumUnavailable,
)
from .authorization import get_authorized_course_for_learner  # Implement for your LMS.


@login_required
@require_POST
def course_assistant(request, course_id):
    # Keep Django CsrfViewMiddleware enabled. No csrf_exempt decorator.
    course = get_authorized_course_for_learner(request.user, course_id)
    if len(request.body) > 16_384:
        return JsonResponse({"error": "request_too_large"}, status=413)
    try:
        payload = json.loads(request.body)
        if not isinstance(payload, dict):
            raise ValueError("Expected an object")
        if set(payload) - {"question", "conversation_id", "response_mode"}:
            raise ValueError("Unexpected fields")
        question = payload["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Invalid question")
        if len(question) > 2_000:
            raise ValueError("Question is too long")
        response_mode = payload.get("response_mode", "verbatim")
        if response_mode not in ("verbatim", "explained"):
            raise ValueError("Invalid response mode")
        conversation_id = payload.get("conversation_id")
        if conversation_id is not None:
            conversation_id = str(UUID(conversation_id))
    except (ValueError, TypeError, KeyError, AttributeError):
        return JsonResponse({"error": "invalid_request"}, status=400)

    try:
        with HRLearniumClient(
            base_url=settings.HRLEARNIUM_API_URL,
            jwt_secret=settings.HRLEARNIUM_JWT_SECRET,
            timeout_seconds=120,
        ) as client:
            result = client.query(
                subject=str(request.user.pk),
                tenant_id=str(course.tenant.assistant_tenant_id),
                course_id=str(course.assistant_course_id),
                question=question,
                conversation_id=conversation_id,
                response_mode=response_mode,
            )
    except HRLearniumUnavailable:
        # A timeout/503/429 is not evidence that the course lacks an answer.
        return JsonResponse({"error": "assistant_temporarily_unavailable"}, status=503)
    except HRLearniumAPIError as exc:
        # 401 from the backend usually indicates server credential/configuration
        # trouble, not that this already authenticated learner must log in again.
        if exc.status_code in {403, 404}:
            return JsonResponse({"error": "assistant_access_denied"}, status=403)
        if exc.status_code in {413, 422}:
            return JsonResponse({"error": "invalid_request"}, status=400)
        return JsonResponse({"error": "assistant_backend_error"}, status=502)
    except HRLearniumProtocolError:
        return JsonResponse({"error": "assistant_backend_error"}, status=502)

    response = JsonResponse(result, json_dumps_params={"ensure_ascii": False})
    response["Cache-Control"] = "no-store"
    return response
```

Keep the normal Django CSRF middleware and send the CSRF token with browser POST
requests. Also configure Django's request body limit; checking `request.body`
length happens after the application reads it. For a high-traffic deployment,
reuse one `HRLearniumClient` per worker process with graceful shutdown rather than
creating a client for each request. Set reverse-proxy and worker timeouts to
accommodate local inference (the connector defaults to 120 seconds), and cap
concurrent requests according to the model host's capacity. Do not automatically
retry a timed-out mutation or conversation request: the original may have
completed, and this API does not advertise idempotency keys.

The optional `conversation_id` is an opaque UUID returned by FastAPI. The browser
may send it back, but possession is not authorization: FastAPI checks its owner,
tenant, and course on every use. Keep separate conversation IDs per course and
reset the active conversation when a learner changes courses. Do not put user
content into token claims. Prefer logs with request ID, status, and latency over
logging authorization headers, complete questions, or source document text.

## Display contract

`POST /v1/courses/{course_id}/query` takes `question`, an optional
`conversation_id`, and `response_mode`. The mode is selected for each request:

| Mode | Answer behavior |
| --- | --- |
| `verbatim` (default) | Exact source excerpts only |
| `explained` | Exact source excerpts followed by a separate, cited explanation |

Omitting the mode preserves the original exact-excerpt behavior. A conversation
does not lock its mode: pass the desired mode again on each follow-up. If this
customer's deployment should offer only exact wording, have Django always pass
`response_mode="verbatim"` instead of exposing a mode selector to the browser.

Successful HTTP responses have one of three statuses:

| Status | UI behavior |
| --- | --- |
| `answered` | Display exact excerpts and citations, plus a separate explanation when requested |
| `refused` | Display the fixed Persian course-boundary message |
| `clarification` | Display the fixed Persian clarification prompt |

The response includes `request_id`, `status`, `answer`, `excerpts`,
`conversation_id`, `reason_code`, `retrieval_mode`, `response_mode`, `explanation`,
and `policy_version`. The returned mode must match the requested mode. Older
responses without these new fields are accepted only for `verbatim` requests.

For an answered `verbatim` request, `answer` is exactly the selected excerpt
texts joined with two newlines, and `explanation` is `null`. The connector rejects
any generated addition. For an answered `explained` request, `explanation` contains
`statements`: a list of objects with `text` and `citation_ids`. Each citation ID
must identify one of the response's included excerpts. Statements are trimmed and nonempty,
at most 1,200 characters each, with 1–8 distinct citations; there are 1–8 statements.

The combined `answer` begins with the same exact quotes, followed by two
newlines, `توضیح بر اساس متن دوره:`, a newline, and one statement per line.
Statements end with numbered citation markers such as `[1] [2]`, using the
one-based order of `excerpts`. The connector checks that the entire combined
answer matches this structure, that the quotes are unchanged, and that every
explanation citation resolves. A refusal or clarification has no excerpts and
no explanation in either mode. A missing explanation in an answered `explained`
response is a protocol error, not a silent change back to verbatim mode.

Render the explanation separately from the quoted course text and label it as
an explanation. It is generated text, not an exact quotation. Do not paraphrase,
translate, or append new advice in Django or the browser. The connector checks
structure and citation consistency; it does not independently establish that a
generated statement is supported. Semantic grounding is the backend's separate
verification step.

Each excerpt carries `id`, `text`, and `citation` with `document_id`, `version`,
`document_title`, `chapter_number`, `chapter_title`, `section_kind`,
`paragraph_start`, `paragraph_end`, `source_start`, `source_end`, and
`source_sha256`. Render citations separately from the answer. Paragraph and source
offsets refer to the extracted source, not Word's visual page numbers. Show the
document version so previously returned answers remain traceable after an update.
Render all text through escaped templates or DOM `textContent`, with `dir="rtl"`
and `white-space: pre-wrap` for Persian excerpts. Do not interpret document text
as HTML or executable Markdown. Apply the same escaping to explanation statements.

Network errors, HTTP 429, and HTTP 5xx raise `HRLearniumUnavailable`. Other
unsuccessful statuses raise `HRLearniumAPIError` with `status_code`. Invalid JSON
or an inconsistent query answer raises `HRLearniumProtocolError`. These are
operational errors; never replace them with an out-of-course refusal. The client
does not follow redirects, so bearer credentials are not forwarded to a redirect
destination.

## Content administration

Authorize course administration separately from learner enrollment before using
these connector methods. File arguments are opened binary streams; the caller
owns and closes them.

| Connector method | API | Scope |
| --- | --- | --- |
| `upload_document` | `POST /v1/courses/{course}/documents`, multipart `file` | `content:write` |
| `list_documents` | `GET /v1/courses/{course}/documents` | `content:read` |
| `replace_document` | `PUT /v1/courses/{course}/documents/{document}`, multipart `file` | `content:write` |
| `delete_document` | `DELETE /v1/courses/{course}/documents/{document}` | `content:write` |
| `get_excerpt` | `GET /v1/courses/{course}/excerpts/{excerpt}` | `query` |

Replacing a document creates a new version. While that document is active, use
the excerpt endpoint to resolve current or archived citation IDs within the
authorized course. Deletion disables future retrieval and hides that document's
excerpts from this endpoint; it is not a promise of erasing all historical
citation records. Do not cache content across tenants, users, or courses.
Document and excerpt IDs do not grant access by themselves.
