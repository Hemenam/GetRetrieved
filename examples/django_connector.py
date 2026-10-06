"""Server-to-server LMS connector. Keep the signing secret on the Django server.

This module has no Django dependency; copy it into your Django service layer.
The caller must authorize enrollment and tenant membership before calling it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Literal
from urllib.parse import quote, urlsplit
from uuid import UUID, uuid4

import httpx
import jwt


class HRLearniumError(Exception):
    """A connector failure, never a course-content refusal."""


class HRLearniumAPIError(HRLearniumError):
    """The API rejected the request; response bodies are deliberately excluded."""

    def __init__(self, status_code: int, request_id: str | None = None) -> None:
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(f"HRLearnium API returned HTTP {status_code}")


class HRLearniumUnavailable(HRLearniumError):
    """Network, timeout, rate limit, or backend availability failure."""

    def __init__(self, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__("The course assistant is temporarily unavailable")


class HRLearniumProtocolError(HRLearniumError):
    """A successful HTTP response did not follow the expected API contract."""


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def mint_lms_token(
    *,
    secret: str,
    subject: str,
    tenant_id: str,
    course_ids: Sequence[str],
    scopes: Sequence[str],
    ttl_seconds: int = 60,
    issuer: str = "django-lms",
    audience: str = "hrlearnium-api",
) -> str:
    """Mint a narrowly scoped credential after server-side LMS authorization.

    Do not accept these claims directly from browser JSON. The shared HS256 key
    grants the ability to impersonate any LMS user and must remain server-side.
    """
    if not isinstance(secret, str) or len(secret) < 32:
        raise ValueError("The signing secret must contain at least 32 characters")
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int)
        or not 1 <= ttl_seconds <= 300
    ):
        raise ValueError("ttl_seconds must be an integer between 1 and 300")
    _nonempty(subject, "subject")
    _nonempty(tenant_id, "tenant_id")
    _nonempty(issuer, "issuer")
    _nonempty(audience, "audience")
    if isinstance(course_ids, (str, bytes)) or not course_ids:
        raise ValueError("course_ids must be a nonempty sequence of authorized course IDs")
    if isinstance(scopes, (str, bytes)) or not scopes:
        raise ValueError("scopes must be a nonempty sequence")
    authorized_courses = list(dict.fromkeys(_nonempty(item, "course_id") for item in course_ids))
    authorized_scopes = list(dict.fromkeys(_nonempty(item, "scope") for item in scopes))
    if any("*" in item for item in authorized_courses + authorized_scopes):
        raise ValueError("Wildcard course IDs and scopes are not permitted")
    issued_at = int(time.time())
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": subject,
            "tenant_id": tenant_id,
            "course_ids": authorized_courses,
            "scopes": authorized_scopes,
            "iat": issued_at,
            "exp": issued_at + ttl_seconds,
            "jti": str(uuid4()),
        },
        secret,
        algorithm="HS256",
    )


def _path_id(value: str, name: str) -> str:
    value = _nonempty(value, name)
    if value in {".", ".."}:
        raise ValueError(f"Invalid {name}")
    return quote(value, safe="")


def _validate_query_answer(result: dict[str, Any], response_mode: str) -> None:
    """Check the wire contract before Django displays quotes or an explanation.

    This is structural validation, not a second semantic grounding model.
    """
    returned_mode = result.get("response_mode", "verbatim")
    if returned_mode != response_mode:
        raise HRLearniumProtocolError("The response mode differs from the requested mode")
    excerpts = result["excerpts"]
    explanation = result.get("explanation")
    if result["status"] != "answered":
        if excerpts or explanation is not None:
            raise HRLearniumProtocolError(
                "A refusal or clarification cannot contain evidence or an explanation"
            )
        return
    if not 1 <= len(excerpts) <= 8 or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("text"), str)
        or not item["text"].strip()
        or not isinstance(item.get("id"), str)
        or not item["id"].strip()
        for item in excerpts
    ):
        raise HRLearniumProtocolError(
            "An answered response must contain identified source excerpts"
        )
    excerpt_ids = [item["id"] for item in excerpts]
    if len(excerpt_ids) != len(set(excerpt_ids)):
        raise HRLearniumProtocolError("Source excerpt IDs must be unique")
    expected_answer = "\n\n".join(item["text"] for item in excerpts)
    if response_mode == "verbatim":
        if explanation is not None:
            raise HRLearniumProtocolError(
                "Verbatim responses cannot contain generated explanations"
            )
    else:
        expected_answer += _render_explanation(explanation, excerpt_ids)
    if result["answer"] != expected_answer:
        raise HRLearniumProtocolError(
            "The answer differs from its source excerpts and declared explanation"
        )


def _render_explanation(explanation: Any, excerpt_ids: list[str]) -> str:
    """Validate the explanation schema without requiring the FastAPI package."""
    if not isinstance(explanation, dict) or set(explanation) != {"statements"}:
        raise HRLearniumProtocolError("An explained answer must contain structured statements")
    statements = explanation["statements"]
    if not isinstance(statements, list) or not 1 <= len(statements) <= 8:
        raise HRLearniumProtocolError("Invalid number of explanation statements")
    citation_numbers = {identifier: index for index, identifier in enumerate(excerpt_ids, start=1)}
    rendered = []
    for statement in statements:
        if not isinstance(statement, dict) or set(statement) != {"text", "citation_ids"}:
            raise HRLearniumProtocolError("Invalid explanation statement")
        text, citation_ids = statement["text"], statement["citation_ids"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or text != text.strip()
            or len(text) > 1200
        ):
            raise HRLearniumProtocolError("Invalid explanation text")
        if (
            not isinstance(citation_ids, list)
            or not 1 <= len(citation_ids) <= 8
            or any(
                not isinstance(identifier, str) or not identifier.strip()
                for identifier in citation_ids
            )
            or len(citation_ids) != len(set(citation_ids))
            or any(identifier not in citation_numbers for identifier in citation_ids)
        ):
            raise HRLearniumProtocolError(
                "Explanation citations must identify included source excerpts"
            )
        markers = " ".join(f"[{citation_numbers[identifier]}]" for identifier in citation_ids)
        rendered.append(f"{text} {markers}")
    return "\n\nتوضیح بر اساس متن دوره:\n" + "\n".join(rendered)


class HRLearniumClient:
    """Small synchronous connector for Django views or worker tasks.

    Reuse a client per worker process and close it at shutdown, or use a context
    manager. This client does not retry requests or follow redirects. Document
    ingestion is privileged; callers must separately authorize content editing.
    """

    def __init__(
        self,
        *,
        base_url: str,
        jwt_secret: str,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 120,
        issuer: str = "django-lms",
        audience: str = "hrlearnium-api",
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, a query, or a fragment")
        if not isinstance(jwt_secret, str) or len(jwt_secret) < 32:
            raise ValueError("The signing secret must contain at least 32 characters")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self._secret = jwt_secret
        self._issuer = issuer
        self._audience = audience
        self._timeout = httpx.Timeout(timeout_seconds, connect=5.0)
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(trust_env=False)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HRLearniumClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        subject: str,
        tenant_id: str,
        course_id: str,
        scope: str,
        **kwargs: Any,
    ) -> Any:
        token = mint_lms_token(
            secret=self._secret,
            subject=subject,
            tenant_id=tenant_id,
            course_ids=[course_id],
            scopes=[scope],
            issuer=self._issuer,
            audience=self._audience,
        )
        try:
            response = self._client.request(
                method,
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                timeout=self._timeout,
                follow_redirects=False,
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise HRLearniumUnavailable() from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise HRLearniumUnavailable(response.status_code)
        if not 200 <= response.status_code < 300:
            raise HRLearniumAPIError(response.status_code, response.headers.get("x-request-id"))
        if response.status_code == 204:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise HRLearniumProtocolError("Expected a JSON API response") from exc

    @staticmethod
    def _course_path(course_id: str) -> str:
        return f"/v1/courses/{_path_id(course_id, 'course_id')}"

    def query(
        self,
        *,
        subject: str,
        tenant_id: str,
        course_id: str,
        question: str,
        conversation_id: str | UUID | None = None,
        response_mode: Literal["verbatim", "explained"] = "verbatim",
    ) -> dict[str, Any]:
        """Return exact excerpts with an optional cited explanation or a fixed refusal."""
        if not isinstance(response_mode, str) or response_mode not in {"verbatim", "explained"}:
            raise ValueError("response_mode must be verbatim or explained")
        payload: dict[str, Any] = {
            "question": _nonempty(question, "question"),
            "response_mode": response_mode,
        }
        if conversation_id is not None:
            payload["conversation_id"] = str(UUID(str(conversation_id)))
        result = self._request(
            "POST",
            f"{self._course_path(course_id)}/query",
            subject=subject,
            tenant_id=tenant_id,
            course_id=course_id,
            scope="query",
            json=payload,
        )
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("status"), str)
            or result["status"] not in {"answered", "refused", "clarification"}
        ):
            raise HRLearniumProtocolError("Invalid query response status")
        if not isinstance(result.get("answer"), str) or not isinstance(
            result.get("excerpts"), list
        ):
            raise HRLearniumProtocolError("Invalid answer or excerpts in query response")
        _validate_query_answer(result, response_mode)
        return result

    def list_documents(self, *, subject: str, tenant_id: str, course_id: str) -> Any:
        return self._request(
            "GET",
            f"{self._course_path(course_id)}/documents",
            subject=subject,
            tenant_id=tenant_id,
            course_id=course_id,
            scope="content:read",
        )

    def upload_document(
        self,
        *,
        subject: str,
        tenant_id: str,
        course_id: str,
        file: BinaryIO,
        filename: str,
    ) -> Any:
        return self._upload(
            "POST",
            f"{self._course_path(course_id)}/documents",
            subject=subject,
            tenant_id=tenant_id,
            course_id=course_id,
            file=file,
            filename=filename,
        )

    def replace_document(
        self,
        *,
        subject: str,
        tenant_id: str,
        course_id: str,
        document_id: str,
        file: BinaryIO,
        filename: str,
    ) -> Any:
        return self._upload(
            "PUT",
            f"{self._course_path(course_id)}/documents/{_path_id(document_id, 'document_id')}",
            subject=subject,
            tenant_id=tenant_id,
            course_id=course_id,
            file=file,
            filename=filename,
        )

    def _upload(
        self,
        method: str,
        path: str,
        *,
        subject: str,
        tenant_id: str,
        course_id: str,
        file: BinaryIO,
        filename: str,
    ) -> Any:
        # Multipart content type (including the boundary) is set by httpx.
        safe_filename = Path(_nonempty(filename, "filename").replace("\\", "/")).name
        return self._request(
            method,
            path,
            subject=subject,
            tenant_id=tenant_id,
            course_id=course_id,
            scope="content:write",
            files={
                "file": (
                    safe_filename,
                    file,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    def get_excerpt(self, *, subject: str, tenant_id: str, course_id: str, excerpt_id: str) -> Any:
        return self._request(
            "GET",
            f"{self._course_path(course_id)}/excerpts/{_path_id(excerpt_id, 'excerpt_id')}",
            subject=subject,
            tenant_id=tenant_id,
            course_id=course_id,
            scope="query",
        )

    def delete_document(
        self, *, subject: str, tenant_id: str, course_id: str, document_id: str
    ) -> Any:
        return self._request(
            "DELETE",
            f"{self._course_path(course_id)}/documents/{_path_id(document_id, 'document_id')}",
            subject=subject,
            tenant_id=tenant_id,
            course_id=course_id,
            scope="content:write",
        )
