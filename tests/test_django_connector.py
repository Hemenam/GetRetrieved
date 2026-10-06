from __future__ import annotations

import io
import json
from uuid import UUID, uuid4

import httpx
import jwt
import pytest

from examples.django_connector import (
    HRLearniumAPIError,
    HRLearniumClient,
    HRLearniumProtocolError,
    HRLearniumUnavailable,
    mint_lms_token,
)
from hrlearnium.schemas import Excerpt, Explanation, render_answer

SECRET = "test-connector-signing-secret-32-characters-minimum"
IDENTITY = {"subject": "learner-1", "tenant_id": "tenant-1", "course_id": "course-1"}


def decode_request_token(request: httpx.Request) -> dict:
    prefix, token = request.headers["authorization"].split(" ", 1)
    assert prefix == "Bearer"
    return jwt.decode(
        token, SECRET, algorithms=["HS256"], audience="hrlearnium-api", issuer="django-lms"
    )


def client_for(handler) -> HRLearniumClient:
    return HRLearniumClient(
        base_url="https://assistant.example.test",
        jwt_secret=SECRET,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_token_is_short_lived_and_scoped_with_unique_identifier():
    kwargs = {
        "secret": SECRET,
        "subject": "learner-1",
        "tenant_id": "tenant-1",
        "course_ids": ["course-1"],
        "scopes": ["query"],
    }
    first = jwt.decode(
        mint_lms_token(**kwargs), SECRET, algorithms=["HS256"], audience="hrlearnium-api"
    )
    second = jwt.decode(
        mint_lms_token(**kwargs), SECRET, algorithms=["HS256"], audience="hrlearnium-api"
    )
    assert first["iss"] == "django-lms"
    assert first["sub"] == "learner-1"
    assert first["tenant_id"] == "tenant-1"
    assert first["course_ids"] == ["course-1"]
    assert first["scopes"] == ["query"]
    assert first["exp"] - first["iat"] == 60
    assert UUID(first["jti"])
    assert first["jti"] != second["jti"]


@pytest.mark.parametrize(
    "change",
    [
        {"secret": "too-short"},
        {"course_ids": ["*"]},
        {"course_ids": "course-1"},
        {"scopes": ["*"]},
        {"subject": ""},
        {"tenant_id": " "},
        {"ttl_seconds": 0},
        {"ttl_seconds": 301},
        {"ttl_seconds": True},
    ],
)
def test_invalid_authorization_claims_are_rejected_locally(change):
    values = {
        "secret": SECRET,
        "subject": "learner-1",
        "tenant_id": "tenant-1",
        "course_ids": ["course-1"],
        "scopes": ["query"],
    }
    values.update(change)
    with pytest.raises(ValueError):
        mint_lms_token(**values)


def test_query_sends_only_question_and_conversation_with_scoped_token():
    conversation_id = str(uuid4())
    excerpt_text = "واقعیت\nاقدام سازمان\nگام بعدی کارکنان"

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/courses/course-1/query"
        assert json.loads(request.content) == {
            "question": "پیام اضطراری چیست؟",
            "conversation_id": conversation_id,
            "response_mode": "verbatim",
        }
        claims = decode_request_token(request)
        assert claims["course_ids"] == ["course-1"]
        assert claims["tenant_id"] == "tenant-1"
        assert claims["sub"] == "learner-1"
        assert claims["scopes"] == ["query"]
        assert request.extensions["timeout"]["read"] == 120
        return httpx.Response(
            200,
            json={
                "status": "answered",
                "answer": excerpt_text,
                "excerpts": [{"id": "excerpt-1", "text": excerpt_text, "citation": {}}],
                "conversation_id": conversation_id,
            },
        )

    result = client_for(handler).query(
        **IDENTITY, question="پیام اضطراری چیست؟", conversation_id=conversation_id
    )
    assert result["answer"] == excerpt_text


@pytest.mark.parametrize("status", ["refused", "clarification"])
def test_content_refusal_is_a_normal_success_response(status):
    result = client_for(
        lambda request: httpx.Response(
            200,
            json={
                "status": status,
                "answer": "فقط بر اساس محتوای دوره پاسخ می‌دهم.",
                "excerpts": [],
            },
        )
    ).query(**IDENTITY, question="یک پرسش")
    assert result["status"] == status


@pytest.mark.parametrize("status_code", [429, 500, 502, 503])
def test_backend_errors_are_never_converted_to_content_refusals(status_code):
    with pytest.raises(HRLearniumUnavailable) as caught:
        client_for(
            lambda request: httpx.Response(status_code, json={"detail": "internal detail"})
        ).query(
            **IDENTITY,
            question="یک پرسش",
        )
    assert caught.value.status_code == status_code
    assert "internal detail" not in str(caught.value)


def test_timeout_is_reported_as_operational_failure_without_retry():
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("Timed out", request=request)

    with pytest.raises(HRLearniumUnavailable):
        client_for(handler).query(**IDENTITY, question="یک پرسش")
    assert len(requests) == 1


@pytest.mark.parametrize("status_code", [401, 403, 404, 413, 422])
def test_api_status_is_preserved_without_leaking_body(status_code):
    with pytest.raises(HRLearniumAPIError) as caught:
        client_for(
            lambda request: httpx.Response(
                status_code,
                headers={"x-request-id": "req-1"},
                json={"detail": "private source text"},
            )
        ).query(**IDENTITY, question="یک پرسش")
    assert caught.value.status_code == status_code
    assert caught.value.request_id == "req-1"
    assert "private source text" not in str(caught.value)


def test_redirects_are_not_followed():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            307, headers={"location": "https://different.example.test/steal-token"}
        )

    with pytest.raises(HRLearniumAPIError) as caught:
        client_for(handler).query(**IDENTITY, question="یک پرسش")
    assert caught.value.status_code == 307
    assert len(requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        {"status": "answered", "answer": "invented answer", "excerpts": [{"text": "original"}]},
        {"status": "answered", "answer": "", "excerpts": []},
        {"status": "answered", "answer": "original", "excerpts": [{"text": 42}]},
        {"status": "unknown", "answer": "", "excerpts": []},
        {"status": ["answered"], "answer": "", "excerpts": []},
        {"status": "refused", "answer": 42, "excerpts": []},
        [],
    ],
)
def test_malformed_or_synthesized_query_response_is_rejected(response):
    with pytest.raises(HRLearniumProtocolError):
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY, question="یک پرسش"
        )


def test_non_json_response_is_rejected():
    with pytest.raises(HRLearniumProtocolError):
        client_for(lambda request: httpx.Response(200, text="not JSON")).query(
            **IDENTITY, question="یک پرسش"
        )


def test_document_and_excerpt_operations_use_endpoint_specific_scopes():
    calls = []

    def handler(request):
        claims = decode_request_token(request)
        calls.append((request.method, request.url.path, claims["scopes"]))
        assert claims["course_ids"] == ["course-1"]
        if request.method in {"POST", "PUT"}:
            assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
            assert b'name="file"; filename="course.docx"' in request.content
            assert b"fake docx bytes" in request.content
        return (
            httpx.Response(204)
            if request.method == "DELETE"
            else httpx.Response(200, json={"ok": True})
        )

    client = client_for(handler)
    client.list_documents(**IDENTITY)
    client.upload_document(
        **IDENTITY, file=io.BytesIO(b"fake docx bytes"), filename="private/path/course.docx"
    )
    client.replace_document(
        **IDENTITY, document_id="doc-1", file=io.BytesIO(b"fake docx bytes"), filename="course.docx"
    )
    client.get_excerpt(**IDENTITY, excerpt_id="excerpt-1")
    assert client.delete_document(**IDENTITY, document_id="doc-1") is None
    assert calls == [
        ("GET", "/v1/courses/course-1/documents", ["content:read"]),
        ("POST", "/v1/courses/course-1/documents", ["content:write"]),
        ("PUT", "/v1/courses/course-1/documents/doc-1", ["content:write"]),
        ("GET", "/v1/courses/course-1/excerpts/excerpt-1", ["query"]),
        ("DELETE", "/v1/courses/course-1/documents/doc-1", ["content:write"]),
    ]


def test_injected_http_client_is_not_closed_by_connector():
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(204)))
    with HRLearniumClient(
        base_url="https://assistant.example.test", jwt_secret=SECRET, http_client=http_client
    ):
        pass
    assert not http_client.is_closed
    http_client.close()


def explained_response():
    excerpts = []
    for index, text in enumerate(["  واقعیت\nاقدام سازمان  ", "گام بعدی کارکنان"], start=1):
        excerpts.append(
            {
                "id": f"excerpt-{index}",
                "text": text,
                "citation": {
                    "document_id": "document-1",
                    "version": 1,
                    "document_title": "دوره",
                    "chapter_number": 1,
                    "chapter_title": "پیام اضطراری",
                    "section_kind": "explanation",
                    "paragraph_start": index,
                    "paragraph_end": index,
                    "source_start": 0,
                    "source_end": len(text),
                    "source_sha256": "a" * 64,
                },
            }
        )
    return {
        "status": "answered",
        "response_mode": "explained",
        "excerpts": excerpts,
        "explanation": {
            "statements": [
                {"text": "پیام شامل واقعیت و اقدام سازمان است.", "citation_ids": ["excerpt-1"]},
                {
                    "text": "گام بعدی کارکنان همراه با اقدام سازمان بیان می‌شود.",
                    "citation_ids": ["excerpt-2", "excerpt-1"],
                },
            ]
        },
        "answer": (
            "  واقعیت\nاقدام سازمان  \n\nگام بعدی کارکنان"
            "\n\nتوضیح بر اساس متن دوره:\n"
            "پیام شامل واقعیت و اقدام سازمان است. [1]\n"
            "گام بعدی کارکنان همراه با اقدام سازمان بیان می‌شود. [2] [1]"
        ),
    }


def test_explained_query_sends_mode_and_preserves_quotes_and_numbered_statement_citations():
    response = explained_response()

    def handler(request):
        assert json.loads(request.content) == {
            "question": "پیام اضطراری چیست؟",
            "response_mode": "explained",
        }
        assert decode_request_token(request)["scopes"] == ["query"]
        return httpx.Response(200, json=response)

    result = client_for(handler).query(
        **IDENTITY, question="پیام اضطراری چیست؟", response_mode="explained"
    )
    assert result == response
    assert result["answer"] == render_answer(
        [Excerpt.model_validate(item) for item in result["excerpts"]],
        Explanation.model_validate(result["explanation"]),
    )


@pytest.mark.parametrize("mode", [None, "summary", 1, ["explained"], True])
def test_invalid_requested_response_mode_is_rejected_before_network_request(mode):
    def unexpected_request(request):
        pytest.fail("Invalid response mode reached the network")

    with pytest.raises(ValueError, match="response_mode"):
        client_for(unexpected_request).query(**IDENTITY, question="یک پرسش", response_mode=mode)


@pytest.mark.parametrize(
    "requested,returned",
    [
        ("verbatim", "explained"),
        ("explained", "verbatim"),
        ("explained", "missing"),
        ("verbatim", None),
        ("verbatim", "unknown"),
        ("explained", ["explained"]),
    ],
)
def test_response_mode_must_match_requested_mode(requested, returned):
    response = explained_response()
    if returned == "missing":
        response.pop("response_mode")
    else:
        response["response_mode"] = returned
    with pytest.raises(HRLearniumProtocolError, match="mode"):
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY,
            question="یک پرسش",
            response_mode=requested,
        )


@pytest.mark.parametrize("status", ["refused", "clarification"])
@pytest.mark.parametrize("mode", ["verbatim", "explained"])
def test_refusals_and_clarifications_preserve_requested_mode_without_explanation(status, mode):
    response = {
        "status": status,
        "answer": "پیام ثابت دوره",
        "excerpts": [],
        "response_mode": mode,
        "explanation": None,
    }
    assert (
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY,
            question="یک پرسش",
            response_mode=mode,
        )
        == response
    )


def test_verbatim_cannot_include_generated_explanation_even_when_quotes_are_preserved():
    response = explained_response()
    response["response_mode"] = "verbatim"
    with pytest.raises(HRLearniumProtocolError, match="Verbatim"):
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY, question="یک پرسش"
        )


@pytest.mark.parametrize("status", ["refused", "clarification"])
@pytest.mark.parametrize("unwanted", ["excerpts", "explanation"])
def test_nonanswered_statuses_cannot_smuggle_quotes_or_generated_statements(status, unwanted):
    response = explained_response()
    response["status"] = status
    if unwanted == "excerpts":
        response["explanation"] = None
    else:
        response["excerpts"] = []
    with pytest.raises(HRLearniumProtocolError):
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY,
            question="یک پرسش",
            response_mode="explained",
        )


@pytest.mark.parametrize(
    "citation_ids",
    [
        [],
        ["excerpt-1", "excerpt-1"],
        ["foreign-excerpt"],
        [""],
        [1],
        [["excerpt-1"]],
        None,
        "excerpt-1",
        ["excerpt-1"] * 9,
    ],
)
def test_explanation_citations_must_be_nonempty_unique_bounded_and_included(citation_ids):
    response = explained_response()
    response["explanation"]["statements"][0]["citation_ids"] = citation_ids
    with pytest.raises(HRLearniumProtocolError, match="citations"):
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY,
            question="یک پرسش",
            response_mode="explained",
        )


@pytest.mark.parametrize("text", ["", "   ", "x" * 1201, 42, " متن با فاصله "])
def test_explanation_text_follows_shared_schema_bounds(text):
    response = explained_response()
    response["explanation"]["statements"][0]["text"] = text
    with pytest.raises(HRLearniumProtocolError, match="text"):
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY,
            question="یک پرسش",
            response_mode="explained",
        )


@pytest.mark.parametrize(
    "explanation",
    [
        None,
        {},
        {"statements": []},
        {"statements": None},
        {"statements": "invalid"},
        {"statements": [{"text": "متن", "citation_ids": ["excerpt-1"]}] * 9},
        {"statements": [{"text": "متن", "citation_ids": ["excerpt-1"], "new_advice": "extra"}]},
        {"statements": [{"text": "متن"}]},
        {"statements": [None]},
        {"statements": [{"text": "متن", "citation_ids": ["excerpt-1"]}], "answer": "extra"},
    ],
)
def test_explained_answers_require_strict_structured_explanation(explanation):
    response = explained_response()
    response["explanation"] = explanation
    with pytest.raises(HRLearniumProtocolError):
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY,
            question="یک پرسش",
            response_mode="explained",
        )


@pytest.mark.parametrize(
    "mutation", ["added_text", "altered_quote", "wrong_number", "missing_explanation"]
)
def test_combined_answer_must_exactly_match_quotes_and_declared_explanation(mutation):
    response = explained_response()
    if mutation == "added_text":
        response["answer"] += "\nیک پیشنهاد جدید"
    elif mutation == "altered_quote":
        response["answer"] = response["answer"].replace("  واقعیت", "واقعیت", 1)
    elif mutation == "wrong_number":
        response["answer"] = response["answer"].replace("[2] [1]", "[1] [2]")
    else:
        response["answer"] = "\n\n".join(item["text"] for item in response["excerpts"])
    with pytest.raises(HRLearniumProtocolError, match="differs"):
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY,
            question="یک پرسش",
            response_mode="explained",
        )


def test_duplicate_excerpt_ids_cannot_make_explanation_citation_numbers_ambiguous():
    response = explained_response()
    response["excerpts"][1]["id"] = response["excerpts"][0]["id"]
    with pytest.raises(HRLearniumProtocolError, match="unique"):
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY,
            question="یک پرسش",
            response_mode="explained",
        )


def test_explanation_accepts_shared_schema_maximum_bounds():
    response = explained_response()
    response["excerpts"] = [
        {**response["excerpts"][0], "id": f"excerpt-{index}"} for index in range(1, 9)
    ]
    response["explanation"] = {
        "statements": [
            {"text": "م" * 1200, "citation_ids": [item["id"] for item in response["excerpts"]]}
            for _ in range(8)
        ]
    }
    response["answer"] = render_answer(
        [Excerpt.model_validate(item) for item in response["excerpts"]],
        Explanation.model_validate(response["explanation"]),
    )
    assert (
        client_for(lambda request: httpx.Response(200, json=response)).query(
            **IDENTITY,
            question="یک پرسش",
            response_mode="explained",
        )
        == response
    )
