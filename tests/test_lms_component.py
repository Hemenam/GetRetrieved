"""Token-free embedding and injected LMS identity; controlled sources/models only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_api import BASE, COURSE, FakeGateway, docx, token

from hrlearnium.access import AccessDenied
from hrlearnium.component import create_assistant
from hrlearnium.config import Settings
from hrlearnium.core_config import CoreSettings
from hrlearnium.main import create_app, create_lms_app
from hrlearnium.schemas import Principal, QueryRequest
from hrlearnium.service import ConversationNotFound


def context(*, user="learner-1", tenant="tenant-1", courses=(COURSE,), scopes=("query",)):
    return Principal(sub=user, tenant_id=tenant, course_ids=list(courses), scopes=list(scopes))


@pytest.fixture
def settings(tmp_path):
    return CoreSettings(
        _env_file=None,
        database_path=tmp_path / "component.db",
        model_backend="ollama",
        retrieval_mode="full_context",
    )


@pytest.fixture
def embedded(settings):
    assistant = create_assistant(settings, gateway=FakeGateway())
    manager = context(scopes=("query", "content:read", "content:write"))
    document, _ = assistant.ingest(manager, COURSE, "course.docx", docx())
    yield assistant, manager, document
    assistant.close()


def test_core_settings_need_no_jwt_secret_but_standalone_still_does(settings):
    assert "jwt_secret" not in type(settings).model_fields
    assert not settings.enable_chat_ui
    with pytest.raises(ValidationError, match="HR_JWT_SECRET"):
        Settings(_env_file=None, jwt_secret="", database_path=settings.database_path)


def test_embedded_query_outline_sources_and_content_management_without_tokens(embedded):
    assistant, manager, document = embedded
    learner = context()
    response = assistant.query(
        learner, COURSE, QueryRequest(question="پیام اضطراری چیست؟", response_mode="verbatim")
    )
    assert response.status == "answered" and response.request_id and response.excerpts
    assert len(assistant.course_outline(learner, COURSE).documents[0].chapters) == 2
    assert (
        assistant.get_excerpt(learner, COURSE, response.excerpts[0].id).text
        == response.excerpts[0].text
    )
    assert assistant.list_documents(manager, COURSE)[0].id == document.id
    assert assistant.retire_document(manager, COURSE, document.id)
    assert assistant.get_excerpt(learner, COURSE, response.excerpts[0].id) is None


@pytest.mark.parametrize(
    "method",
    [
        "query",
        "ingest",
        "retrieval_options",
        "course_outline",
        "list_documents",
        "get_excerpt",
        "retire_document",
    ],
)
def test_embedded_course_permission_is_enforced_before_storage_or_model_work(
    embedded, monkeypatch, method
):
    assistant, _, document = embedded
    unauthorized = context(
        courses=("another-course",), scopes=("query", "content:read", "content:write")
    )

    def forbidden(*a, **kw):
        pytest.fail("Denied access must not touch storage/model")

    monkeypatch.setattr(assistant.storage, "search_passages", forbidden)
    monkeypatch.setattr(assistant.storage, "course_outline", forbidden)
    monkeypatch.setattr(assistant.storage, "get_excerpts", forbidden)
    monkeypatch.setattr(assistant.storage, "get_document", forbidden)
    monkeypatch.setattr(assistant.storage, "list_documents", forbidden)
    monkeypatch.setattr(assistant.storage, "retire_document", forbidden)
    calls = {
        "query": lambda: assistant.query(unauthorized, COURSE, QueryRequest(question="سلام")),
        "ingest": lambda: assistant.ingest(unauthorized, COURSE, "course.docx", b"invalid"),
        "retrieval_options": lambda: assistant.retrieval_options(unauthorized, COURSE),
        "course_outline": lambda: assistant.course_outline(unauthorized, COURSE),
        "list_documents": lambda: assistant.list_documents(unauthorized, COURSE),
        "get_excerpt": lambda: assistant.get_excerpt(unauthorized, COURSE, "anything"),
        "retire_document": lambda: assistant.retire_document(unauthorized, COURSE, document.id),
    }
    with pytest.raises(AccessDenied):
        calls[method]()


def test_embedded_action_scopes_and_tenant_filters_are_preserved(embedded):
    assistant, _, document = embedded
    learner = context()
    with pytest.raises(AccessDenied):
        assistant.ingest(learner, COURSE, "new.docx", docx())
    with pytest.raises(AccessDenied):
        assistant.list_documents(learner, COURSE)
    with pytest.raises(AccessDenied):
        assistant.retire_document(learner, COURSE, document.id)
    reader = context(scopes=("content:read",))
    with pytest.raises(AccessDenied):
        assistant.query(reader, COURSE, QueryRequest(question="سلام"))
    other = context(tenant="tenant-2", scopes=("query", "content:read", "content:write"))
    assert assistant.course_outline(other, COURSE).documents == []
    assert assistant.list_documents(other, COURSE) == []
    assert not assistant.retire_document(other, COURSE, document.id)


@pytest.mark.parametrize("other", [context(user="another-user"), context(tenant="tenant-2")])
def test_embedded_conversation_ownership_survives_without_jwt(embedded, other):
    assistant, _, _ = embedded
    first = assistant.query(context(), COURSE, QueryRequest(question="سلام"))
    with pytest.raises(ConversationNotFound):
        assistant.query(
            other, COURSE, QueryRequest(question="سلام", conversation_id=first.conversation_id)
        )


@pytest.fixture
def lms_http(settings):
    def resolver(request: Request):
        # Test-only simulated LMS identity; no browser headers/JSON are consulted.
        return request.app.state.test_principal

    settings.enable_chat_ui = True  # LMS factory must suppress even an explicit dev-UI flag.
    app = create_lms_app(resolver, settings, FakeGateway())
    app.state.test_principal = context(scopes=("query", "content:read", "content:write"))
    with TestClient(app) as client:
        result = client.post(BASE + "/documents", files={"file": ("course.docx", docx())})
        assert result.status_code == 201
        yield client, app


def test_lms_http_requires_no_bearer_and_has_no_standalone_token_ui(lms_http):
    client, app = lms_http
    assert app.state.authentication_adapter == "lms"
    response = client.post(BASE + "/query", json={"question": "این دوره چند فصل دارد؟"})
    assert response.status_code == 200 and response.json()["status"] == "answered"
    assert "2" in response.json()["answer"]
    assert client.get("/chat").status_code == 404
    assert client.get("/chat/assets/app.mjs").status_code == 404
    assert "HTTPBearer" not in json.dumps(app.openapi())


@pytest.mark.parametrize("invalid", [None, {"sub": "forged-user", "tenant_id": "tenant-1"}])
def test_lms_missing_identity_never_falls_back_to_jwt_or_browser_claims(lms_http, invalid):
    client, app = lms_http
    app.state.test_principal = invalid
    headers = {
        **token(),
        "X-LMS-User": "learner-1",
        "X-LMS-Tenant": "tenant-1",
        "X-LMS-Course": COURSE,
    }
    response = client.post(BASE + "/query", headers=headers, json={"question": "سلام"})
    assert response.status_code == 401


def test_lms_http_still_rejects_wrong_course_action_and_body_identity_fields(lms_http):
    client, app = lms_http
    app.state.test_principal = context()
    assert (
        client.post("/v1/courses/another-course/query", json={"question": "سلام"}).status_code
        == 403
    )
    assert (
        client.post(BASE + "/documents", files={"file": ("course.docx", docx())}).status_code == 403
    )
    assert client.get(BASE + "/documents").status_code == 403
    assert (
        client.post(
            BASE + "/query",
            json={"question": "سلام", "tenant_id": "another-tenant", "user_id": "admin"},
        ).status_code
        == 422
    )


def test_lms_http_conversation_is_bound_to_resolved_lms_user(lms_http):
    client, app = lms_http
    first = client.post(BASE + "/query", json={"question": "سلام"}).json()
    app.state.test_principal = context(user="another-learner")
    response = client.post(
        BASE + "/query", json={"question": "سلام", "conversation_id": first["conversation_id"]}
    )
    assert response.status_code == 404


def test_lms_factory_cannot_be_started_without_explicit_resolver(settings):
    with pytest.raises(ValueError, match="resolver"):
        create_lms_app(None, settings, FakeGateway())
    with pytest.raises(ValueError, match="create_lms_app"):
        create_app(settings, FakeGateway())


def test_async_lms_auth_dependency_and_401_errors_are_supported(settings):
    async def authenticated_by_lms():
        return context()

    with TestClient(create_lms_app(authenticated_by_lms, settings, FakeGateway())) as client:
        assert client.post(BASE + "/query", json={"question": "سلام"}).status_code == 200

    async def missing_session():
        raise HTTPException(status_code=401, detail="LMS session required")

    with TestClient(create_lms_app(missing_session, settings, FakeGateway())) as client:
        assert client.post(BASE + "/query", json={"question": "سلام"}).status_code == 401


def test_embedded_entrypoint_imports_without_jwt_or_fastapi():
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'jwt','fastapi'}:
            raise RuntimeError('Forbidden adapter import: ' + fullname)
sys.meta_path.insert(0, Block())
from hrlearnium.component import create_assistant
from hrlearnium.core_config import CoreSettings
settings=CoreSettings(_env_file=None, model_backend='literal')
assert 'jwt_secret' not in type(settings).model_fields
assert 'hrlearnium.adapters.jwt' not in sys.modules
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_lms_http_factory_does_not_import_jwt_in_a_fresh_process(tmp_path):
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] == 'jwt':
            raise RuntimeError('JWT adapter must not be loaded')
sys.meta_path.insert(0, Block())
from hrlearnium.core_config import CoreSettings
from hrlearnium.main import create_lms_app
app=create_lms_app(lambda: None, CoreSettings(_env_file=None, model_backend='literal', database_path=sys.argv[1]))
assert app.state.authentication_adapter == 'lms'
assert 'hrlearnium.adapters.jwt' not in sys.modules
app.state.service.close()
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "lms-http.db")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_operator_source_verification_does_not_require_jwt_secret(embedded, tmp_path):
    assistant, _, _ = embedded
    env = {
        **os.environ,
        "HR_JWT_SECRET": "",
        "HR_DATABASE_PATH": str(assistant.settings.database_path),
        "HR_MODEL_BACKEND": "literal",
        "HR_RETRIEVAL_MODE": "full_context",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hrlearnium",
            "verify-source",
            "--tenant",
            "tenant-1",
            "--course",
            COURSE,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified_passages"] > 0
