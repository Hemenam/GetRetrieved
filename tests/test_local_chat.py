"""Token-free local UI does not disable auth in standalone or LMS deployments."""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_api import BASE, COURSE, FakeGateway, docx

from hrlearnium.adapters.local_chat import create_local_chat_app
from hrlearnium.core_config import CoreSettings
from hrlearnium.schemas import Principal


@pytest.fixture
def local(tmp_path):
    settings = CoreSettings(
        _env_file=None,
        database_path=tmp_path / "local.db",
        model_backend="ollama",
        retrieval_mode="full_context",
    )
    model = FakeGateway()
    app = create_local_chat_app(tenant="tenant-1", course=COURSE, settings=settings, gateway=model)
    admin = Principal(
        sub="setup", tenant_id="tenant-1", course_ids=[COURSE], scopes=["content:write"]
    )
    app.state.service.ingest(admin, COURSE, "test.docx", docx())
    with TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)) as client:
        yield client, app, model


def test_local_page_bootstraps_fixed_course_and_query_without_token(local):
    client, app, model = local
    page = client.get("/chat")
    assert page.status_code == 200 and f'data-local-course="{COURSE}"' in page.text
    assert 'id="connection-button" hidden' in page.text
    assert 'id="connection-dialog" hidden' in page.text
    assert "Local testing · no JWT required" in page.text
    assert "connect-src 'self'" in page.headers["content-security-policy"]
    assert app.state.authentication_adapter == "local_test_only"
    result = client.post(BASE + "/query", json={"question": "سلام"})
    assert result.status_code == 200 and result.json()["status"] == "conversation"
    assert client.get(BASE + "/retrieval-options").status_code == 200
    assert client.get(BASE + "/outline").status_code == 200
    assert model.selections == []


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "evil.example:8000"},
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_foreign_hosts_origins_and_cross_site_fetches_are_rejected(local, headers):
    client, _, model = local
    assert (
        client.post(BASE + "/query", headers=headers, json={"question": "سلام"}).status_code == 403
    )
    assert model.selections == []


def test_same_origin_works_and_browser_identity_cannot_grant_other_course_or_writes(local):
    client, _, model = local
    assert (
        client.post(
            BASE + "/query", headers={"Origin": "http://127.0.0.1:8000"}, json={"question": "سلام"}
        ).status_code
        == 200
    )
    forged = {"Authorization": "Bearer forged", "X-Tenant-ID": "other", "X-Course-ID": "other"}
    assert client.get("/v1/courses/other/outline", headers=forged).status_code == 403
    assert client.get(BASE + "/documents", headers=forged).status_code == 403
    assert client.post(BASE + "/documents", files={"file": ("new.docx", docx())}).status_code == 403
    assert client.delete(BASE + "/documents/not-a-document").status_code == 403
    assert model.selections == []


def test_non_loopback_peer_cannot_use_local_adapter(local):
    _, app, model = local
    with TestClient(app, base_url="http://127.0.0.1:8000", client=("192.0.2.10", 50000)) as client:
        assert client.get("/chat").status_code == 403
        assert client.post(BASE + "/query", json={"question": "سلام"}).status_code == 403
    assert model.selections == []


def test_dev_chat_cli_cannot_bind_publicly_or_trust_forwarded_headers(monkeypatch):
    from hrlearnium import cli

    calls = []
    app = object()
    monkeypatch.setattr("hrlearnium.adapters.local_chat.create_local_chat_app", lambda **kw: app)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(
        sys, "argv", ["hrlearnium", "dev-chat", "--tenant", "tenant-1", "--course", COURSE]
    )
    assert cli.main() == 0
    assert calls == [((app,), {"host": "127.0.0.1", "port": 8000, "proxy_headers": False})]
    monkeypatch.setattr(
        sys,
        "argv",
        ["hrlearnium", "dev-chat", "--tenant", "tenant-1", "--course", COURSE, "--host", "0.0.0.0"],
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_local_factory_imports_no_jwt_and_needs_no_signing_secret(tmp_path):
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] == 'jwt':
            raise RuntimeError('JWT must not load')
sys.meta_path.insert(0, Block())
from hrlearnium.adapters.local_chat import create_local_chat_app
from hrlearnium.core_config import CoreSettings
app = create_local_chat_app(tenant='demo', course='demo', settings=CoreSettings(_env_file=None, model_backend='literal', database_path=sys.argv[1]))
assert app.state.authentication_adapter == 'local_test_only'
assert 'hrlearnium.adapters.jwt' not in sys.modules
app.state.service.close()
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "no-jwt.db")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
