"""The public UI shell must not weaken authenticated course access."""

import pytest
from test_api import api_factory as api_factory


def test_chat_ui_and_assets_do_not_call_models_or_expose_configuration(api_factory):
    client, _, gateway = api_factory()
    page = client.get("/chat")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert 'charset="utf-8"' in page.text
    assert "مدل ۳۰-۳۰-۳۰" in page.text
    assert 'src="/chat/assets/app.mjs"' in page.text
    assert page.headers["cache-control"] == "no-store"
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["x-frame-options"] == "DENY"
    assert "connect-src 'self'" in page.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert page.headers["referrer-policy"] == "no-referrer"
    assert "test-secret-at-least" not in page.text
    assert client.get("/", follow_redirects=False).headers["location"] == "/chat"
    for asset in ["app.mjs", "styles.css"]:
        response = client.get("/chat/assets/" + asset)
        assert response.status_code == 200
        assert "text/html" not in response.headers["content-type"]
        assert "test-secret-at-least" not in response.text
    assert gateway.selections == []
    assert "/chat" not in client.get("/openapi.json").json()["paths"]


def test_chat_ui_does_not_remove_query_authentication(api_factory):
    client, _, gateway = api_factory()
    assert client.get("/chat").status_code == 200
    response = client.post("/v1/courses/course-1/query", json={"question": "What is this?"})
    assert response.status_code == 401
    assert gateway.selections == []


@pytest.mark.parametrize("path", ["/", "/chat", "/chat/assets/app.mjs"])
def test_chat_ui_can_be_disabled_independently_of_api_docs(api_factory, path):
    client, _, _ = api_factory(enable_chat_ui=False)
    assert client.get(path).status_code == 404
    assert client.get("/docs").status_code == 200


def test_chat_assets_cannot_serve_settings_or_other_package_files(api_factory):
    client, _, _ = api_factory()
    for path in ["/chat/assets/.env", "/chat/assets/%2e%2e/%2e%2e/config.py"]:
        assert client.get(path).status_code == 404
