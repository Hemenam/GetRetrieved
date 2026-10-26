"""Explicit loopback-only testing adapter. Never use as production LMS auth."""

from ipaddress import ip_address

from fastapi import Request
from fastapi.responses import JSONResponse

from hrlearnium.core_config import CoreSettings
from hrlearnium.main import create_lms_app, mount_chat_ui
from hrlearnium.schemas import Principal


def create_local_chat_app(*, tenant: str, course: str, port=8000, settings=None, gateway=None):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Local chat port must be between 1 and 65535")
    # Values are validated IDs and operator-supplied, never resolved from browser data.
    principal = Principal(
        sub="local-test-operator", tenant_id=tenant, course_ids=[course], scopes=["query"]
    )

    def local_principal():
        return principal.model_copy(deep=True)

    app = create_lms_app(local_principal, settings or CoreSettings(), gateway=gateway)
    app.state.authentication_adapter = "local_test_only"
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    if port == 80:
        allowed_hosts.update({"127.0.0.1", "localhost", "[::1]"})

    @app.middleware("http")
    async def loopback_boundary(request: Request, call_next):
        try:
            local_peer = request.client is not None and ip_address(request.client.host).is_loopback
        except ValueError:
            local_peer = False
        host = request.headers.get("host", "").lower()
        origin = request.headers.get("origin")
        foreign_origin = origin is not None and origin.lower() != f"http://{host}"
        foreign_fetch = request.headers.get("sec-fetch-site") == "cross-site" and not (
            request.method == "GET" and request.url.path in {"/", "/chat"}
        )
        if not local_peer or host not in allowed_hosts or foreign_origin or foreign_fetch:
            return JSONResponse({"detail": "Local testing requests only"}, status_code=403)
        return await call_next(request)

    mount_chat_ui(app, local_course=course)
    return app
