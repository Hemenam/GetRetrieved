import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from hrlearnium.auth import require_scope
from hrlearnium.config import Settings
from hrlearnium.ingestion import IngestionError
from hrlearnium.models import ModelGateway, ModelUnavailable, create_gateway
from hrlearnium.reranking import Reranker
from hrlearnium.schemas import (
    DocumentResponse,
    Excerpt,
    Identifier,
    Principal,
    QueryRequest,
    QueryResponse,
)
from hrlearnium.service import ConversationNotFound, CourseService, ServiceBusy
from hrlearnium.storage import Storage, StorageConflict

logger = logging.getLogger("hrlearnium.requests")


class RequestLimitsMiddleware:
    """Enforce a streaming body limit before multipart parsing spools an upload."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            too_large = int(headers.get(b"content-length", b"0")) > self.limit
        except ValueError:
            too_large = True
        if too_large:
            await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                scope, receive, send
            )
            return
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.limit:
                    raise HTTPException(status_code=413, detail="Request body too large")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(
    settings: Settings | None = None,
    gateway: ModelGateway | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    settings = settings or Settings()
    storage = Storage(settings.database_path)
    gateway = gateway or create_gateway(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        gateway.close()

    app = FastAPI(
        title="HRLearnium Course Assistant API",
        version="0.3.0",
        lifespan=lifespan,
        description="Authenticated course Q&A with cited evidence. Natural questions can return "
        "a natural grounded explanation (default) or verbatim excerpts (response_mode). "
        "Social replies are labelled conversation and contain no course evidence. "
        "An LLM backend is required for natural-language answerability and explanations. "
        f"Configured backend: {settings.model_backend}. "
        + (
            "The literal backend is diagnostic only; configure a model backend for student Q&A."
            if settings.model_backend == "literal"
            else ""
        ),
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    app.state.settings, app.state.storage, app.state.gateway = settings, storage, gateway
    service = CourseService(settings, storage, gateway, reranker)
    app.state.service = service
    app.add_middleware(RequestLimitsMiddleware, limit=settings.max_upload_bytes + 64 * 1024)

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        request.state.request_id = str(uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        if request.url.path == "/chat" or request.url.path.startswith("/chat/assets/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self'; font-src 'self'; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            )
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-Frame-Options"] = "DENY"
        logger.info(
            "request_id=%s method=%s status=%s duration_ms=%d",
            request.state.request_id,
            request.method,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # FastAPI's default validation response can echo raw inputs; do not reflect source/secrets.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
                    for item in error.errors()
                ],
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(ModelUnavailable)
    async def model_error(request: Request, error: ModelUnavailable):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Evidence service unavailable; check model readiness or reindex the course",
                "request_id": request.state.request_id,
            },
            headers={"Retry-After": "10"},
        )

    @app.exception_handler(ServiceBusy)
    async def busy_error(request: Request, error: ServiceBusy):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Evidence service busy; retry shortly",
                "request_id": request.state.request_id,
            },
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(StorageConflict)
    async def conflict_error(request: Request, error: StorageConflict):
        return JSONResponse(
            status_code=409, content={"detail": str(error), "request_id": request.state.request_id}
        )

    @app.exception_handler(IngestionError)
    async def ingestion_error(request: Request, error: IngestionError):
        return JSONResponse(
            status_code=422, content={"detail": str(error), "request_id": request.state.request_id}
        )

    @app.exception_handler(ConversationNotFound)
    async def conversation_error(request: Request, error: ConversationNotFound):
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Conversation not found or expired",
                "request_id": request.state.request_id,
            },
        )

    def limit_request(principal: Principal):
        if not storage.allow_request(
            principal.tenant_id,
            principal.sub,
            settings.requests_per_minute,
            settings.tenant_requests_per_minute,
        ):
            raise HTTPException(
                status_code=429, detail="Request limit exceeded", headers={"Retry-After": "60"}
            )

    @app.get("/health/live", tags=["health"])
    def live():
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    def ready(response: Response):
        try:
            with storage.connection() as db:
                db.execute("SELECT 1")
            result = gateway.readiness()
            if service.reranker and hasattr(service.reranker, "readiness"):
                result["reranker"] = service.reranker.readiness()
                result["ready"] = bool(result.get("ready") and result["reranker"]["ready"])
            if not result.get("ready"):
                response.status_code = 503
            return result
        except ModelUnavailable:
            response.status_code = 503
            return {"ready": False, "backend": settings.model_backend}

    @app.post("/v1/courses/{course_id}/query", response_model=QueryResponse, tags=["query"])
    def query(
        course_id: Identifier,
        body: QueryRequest,
        request: Request,
        principal: Annotated[Principal, Depends(require_scope("query"))],
    ):
        limit_request(principal)
        return service.query(principal, course_id, body, request.state.request_id)

    async def process_upload(
        course: str, principal: Principal, file: UploadFile, document_id: str | None = None
    ):
        limit_request(principal)
        filename = PurePosixPath((file.filename or "course.docx").replace("\\", "/")).name
        if not filename.lower().endswith(".docx") or len(filename) > 255:
            raise HTTPException(status_code=415, detail="Only DOCX files are supported")
        data = await file.read(settings.max_upload_bytes + 1)
        await file.close()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Document exceeds upload limit")
        try:
            document, changed = await run_in_threadpool(
                service.ingest, principal, course, filename, data, document_id
            )
        except LookupError:
            raise HTTPException(status_code=404, detail="Document not found") from None
        return document, changed

    @app.post(
        "/v1/courses/{course_id}/documents",
        response_model=DocumentResponse,
        status_code=201,
        tags=["content"],
    )
    async def upload_document(
        course_id: Identifier,
        response: Response,
        principal: Annotated[Principal, Depends(require_scope("content:write"))],
        file: Annotated[UploadFile, File()],
    ):
        document, changed = await process_upload(course_id, principal, file)
        if not changed:
            response.status_code = 200
        return document

    @app.get(
        "/v1/courses/{course_id}/documents", response_model=list[DocumentResponse], tags=["content"]
    )
    def list_documents(
        course_id: Identifier,
        principal: Annotated[Principal, Depends(require_scope("content:read"))],
    ):
        return storage.list_documents(principal.tenant_id, course_id)

    @app.put(
        "/v1/courses/{course_id}/documents/{document_id}",
        response_model=DocumentResponse,
        tags=["content"],
    )
    async def replace_document(
        course_id: Identifier,
        document_id: str,
        principal: Annotated[Principal, Depends(require_scope("content:write"))],
        file: Annotated[UploadFile, File()],
    ):
        document, _ = await process_upload(course_id, principal, file, document_id)
        return document

    @app.delete(
        "/v1/courses/{course_id}/documents/{document_id}", status_code=204, tags=["content"]
    )
    def retire_document(
        course_id: Identifier,
        document_id: str,
        principal: Annotated[Principal, Depends(require_scope("content:write"))],
    ):
        limit_request(principal)
        if not storage.retire_document(principal.tenant_id, course_id, document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return Response(status_code=204)

    @app.get(
        "/v1/courses/{course_id}/excerpts/{excerpt_id}", response_model=Excerpt, tags=["query"]
    )
    def get_excerpt(
        course_id: Identifier,
        excerpt_id: str,
        principal: Annotated[Principal, Depends(require_scope("query"))],
    ):
        excerpts = storage.get_excerpts(principal.tenant_id, course_id, [excerpt_id])
        if not excerpts:
            raise HTTPException(status_code=404, detail="Excerpt not found")
        return excerpts[0]

    if settings.enable_chat_ui:
        web_directory = Path(__file__).parent / "web"
        app.mount(
            "/chat/assets", StaticFiles(directory=web_directory / "assets"), name="chat-assets"
        )

        @app.get("/", include_in_schema=False)
        def chat_redirect():
            return RedirectResponse("/chat")

        @app.get("/chat", include_in_schema=False)
        def chat_page():
            # Only the static shell is public. All course queries still require a service JWT.
            return FileResponse(web_directory / "index.html", media_type="text/html")

    return app
