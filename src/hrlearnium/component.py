"""Embedded LMS entry point. No JWT, FastAPI, login or token-generation dependency."""

from hrlearnium.core_config import CoreSettings
from hrlearnium.models import ModelGateway, create_gateway
from hrlearnium.reranking import Reranker
from hrlearnium.service import CourseService
from hrlearnium.storage import Storage


def create_assistant(
    settings: CoreSettings | None = None,
    *,
    gateway: ModelGateway | None = None,
    reranker: Reranker | None = None,
) -> CourseService:
    """Create once per LMS worker. Caller supplies trusted Principal context per operation.

    This component owns the supplied/created gateway's lifecycle: call close() at
    worker shutdown. LMS owns login, enrollment checks, CSRF, HTTP limits and UI.
    Authorization here is tenant/course/action-level, not document/section-level.
    """
    settings = settings or CoreSettings()
    storage = Storage(settings.database_path)
    return CourseService(settings, storage, gateway or create_gateway(settings), reranker)
