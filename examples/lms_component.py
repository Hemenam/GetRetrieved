"""Token-free service-layer example. Call ONLY after your LMS authorizes access.

This is not a login/enrollment implementation or a public endpoint. Do not obtain
the trusted arguments below from learner JSON or unsigned request headers.
"""

from hrlearnium.schemas import Principal, QueryRequest
from hrlearnium.service import CourseService


def ask_authorized_course(
    assistant: CourseService,
    *,
    authenticated_user_id: str,
    authorized_tenant_id: str,
    authorized_course_id: str,
    learner_payload: dict,
) -> dict:
    allowed = {"question", "conversation_id", "response_mode", "retrieval_mode"}
    if not isinstance(learner_payload, dict) or set(learner_payload) - allowed:
        raise ValueError("Unexpected learner request fields")
    principal = Principal(
        sub=authenticated_user_id,
        tenant_id=authorized_tenant_id,
        course_ids=[authorized_course_id],
        scopes=["query"],
    )
    request = QueryRequest.model_validate(learner_payload)
    return assistant.query(principal, authorized_course_id, request).model_dump(mode="json")
