"""Transport-independent checks on context supplied by the trusted LMS backend."""

from hrlearnium.schemas import Principal


class AccessDenied(PermissionError):
    """Identity is known but the requested course/action is not authorized."""


def require_access(principal: Principal, course: str, scope: str) -> Principal:
    if (
        not isinstance(principal, Principal)
        or course not in principal.course_ids
        or scope not in principal.scopes
    ):
        raise AccessDenied("Course or action is not authorized")
    return principal
