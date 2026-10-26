"""HTTP adapter for an injected LMS authentication dependency; no JWT imports."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException

from hrlearnium.access import require_access
from hrlearnium.schemas import Principal


def require_scope(scope: str, principal_resolver: Callable):
    def authorized(course_id: str, principal: Annotated[Principal, Depends(principal_resolver)]):
        if not isinstance(principal, Principal):
            raise HTTPException(
                status_code=401, detail="LMS authentication context is missing or invalid"
            )
        return require_access(principal, course_id, scope)

    return authorized
