import time
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from hrlearnium.schemas import Principal

bearer = HTTPBearer(auto_error=False)


def authenticated(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    unauthorized = HTTPException(
        status_code=401,
        detail="Invalid or expired service token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or len(credentials.credentials) > 12000:
        raise unauthorized
    settings = request.app.state.settings
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={
                "require": [
                    "exp",
                    "iat",
                    "iss",
                    "aud",
                    "sub",
                    "tenant_id",
                    "course_ids",
                    "scopes",
                    "jti",
                ]
            },
            leeway=5,
        )
        if (
            type(payload["exp"]) is not int
            or type(payload["iat"]) is not int
            or payload["exp"] <= payload["iat"]
            or payload["exp"] - payload["iat"] > settings.max_token_lifetime_seconds
            or payload["iat"] > time.time() + 5
            or not isinstance(payload["jti"], str)
            or not 1 <= len(payload["jti"]) <= 200
        ):
            raise unauthorized
        return Principal.model_validate(
            {key: payload[key] for key in ("sub", "tenant_id", "course_ids", "scopes")}
        )
    except (jwt.PyJWTError, ValidationError, KeyError, TypeError, ValueError):
        raise unauthorized from None


def require_scope(scope: str):
    def authorized(
        course_id: str, principal: Annotated[Principal, Depends(authenticated)]
    ) -> Principal:
        if course_id not in principal.course_ids or scope not in principal.scopes:
            raise HTTPException(status_code=403, detail="Course or action is not authorized")
        return principal

    return authorized
