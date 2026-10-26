"""Compatibility import for the standalone JWT adapter, not the chatbot core."""

from hrlearnium.adapters.http import require_scope as _require_scope
from hrlearnium.adapters.jwt import authenticated


def require_scope(scope: str):
    return _require_scope(scope, authenticated)
