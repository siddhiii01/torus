"""Bearer-token authentication. Fails closed when no token is configured."""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from torus.service.errors import AuthNotConfiguredError
from torus.service.settings import TOKEN_ENV

_scheme = HTTPBearer(auto_error=False)


def load_token(env: dict[str, str] | None = None) -> str:
    token = (env if env is not None else os.environ).get(TOKEN_ENV, "").strip()
    if len(token) < 16:
        raise AuthNotConfiguredError(f"{TOKEN_ENV} must be set (at least 16 chars)")
    return token


def make_auth_dependency(token: str) -> Callable[..., None]:
    def require_token(
        creds: HTTPAuthorizationCredentials | None = Depends(_scheme),  # noqa: B008
    ) -> None:
        supplied = creds.credentials if creds else ""
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "invalid or missing token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_token
