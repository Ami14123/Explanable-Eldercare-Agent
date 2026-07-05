# Shared FastAPI dependencies for request security.
from __future__ import annotations

from secrets import compare_digest

from fastapi import Header, HTTPException, status

from app.config import get_settings


# Protect admin-only endpoints with the configured token.
def require_admin_token(
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
) -> None:
    """Require an admin token for sensitive local operations."""

    # Read the token at request time so env changes are picked up after cache reset.
    configured_token = get_settings().admin_api_token.strip()
    # Refuse admin access when the project has not been configured safely.
    if not configured_token or configured_token == "change_me":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin access is not configured.",
        )

    # Prefer the explicit admin header, but also accept a bearer token.
    provided_token = (x_admin_token or "").strip()
    if not provided_token and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            provided_token = token.strip()

    # Missing credentials should prompt the caller to provide a token.
    if not provided_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin token is required.",
        )

    # Constant-time comparison avoids leaking token details through timing.
    if not compare_digest(provided_token, configured_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token.",
        )
