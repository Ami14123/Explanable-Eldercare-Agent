from __future__ import annotations

from secrets import compare_digest

from fastapi import Header, HTTPException, status

from app.config import get_settings


def require_admin_token(
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
) -> None:
    """Require an admin token for sensitive local operations."""

    configured_token = get_settings().admin_api_token.strip()
    if not configured_token or configured_token == "change_me":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin access is not configured.",
        )

    provided_token = (x_admin_token or "").strip()
    if not provided_token and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            provided_token = token.strip()

    if not provided_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin token is required.",
        )

    if not compare_digest(provided_token, configured_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token.",
        )
