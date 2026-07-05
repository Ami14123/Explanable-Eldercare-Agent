# Input gateway that sanitizes public chat requests before graph execution.
from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4

from fastapi import HTTPException, status

from app.schemas import ChatRequest


# Limit message and id sizes to protect the local prototype from huge inputs.
MAX_MESSAGE_LENGTH = 2_000
MAX_ID_LENGTH = 120


# Bundle a sanitized request with a request id for technical tracing.
@dataclass(frozen=True)
class GatewayRequest:
    """Validated gateway request passed into the LangGraph workflow."""

    request_id: str
    chat_request: ChatRequest


# Remove control characters and normalize whitespace in free text.
def _clean_text(value: str) -> str:
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)
    return re.sub(r"\s+", " ", normalized).strip()


# Clean required identifiers and reject missing or oversized values.
def _clean_identifier(value: str, field_name: str) -> str:
    cleaned = _clean_text(value)
    # Empty ids break memory grouping, so fail with a clear client error.
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} is required.",
        )
    # Cap ids so logs and database rows stay bounded.
    if len(cleaned) > MAX_ID_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} is too long.",
        )
    return cleaned


# Validate one chat request and return a sanitized copy for the workflow.
def validate_chat_request(request: ChatRequest) -> GatewayRequest:
    """
    Validate and sanitize the public chat request.

    Vietnamese note: Day la lop gateway mong truoc LangGraph, dung de kiem tra
    input co ban va tao request_id cho technical trace.
    """

    # Message text is required because the graph routes from it.
    message = _clean_text(request.message)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="message is required.",
        )
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"message must be {MAX_MESSAGE_LENGTH} characters or fewer.",
        )

    # User and conversation ids control memory scope.
    user_id = _clean_identifier(request.user_id, "user_id")
    conversation_id = _clean_identifier(request.conversation_id, "conversation_id")

    # Keep only the most recent client history and accepted chat roles.
    sanitized_history = []
    for item in request.client_history[-10:]:
        role = _clean_text(str(item.get("role", "")))
        text = _clean_text(str(item.get("message", item.get("content", ""))))
        if role in {"user", "assistant"} and text:
            sanitized_history.append({"role": role, "message": text[:MAX_MESSAGE_LENGTH]})

    # Create a fresh request id so traces can connect logs to this turn.
    return GatewayRequest(
        request_id=str(uuid4()),
        chat_request=ChatRequest(
            message=message,
            user_id=user_id,
            conversation_id=conversation_id,
            client_history=sanitized_history,
        ),
    )
