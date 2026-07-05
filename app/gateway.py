from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4

from fastapi import HTTPException, status

from app.schemas import ChatRequest


MAX_MESSAGE_LENGTH = 2_000
MAX_ID_LENGTH = 120


@dataclass(frozen=True)
class GatewayRequest:
    """Validated gateway request passed into the LangGraph workflow."""

    request_id: str
    chat_request: ChatRequest


def _clean_text(value: str) -> str:
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_identifier(value: str, field_name: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} is required.",
        )
    if len(cleaned) > MAX_ID_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} is too long.",
        )
    return cleaned


def validate_chat_request(request: ChatRequest) -> GatewayRequest:
    """
    Validate and sanitize the public chat request.

    Vietnamese note: Day la lop gateway mong truoc LangGraph, dung de kiem tra
    input co ban va tao request_id cho technical trace.
    """

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

    user_id = _clean_identifier(request.user_id, "user_id")
    conversation_id = _clean_identifier(request.conversation_id, "conversation_id")

    sanitized_history = []
    for item in request.client_history[-10:]:
        role = _clean_text(str(item.get("role", "")))
        text = _clean_text(str(item.get("message", item.get("content", ""))))
        if role in {"user", "assistant"} and text:
            sanitized_history.append({"role": role, "message": text[:MAX_MESSAGE_LENGTH]})

    return GatewayRequest(
        request_id=str(uuid4()),
        chat_request=ChatRequest(
            message=message,
            user_id=user_id,
            conversation_id=conversation_id,
            client_history=sanitized_history,
        ),
    )
