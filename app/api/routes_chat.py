from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.errors import build_safe_chat_error_response
from app.gateway import validate_chat_request
from app.graph import run_elderguard_workflow
from app.schemas import ChatRequest, ChatResponse


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one ElderGuard chat turn."""

    try:
        # Vietnamese note: API entry point -> gateway validation -> LangGraph workflow.
        gateway_request = validate_chat_request(request)
        response = run_elderguard_workflow(gateway_request.chat_request)
        if not response.final_message:
            response.final_message = (
                "I'm here to help. Could you tell me more about what you need?"
            )
        response.developer_state["request_id"] = gateway_request.request_id
        return response
    except HTTPException:
        raise
    except Exception:
        return build_safe_chat_error_response()
