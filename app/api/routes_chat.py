# Public chat route that validates input and runs the ElderGuard workflow.
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.errors import build_safe_chat_error_response
from app.gateway import validate_chat_request
from app.graph import run_elderguard_workflow
from app.schemas import ChatRequest, ChatResponse


# Keep chat endpoints on their own router for main app registration.
router = APIRouter()


# Run one chat turn and return the structured explainable response.
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one ElderGuard chat turn."""

    try:
        # Vietnamese note: API entry point -> gateway validation -> LangGraph workflow.
        # Validate request metadata before the graph touches the message.
        gateway_request = validate_chat_request(request)
        # Execute the full reasoning workflow with the sanitized request.
        response = run_elderguard_workflow(gateway_request.chat_request)
        # Guarantee a friendly fallback if the graph returns no final text.
        if not response.final_message:
            response.final_message = (
                "I'm here to help. Could you tell me more about what you need?"
            )
        # Attach request id for debugging without changing the public schema.
        response.developer_state["request_id"] = gateway_request.request_id
        return response
    except HTTPException:
        # Preserve intentional HTTP errors such as validation failures.
        raise
    except Exception:
        # Convert unexpected failures into the standard safe chat response.
        return build_safe_chat_error_response()
