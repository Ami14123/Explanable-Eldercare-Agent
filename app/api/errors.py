from __future__ import annotations

import traceback
from pathlib import Path

from app.schemas import ChatResponse


def build_safe_chat_error_response() -> ChatResponse:
    """
    Store the traceback locally and return a sanitized chat error.

    The API response deliberately avoids local paths and raw tracebacks.
    """

    Path("data").mkdir(exist_ok=True)
    Path("data/last_chat_error.txt").write_text(
        traceback.format_exc(),
        encoding="utf-8",
    )

    return ChatResponse(
        final_message="I encountered an issue. Please try again.",
        active_mode="Error",
        final_message_source="error_handler",
        current_topic="",
        is_follow_up=False,
        known_information=[],
        follow_up_questions=[],
        specificity_score=0,
        missing_information=[],
        hypotheses=[],
        need_more_information=False,
        reasoning_summary="Chat endpoint error",
        xai_simple="An error occurred",
        caregiver_summary="",
        activated_agents=[],
        overall_risk="medium",
        main_concern="System error",
        advice="Try again",
        caregiver_alert="",
        next_step="Retry",
        memory_id=0,
        human_confirmation_required=False,
        confirmation_reason="",
        routing_explanation="",
        graph_reasoning={},
        detected_signals=[],
        risk_scores={},
        memory_summary={},
        situation={},
        care_plan={},
        developer_state={"error": "chat_failed"},
    )
