from __future__ import annotations

from copy import deepcopy
from typing import Any


SECRET_KEY_FRAGMENTS = [
    "api_key",
    "authorization",
    "password",
    "token",
    "secret",
    "credential",
    "openrouter_api_key",
]

HIDDEN_PROMPT_KEYS = [
    "system_prompt",
    "hidden_prompt",
    "full_prompt",
    "messages",
    "headers",
]


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in SECRET_KEY_FRAGMENTS + HIDDEN_PROMPT_KEYS)


def sanitize_developer_state(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                clean[key] = "[redacted]"
            else:
                clean[key] = sanitize_developer_state(item)
        return clean
    if isinstance(value, list):
        return [sanitize_developer_state(item) for item in value]
    if isinstance(value, str):
        if "sk-or-" in value or "Bearer " in value:
            return "[redacted]"
        return value
    return deepcopy(value)


def family_safe_response(response: dict[str, Any]) -> dict[str, Any]:
    safe = {
        "final_message": response.get("final_message", ""),
        "overall_risk": response.get("overall_risk", "low"),
        "main_concern": response.get("main_concern", ""),
        "caregiver_alert": response.get("caregiver_alert", ""),
        "next_step": response.get("next_step", ""),
        "activated_agents": response.get("activated_agents", []),
        "routing_explanation": response.get("routing_explanation", ""),
        "detected_signals": response.get("detected_signals", []),
        "graph_reasoning": response.get("graph_reasoning", {}),
        "care_plan": response.get("care_plan", {}),
        "developer_state": sanitize_developer_state(response.get("developer_state", {})),
    }
    return safe
