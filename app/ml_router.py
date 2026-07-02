from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib


VALID_MODEL_ROUTES = {
    "fraud_agent",
    "mental_care_agent",
    "daily_care_agent",
}


MODEL_TO_GRAPH_ROUTE = {
    "fraud_agent": "safety_agent",
    "mental_care_agent": "emotional_social_agent",
    "daily_care_agent": "health_daily_care_agent",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "router_best_model.joblib"


class MLRouterError(RuntimeError):
    """Raised when the ML router cannot make a valid prediction."""


def get_model_path() -> Path:
    configured_path = os.getenv("ML_ROUTER_MODEL_PATH", "").strip()
    if not configured_path:
        return DEFAULT_MODEL_PATH

    model_path = Path(configured_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path.resolve()


@lru_cache(maxsize=1)
def load_ml_router():
    """Load the trusted local router model once and cache it."""

    model_path = get_model_path()
    if not model_path.exists():
        raise MLRouterError(f"Router model was not found: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as error:
        raise MLRouterError(f"Cannot load router model: {error}") from error

    if not hasattr(model, "predict"):
        raise MLRouterError("Loaded object does not have a predict method.")
    return model


def predict_route(text: str) -> str:
    """Predict one existing graph specialist from a user message."""

    cleaned_text = str(text).strip()
    if not cleaned_text:
        raise MLRouterError("Cannot route an empty user message.")

    model = load_ml_router()
    try:
        prediction = model.predict([cleaned_text])
    except Exception as error:
        raise MLRouterError(f"Model prediction failed: {error}") from error

    if len(prediction) == 0:
        raise MLRouterError("Model returned no prediction.")

    model_route = str(prediction[0]).strip()
    if model_route not in VALID_MODEL_ROUTES:
        raise MLRouterError(f"Model returned an unsupported route: {model_route}")
    return MODEL_TO_GRAPH_ROUTE[model_route]


def extract_latest_user_text(state: Mapping[str, Any]) -> str:
    """Find the latest user text from common LangGraph state structures."""

    request = state.get("request")
    request_message = getattr(request, "message", "")
    if isinstance(request_message, str) and request_message.strip():
        return request_message.strip()

    direct_text_keys = (
        "user_message",
        "latest_user_message",
        "message",
        "guest_message",
        "input",
        "text",
        "query",
    )
    for key in direct_text_keys:
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = state.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, str) and message.strip():
                return message.strip()

            if isinstance(message, dict):
                role = str(message.get("role", message.get("type", ""))).lower()
                content = message.get("content", message.get("text", ""))
                if role in {"user", "human"} and isinstance(content, str) and content.strip():
                    return content.strip()

            role = str(getattr(message, "type", getattr(message, "role", ""))).lower()
            content = getattr(message, "content", None)
            if role in {"user", "human"} and isinstance(content, str) and content.strip():
                return content.strip()

        if messages:
            latest_message = messages[-1]
            if isinstance(latest_message, dict):
                content = latest_message.get("content", latest_message.get("text", ""))
            else:
                content = getattr(latest_message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()

    raise MLRouterError("Cannot find the latest user message in state.")
