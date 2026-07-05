# Shared LLM gateway helpers used by the agent graph.
from __future__ import annotations

from contextvars import ContextVar

from app.config import get_settings

# Track calls per request so tests and traces can explain LLM usage.
_LLM_CALLS_THIS_TURN: ContextVar[int] = ContextVar("llm_calls_this_turn", default=0)


# Reset the per-turn counter before a new graph run starts.
def reset_llm_call_counter() -> None:
    _LLM_CALLS_THIS_TURN.set(0)


# Read the current counter so diagnostics can report live LLM calls.
def get_llm_call_counter() -> int:
    return _LLM_CALLS_THIS_TURN.get()


# Send one chat completion request through OpenRouter when live mode is enabled.
def call_openrouter_llm(messages: list[dict[str, str]], temperature: float = 0.4) -> str:
    """Shared OpenRouter gateway. This is the only live LLM call path."""
    # Load settings here so tests can switch mock/live mode through env vars.
    settings = get_settings()
    settings.validate()
    # Block accidental network calls while the app is in mock mode.
    if settings.llm_mode == "mock":
        raise RuntimeError("LLM_MODE=mock does not call OpenRouter.")

    # Count this call for the explainability trace.
    _LLM_CALLS_THIS_TURN.set(_LLM_CALLS_THIS_TURN.get() + 1)

    # Import lazily so mock mode does not require constructing a client.
    from openai import OpenAI

    # Use the OpenAI-compatible OpenRouter endpoint configured by settings.
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )
    # Ask the selected model for one response using the provided messages.
    response = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    # Return an empty string if the provider sends no text.
    return response.choices[0].message.content or ""
