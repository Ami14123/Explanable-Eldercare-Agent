from __future__ import annotations

from contextvars import ContextVar

from app.config import get_settings

_LLM_CALLS_THIS_TURN: ContextVar[int] = ContextVar("llm_calls_this_turn", default=0)


def reset_llm_call_counter() -> None:
    _LLM_CALLS_THIS_TURN.set(0)


def get_llm_call_counter() -> int:
    return _LLM_CALLS_THIS_TURN.get()


def call_openrouter_llm(messages: list[dict[str, str]], temperature: float = 0.4) -> str:
    """Shared OpenRouter gateway. This is the only live LLM call path."""
    settings = get_settings()
    settings.validate()
    if settings.llm_mode == "mock":
        raise RuntimeError("LLM_MODE=mock does not call OpenRouter.")

    _LLM_CALLS_THIS_TURN.set(_LLM_CALLS_THIS_TURN.get() + 1)

    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )
    response = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
