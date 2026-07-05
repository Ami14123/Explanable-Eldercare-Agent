# LLM-mode tests that verify mock/live behavior without real network calls.
import os
import sys
import types
from pathlib import Path

from app.config import get_settings


# Track fake OpenRouter calls and configured fake behavior.
CALLS = {"count": 0, "raise": None, "content": '{"final_message":"LIVE RESPONSE","xai_simple":"live","follow_up_questions":[]}'}


# Fake OpenAI choice object matching the field the gateway reads.
class FakeChoice:
    def __init__(self, content: str):
        self.message = types.SimpleNamespace(content=content)


# Fake chat completions endpoint that records each call.
class FakeCompletions:
    def create(self, **kwargs):
        # Count calls so tests can enforce one live request per turn.
        CALLS["count"] += 1
        if CALLS["raise"]:
            raise RuntimeError(str(CALLS["raise"]))
        return types.SimpleNamespace(choices=[FakeChoice(str(CALLS["content"]))])


# Fake OpenAI client injected into sys.modules.
class FakeOpenAI:
    def __init__(self, **kwargs):
        self.chat = types.SimpleNamespace(completions=FakeCompletions())


# Install the fake OpenAI module before live-mode workflow calls.
def install_fake_openai() -> None:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=FakeOpenAI)


# Reset env vars and fake call counters between test sections.
def reset_env(mode: str, key: str = "", model: str = "openai/gpt-4o-mini") -> None:
    os.environ["LLM_MODE"] = mode
    os.environ["OPENROUTER_API_KEY"] = key
    os.environ["OPENROUTER_MODEL"] = model
    os.environ["GEMINI_API_KEY"] = "ignored"
    os.environ["GEMINI_MODEL"] = "ignored"
    get_settings.cache_clear()
    CALLS["count"] = 0
    CALLS["raise"] = None
    CALLS["content"] = '{"final_message":"LIVE RESPONSE","xai_simple":"live","follow_up_questions":[]}'


# Small assertion helper keeps standalone failures readable.
def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


# Run one workflow turn after importing graph dependencies.
def workflow_response(message: str):
    from app.memory import init_db
    from app.graph import run_elderguard_workflow
    from app.schemas import ChatRequest

    # Initialize DB so the workflow can persist memory during the test.
    init_db()
    return run_elderguard_workflow(ChatRequest(message=message, user_id="llm_mode_test", conversation_id=message[:20]))


# Run all mock/live mode checks.
def main() -> None:
    install_fake_openai()

    # Mock mode should never call OpenRouter.
    reset_env("mock")
    response = workflow_response("I am hungry")
    assert_equal(CALLS["count"], 0, "mock makes zero OpenRouter calls")
    assert_equal(response.developer_state["llm_mode"], "mock", "mock mode recorded")
    assert_equal(response.developer_state["llm_provider"], "none", "mock provider recorded")
    assert_equal(response.developer_state["llm_model"], "none", "mock model recorded")

    # Live mode should call OpenRouter exactly once and record metadata.
    reset_env("live", key="test-key")
    response = workflow_response("I am hungry")
    assert_equal(CALLS["count"], 1, "live makes exactly one OpenRouter call")
    assert_equal(response.final_message, "LIVE RESPONSE", "live uses OpenRouter output")
    assert_equal(response.active_mode, "live", "live mode active")
    assert_equal(response.developer_state["llm_calls_this_turn"], 1, "live call count recorded")
    assert_equal(response.developer_state["llm_provider"], "openrouter", "live provider recorded")
    assert_equal(response.developer_state["llm_model"], "openai/gpt-4o-mini", "live model recorded")
    assert_equal(response.developer_state["llm_caller"], "health_daily_care_agent", "llm caller is selected agent")
    assert "LIVE RESPONSE" in response.developer_state["raw_llm_output"]
    assert_equal(response.developer_state["response_before_guardrail"], "LIVE RESPONSE", "before guardrail recorded")
    assert_equal(response.developer_state["response_after_guardrail"], "LIVE RESPONSE", "after guardrail recorded")
    assert_equal(response.developer_state["final_message_source"], "openrouter_specialist", "source recorded")
    assert response.developer_state["prompt_name"].endswith("_prompt")
    assert_equal(response.developer_state["prompt_version"], "v2", "prompt version recorded")

    # Missing live-mode key should fail configuration validation.
    reset_env("live", key="")
    try:
        get_settings()
    except RuntimeError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("missing OpenRouter key should raise configuration error")

    # Missing live-mode model should also fail validation.
    reset_env("live", key="test-key", model="")
    try:
        get_settings()
    except RuntimeError as exc:
        assert "OPENROUTER_MODEL" in str(exc)
    else:
        raise AssertionError("missing OpenRouter model should raise configuration error")

    # Runtime OpenRouter errors should return an LLMError response.
    reset_env("live", key="test-key")
    CALLS["raise"] = "OpenRouter down"
    response = workflow_response("I am hungry")
    assert_equal(CALLS["count"], 1, "runtime failure still made one OpenRouter call")
    assert_equal(response.active_mode, "LLMError", "runtime failure does not activate mock")
    assert_equal(response.developer_state["llm_error"]["raw_error"], "OpenRouter down", "runtime error recorded")

    # Gemini env vars are ignored because OpenRouter is the only live path.
    reset_env("mock")
    os.environ["GEMINI_API_KEY"] = "should-not-matter"
    os.environ["GEMINI_MODEL"] = "should-not-matter"
    get_settings.cache_clear()
    response = workflow_response("I am hungry")
    assert_equal(response.developer_state["llm_mode"], "mock", "Gemini env ignored")
    assert_equal(CALLS["count"], 0, "Gemini env does not trigger OpenRouter")

    # Source scan guards against reintroducing Gemini dependencies.
    app_text = "\n".join(path.read_text(encoding="utf-8") for path in Path("app").glob("*.py")).lower()
    assert "google.generative" not in app_text
    assert "gemini" not in app_text

    print("All LLM mode tests passed.")


# Allow direct command-line execution.
if __name__ == "__main__":
    main()
