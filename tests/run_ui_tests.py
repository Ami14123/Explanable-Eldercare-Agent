from __future__ import annotations

from pathlib import Path

from app.ui.family_view import build_family_explanation
from app.ui.sanitization import sanitize_developer_state


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sample_response() -> dict:
    return {
        "final_message": "Please sit down and ask someone nearby to check on you.",
        "overall_risk": "high",
        "main_concern": "health_daily_care",
        "caregiver_alert": "Please check on the elder immediately.",
        "next_step": "Contact: caregiver or nearby trusted person",
        "activated_agents": ["health_daily_care_agent"],
        "routing_explanation": "Local router selected broad agents from latest user message.",
        "graph_reasoning": {
            "detected_nodes": ["dizziness", "near_fall"],
            "reasoning_path": [["dizziness", "fall_risk", "red_flag_questions"]],
        },
        "care_plan": {},
        "developer_state": {
            "router_decision": {
                "active_topic": "health_daily_care",
                "selected_agent": "health_daily_care_agent",
                "detected_signals": ["dizzy", "fall"],
                "priority": "high",
                "reason": "Local router selected broad agents from latest user message.",
            },
            "alert_decision": {
                "alert_required": True,
                "alert_type": "emergency",
                "alert_level": "high",
                "alert_reason": "The user reports dizziness and an imminent risk of falling.",
                "user_requested_contact": True,
                "caregiver_message": "Please check on the elder immediately.",
            },
            "trace": {
                "llm_calls_this_turn": 1,
                "spans": [
                    {"name": "memory_context_builder", "order": 1, "duration_ms": 1, "status": "success"},
                    {"name": "router", "order": 2, "duration_ms": 2, "status": "success"},
                    {"name": "conversation_agent", "order": 3, "duration_ms": 3, "status": "success"},
                ],
            },
            "response_before_guardrail": "trust your instincts",
            "response_after_guardrail": "Do not rely on how convincing the caller sounds.",
            "OPENROUTER_API_KEY": "sk-or-secret",
            "system_prompt": "hidden instructions",
        },
    }


def test_sanitization_removes_secrets() -> None:
    sanitized = sanitize_developer_state(sample_response()["developer_state"])
    assert_true(sanitized["OPENROUTER_API_KEY"] == "[redacted]", "API key should be redacted")
    assert_true(sanitized["system_prompt"] == "[redacted]", "Hidden prompts should be redacted")


def test_family_explanation_is_structured_and_safe() -> None:
    explanation = build_family_explanation(
        sample_response(),
        "I feel dizzy and I am about to fall. Call my caregiver.",
    )
    assert_true(explanation["risk"] == "high", "Family view should show risk")
    assert_true(explanation["caregiver_requested"] == "Yes", "Caregiver request should be visible")
    assert_true("dizzy" in explanation["why"], "Explanation should use detected signals")
    rendered = str(explanation)
    assert_true("sk-or-secret" not in rendered, "Family view should not expose secrets")
    assert_true("hidden instructions" not in rendered, "Family view should not expose prompts")


def test_streamlit_elder_view_hides_developer_state() -> None:
    text = Path("streamlit_app.py").read_text(encoding="utf-8")
    elder_function = text.split("def render_elder_view", 1)[1].split("if \"messages\" not in st.session_state", 1)[0]
    assert_true("developer_state" not in elder_function, "Elder view should not display developer state")
    assert_true("raw_json" not in elder_function, "Elder view should not display raw JSON")
    assert_true("I'm thinking carefully about how to help..." in text, "Waiting sentence should be present")


def test_trace_execution_order_and_one_llm_call() -> None:
    trace = sample_response()["developer_state"]["trace"]
    orders = [span["order"] for span in trace["spans"]]
    assert_true(orders == sorted(orders), "Trace spans should stay in execution order")
    assert_true(trace["llm_calls_this_turn"] == 1, "Technical trace should display one LLM call")
    assert_true(
        "response_before_guardrail" in sample_response()["developer_state"]
        and "response_after_guardrail" in sample_response()["developer_state"],
        "Guardrail before/after values should be available",
    )


def test_ui_does_not_call_llm_directly() -> None:
    files = [
        Path("streamlit_app.py"),
        Path("app/ui/family_view.py"),
        Path("app/ui/technical_trace.py"),
        Path("app/ui/sanitization.py"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert_true("call_openrouter_llm" not in combined, "UI must not call OpenRouter")


if __name__ == "__main__":
    test_sanitization_removes_secrets()
    test_family_explanation_is_structured_and_safe()
    test_streamlit_elder_view_hides_developer_state()
    test_trace_execution_order_and_one_llm_call()
    test_ui_does_not_call_llm_directly()
    print("All UI/privacy tests passed.")
