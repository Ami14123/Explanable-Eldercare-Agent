from app.graph import (
    _alert_template,
    _set_alert_if_not_downgrade,
    apply_output_guardrails,
    evaluate_alert_decision,
)


def assert_in(member, container, label: str) -> None:
    if member not in container:
        raise AssertionError(f"{label}: expected {member!r} in {container!r}")


def assert_not_in(member, container, label: str) -> None:
    if member in container:
        raise AssertionError(f"{label}: expected {member!r} not in {container!r}")


def main() -> None:
    fraud_router = {
        "active_topic": "safety",
        "activated_agents": ["safety_agent"],
        "detected_signals": ["email", "link"],
    }
    no_alert = {"alert_required": False, "alert_type": "none", "alert_level": "low"}

    result = apply_output_guardrails(
        "Trust your instincts and prioritize your safety.",
        alert_decision=no_alert,
        router_decision=fraud_router,
    )
    assert_in("vague_safety_advice", result["guardrail_issues"], "vague fraud wording")
    assert_not_in("trust your instincts", result["final_message"].lower(), "vague phrase removed")
    assert_in("official phone number", result["final_message"].lower(), "concrete verification guidance")

    result = apply_output_guardrails(
        "End the call and verify the claim using an official number you find independently.",
        alert_decision=no_alert,
        router_decision=fraud_router,
    )
    assert_not_in("vague_safety_advice", result["guardrail_issues"], "already safe fraud response")
    assert result["fallback_used"] is False

    result = apply_output_guardrails(
        "Call your caregiver immediately.",
        alert_decision=no_alert,
        router_decision={"active_topic": "health_daily_care", "activated_agents": ["health_daily_care_agent"], "detected_signals": []},
    )
    assert_in("alert_decision_response_mismatch", result["guardrail_issues"], "alert mismatch")
    assert result["fallback_used"] is True

    alert = _alert_template()
    _set_alert_if_not_downgrade(
        alert,
        alert_type="emergency",
        level="high",
        reason="high",
        contact="emergency",
        caregiver_message="high",
    )
    _set_alert_if_not_downgrade(
        alert,
        alert_type="caregiver",
        level="medium",
        reason="medium",
        contact="caregiver",
        caregiver_message="medium",
    )
    assert alert["alert_type"] == "emergency"
    assert alert["alert_level"] == "high"

    exact = evaluate_alert_decision("I feel dizzy and I am about to fall, call my caregiver")
    assert exact["alert_required"] is True
    assert exact["alert_type"] == "emergency"
    assert exact["alert_level"] == "high"
    assert exact["user_requested_contact"] is True

    print("All guardrail tests passed.")


if __name__ == "__main__":
    main()
