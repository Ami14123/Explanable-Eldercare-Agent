from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from app.ui.sanitization import family_safe_response


AGENT_LABELS = {
    "safety_agent": "Safety support",
    "health_daily_care_agent": "Health and daily care",
    "emotional_social_agent": "Emotional and social support",
    "action_agent": "Action help",
}


def risk_badge(risk: str) -> str:
    risk = (risk or "low").lower()
    label = risk.title() if risk in {"low", "medium", "high"} else "Low"
    css = risk if risk in {"low", "medium", "high"} else "low"
    return f'<span class="risk-{css}">{label}</span>'


def build_family_explanation(response: dict[str, Any], elder_message: str = "") -> dict[str, Any]:
    safe = family_safe_response(response)
    developer = safe.get("developer_state", {})
    router = developer.get("router_decision") or safe.get("care_plan", {}).get("router_decision", {})
    alert = developer.get("alert_decision") or safe.get("care_plan", {}).get("alert_decision", {})
    graph = safe.get("graph_reasoning", {})
    selected_agent = router.get("selected_agent") or (safe.get("activated_agents") or [""])[0]
    signals = router.get("detected_signals") or [
        item.get("signal", "") for item in safe.get("detected_signals", [])
    ]
    sources = graph.get("retrieved_sources") or graph.get("detected_nodes") or graph.get("reasoning_path") or []
    if sources and isinstance(sources[0], list):
        sources = [" -> ".join(path) for path in sources]

    risk = safe.get("overall_risk") or router.get("priority") or "low"
    alert_required = bool(alert.get("alert_required"))
    user_requested = bool(alert.get("user_requested_contact"))
    concern = safe.get("main_concern") or router.get("active_topic") or "General support"
    reason_parts = []
    if signals:
        reason_parts.append("The system detected: " + ", ".join(map(str, signals[:6])) + ".")
    if alert_required:
        reason_parts.append(str(alert.get("alert_reason", "")).strip())
    if router.get("reason"):
        reason_parts.append(str(router["reason"]))
    reason = " ".join(part for part in reason_parts if part).strip() or "The answer was based on the elder's latest message and recent conversation context."

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "elder_message": elder_message,
        "topic": str(router.get("active_topic") or concern).replace("_", " ").title(),
        "risk": risk,
        "selected_area": AGENT_LABELS.get(str(selected_agent), str(selected_agent).replace("_", " ").title()),
        "main_concern": concern,
        "caregiver_requested": "Yes" if user_requested else "No",
        "alert_status": "Prepared" if alert_required else "Not needed",
        "why": reason,
        "information_used": [str(item) for item in sources[:8]],
        "caregiver_action": alert.get("caregiver_message") or safe.get("caregiver_alert") or "No caregiver action was prepared.",
        "suggested_next_step": safe.get("next_step") or "Continue the conversation if more help is needed.",
    }


def render_family_view(messages: list[dict[str, Any]], last_response: dict[str, Any] | None, logs: list[dict[str, Any]]) -> None:
    st.markdown(
        """
        <div class="eg-header">
          <div class="eg-title">Family Summary</div>
          <div class="eg-subtitle">A clear, nontechnical view for authorized caregivers and family.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Conversation details should only be viewed by an authorized caregiver or family member.")

    elder_message = ""
    for item in reversed(messages):
        if item.get("role") == "user":
            elder_message = str(item.get("content", ""))
            break

    if last_response:
        explanation = build_family_explanation(last_response, elder_message)
        st.markdown('<div class="eg-card">', unsafe_allow_html=True)
        st.markdown("#### What the elder said")
        st.write(explanation["elder_message"] or "No message selected yet.")
        st.markdown("#### What the system detected")
        st.markdown(
            f"Topic: **{explanation['topic']}**  \n"
            f"Risk: {risk_badge(str(explanation['risk']))}  \n"
            f"Support area: **{explanation['selected_area']}**  \n"
            f"Caregiver requested: **{explanation['caregiver_requested']}**",
            unsafe_allow_html=True,
        )
        st.markdown("#### Why this response was given")
        st.write(explanation["why"])
        st.markdown("#### Information used")
        if explanation["information_used"]:
            st.write(", ".join(explanation["information_used"]))
        else:
            st.write("Recent conversation context and local safety rules.")
        st.markdown("#### Caregiver action")
        st.write(explanation["caregiver_action"])
        st.markdown("#### Suggested next step")
        st.write(explanation["suggested_next_step"])
        st.caption(f"Conversation time: {explanation['timestamp']}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("No conversation response yet.")

    with st.expander("Previous conversations", expanded=False):
        if not logs:
            st.write("No previous conversations yet.")
        for item in logs[:10]:
            response = item.get("raw_json", {})
            alert = response.get("alert_decision", {})
            st.markdown(
                f"""
                <div class="eg-card">
                  <strong>{item.get('timestamp', '')}</strong><br>
                  Elder: {item.get('message', '')}<br>
                  Risk: {risk_badge(str(item.get('overall_risk', 'low')))}
                  <br>Alert: {'Prepared' if alert.get('alert_required') else 'Not needed'}
                </div>
                """,
                unsafe_allow_html=True,
            )
