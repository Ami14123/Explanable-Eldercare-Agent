from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import json
import pandas as pd


load_dotenv(
    override=True,
)


import requests
import streamlit as st

from app.ui.family_view import render_family_view
from app.ui.styles import apply_styles
from app.ui.technical_trace import render_trace_view


# Streamlit frontend for the local ElderGuard backend.
API_BASE_URL = "http://127.0.0.1:8000"
# Show this when FastAPI is not reachable.
OFFLINE_MESSAGE = (
    "ElderGuard is not connected right now. Please ask your caregiver to start the local care service."
)
# Model report files live under the project models folder.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_REPORT_ROOT = PROJECT_ROOT / "models"

# Demo prompts let users quickly exercise common care scenarios.
QUICK_PROMPTS = {
    "Bank scam": "Someone called and asked for my bank password.",
    "Dizzy": "I feel dizzy after standing up.",
    "Medicine": "I forgot my blood pressure medicine",
    "Lonely": "I feel lonely because my friends are busy",
}


# Configure the page once before rendering any Streamlit content.
st.set_page_config(page_title="ElderGuard", layout="wide")
apply_styles()

# Friendly names for internal broad-agent routes.
ROUTE_DISPLAY_NAMES = {
    "safety_agent": (
        "Safety and Fraud Protection"
    ),
    "health_daily_care_agent": (
        "Health and Daily Care"
    ),
    "emotional_social_agent": (
        "Emotional and Social Support"
    ),
    "action_agent": (
        "Practical Action Support"
    ),
}


# Friendly names for router decision sources shown in XAI panels.
ROUTER_SOURCE_DISPLAY_NAMES = {
    "rules": (
        "Rule based decision"
    ),
    "ml": (
        "Machine Learning decision"
    ),
    "ml_confirmed": (
        "Machine Learning confirmed by rules"
    ),
    "rules_override": (
        "Safety or workflow rule override"
    ),
    "rules_ml_disagreement": (
        "Rules retained after disagreement"
    ),
    "rules_fallback": (
        "Rule fallback after ML error"
    ),
}

# Render a simple user-facing explanation of the routing decision.
def render_xai_explanation(
    response_data: dict,
) -> None:
    """
    Render a simple user facing routing explanation.
    """

    # Pull explainability data out of the sanitized response shape.
    developer_state = (
        response_data.get(
            "developer_state",
            {},
        )
        or {}
    )

    router_decision = (
        developer_state.get(
            "router_decision",
            {},
        )
        or {}
    )

    xai = (
        router_decision.get(
            "xai",
            {},
        )
        or {}
    )

    # No XAI object means the backend did not provide route explanation data.
    if not xai:
        return

    # Translate internal route/source ids into display labels.
    final_route = str(
        xai.get(
            "final_route",
            router_decision.get(
                "selected_agent",
                "",
            ),
        )
    )

    router_source = str(
        xai.get(
            "router_source",
            router_decision.get(
                "router_source",
                "",
            ),
        )
    )

    route_name = (
        ROUTE_DISPLAY_NAMES.get(
            final_route,
            final_route,
        )
    )

    source_name = (
        ROUTER_SOURCE_DISPLAY_NAMES.get(
            router_source,
            router_source,
        )
    )

    explanation_text = str(
        xai.get(
            "explanation_text",
            router_decision.get(
                "reason",
                "",
            ),
        )
    )

    # Keep the explanation behind an expander so the chat stays simple.
    with st.expander(
        "Why ElderGuard responded this way",
        expanded=False,
    ):
        st.markdown(
            f"**Selected care area:** "
            f"{route_name}"
        )

        st.markdown(
            f"**Decision method:** "
            f"{source_name}"
        )

        if explanation_text:
            st.write(
                explanation_text
            )

        # Show model terms only when the ML router produced them.
        supporting_terms = list(
            xai.get(
                "supporting_terms",
                [],
            )
        )

        if supporting_terms:
            st.markdown(
                "**Important words or phrases:**"
            )

            for item in supporting_terms:
                term = str(
                    item.get(
                        "term",
                        "",
                    )
                )

                if term:
                    st.write(
                        f"• {term}"
                    )

        rule_candidate = str(
            xai.get(
                "rule_candidate",
                "",
            )
        )

        ml_candidate = str(
            xai.get(
                "ml_candidate",
                "",
            )
        )

        if (
            rule_candidate
            and ml_candidate
            and rule_candidate
            != ml_candidate
        ):
            # Disagreement is important because rules intentionally win in safety cases.
            st.info(
                "The ML model and the safety "
                "rules suggested different paths. "
                "ElderGuard used the safer or more "
                "context aware decision."
            )

        decision_margin = xai.get(
            "decision_margin"
        )

        # Margin is diagnostic only and should not be read as probability.
        if decision_margin is not None:
            st.caption(
                "Model separation score: "
                f"{decision_margin}. "
                "This score is not a probability."
            )

# Render caregiver alert text and send/not-sent status.
def render_caregiver_alert(
    response_data: dict,
) -> None:
    """
    Show a transparent caregiver alert status.
    """

    # Do nothing when no alert text is present.
    caregiver_message = str(
        response_data.get(
            "caregiver_alert",
            "",
        )
    ).strip()

    if not caregiver_message:
        return

    # Read the action state to say whether anything was actually sent.
    developer_state = (
        response_data.get(
            "developer_state",
            {},
        )
        or {}
    )

    alert_decision = (
        developer_state.get(
            "alert_decision",
            {},
        )
        or {}
    )

    action_state = (
        alert_decision.get(
            "action_state",
            {},
        )
        or {}
    )

    notification_sent = bool(
        action_state.get(
            "external_notification_sent",
            False,
        )
    )

    # Render the alert as a highlighted panel in the chat response.
    st.markdown(
        f"""
        <div class="care-alert">
          <strong>Caregiver alert prepared</strong><br>
          {caregiver_message}<br><br>
          <strong>Status:</strong> {"Sent" if notification_sent else "Not sent"}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not notification_sent:
        # Make prototype behavior explicit so users know no contact happened.
        st.caption(
            "Prototype note: ElderGuard prepared this alert text, "
            "but this demo did not contact anyone."
        )


# Read an optional access code from environment variables.
def _configured_access_code(name: str) -> str:
    return os.getenv(name, "").strip()


# Gate caregiver and developer views behind simple demo access codes.
def require_view_code(
    view_name: str,
    env_name: str,
) -> bool:
    """
    Gate non-elder views with simple environment-configured demo codes.
    """

    # Missing env code disables the protected view.
    configured_code = _configured_access_code(env_name)
    if not configured_code:
        st.info(
            f"{view_name} is disabled until {env_name} is set in the environment."
        )
        return False

    # Ask for a password-style access code without showing it on screen.
    entered_code = st.text_input(
        f"Enter {view_name} access code",
        type="password",
        key=f"{env_name}_input",
    ).strip()

    if entered_code != configured_code:
        st.warning(
            f"{view_name} requires an access code."
        )
        return False

    return True


# Render the detailed technical trace from the latest response.
def render_technical_trace(
    response_data: dict,
) -> None:
    """
    Render the explainable system trace.
    """

    # Developer state contains routing, alert, and LangGraph trace details.
    developer_state = (
        response_data.get(
            "developer_state",
            {},
        )
        or {}
    )

    router_decision = (
        developer_state.get(
            "router_decision",
            {},
        )
        or {}
    )

    xai = (
        router_decision.get(
            "xai",
            {},
        )
        or {}
    )

    alert_decision = (
        developer_state.get(
            "alert_decision",
            {},
        )
        or {}
    )

    # Summary metrics make the route, source, and risk scannable.
    st.subheader(
        "Explainable AI Trace"
    )

    final_route = str(
        xai.get(
            "final_route",
            router_decision.get(
                "selected_agent",
                "",
            ),
        )
    )

    router_source = str(
        xai.get(
            "router_source",
            router_decision.get(
                "router_source",
                "",
            ),
        )
    )

    priority = str(
        router_decision.get(
            "priority",
            "unknown",
        )
    )

    first_column, second_column, third_column = st.columns(3)

    first_column.metric(
        "Final specialist",
        ROUTE_DISPLAY_NAMES.get(
            final_route,
            final_route or "Unknown",
        ),
    )

    second_column.metric(
        "Decision source",
        ROUTER_SOURCE_DISPLAY_NAMES.get(
            router_source,
            router_source or "Unknown",
        ),
    )

    third_column.metric(
        "Risk priority",
        priority.title(),
    )

    # Show candidates so reviewers can compare rule and ML choices.
    st.markdown(
        "### Routing decision"
    )

    rule_candidate = str(
        xai.get(
            "rule_candidate",
            "Not available",
        )
    )

    ml_candidate = str(
        xai.get(
            "ml_candidate",
            "Not available",
        )
    )

    st.write(
        "**Rule candidate:**",
        ROUTE_DISPLAY_NAMES.get(
            rule_candidate,
            rule_candidate,
        ),
    )

    st.write(
        "**ML candidate:**",
        ROUTE_DISPLAY_NAMES.get(
            ml_candidate,
            ml_candidate,
        ),
    )

    st.write(
        "**Final decision:**",
        ROUTE_DISPLAY_NAMES.get(
            final_route,
            final_route,
        ),
    )

    explanation_text = str(
        xai.get(
            "explanation_text",
            router_decision.get(
                "reason",
                "",
            ),
        )
    )

    if explanation_text:
        st.info(
            explanation_text
        )

    # Supporting terms explain what moved the ML model toward the route.
    supporting_terms = list(
        xai.get(
            "supporting_terms",
            [],
        )
    )

    if supporting_terms:
        st.markdown(
            "### Model supporting terms"
        )

        st.dataframe(
            supporting_terms,
            hide_index=True,
            use_container_width=True,
        )

    opposing_terms = list(
        xai.get(
            "opposing_terms",
            [],
        )
    )

    if opposing_terms:
        # Opposing terms show evidence against the chosen ML route.
        st.markdown(
            "### Model opposing terms"
        )

        st.dataframe(
            opposing_terms,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown(
        "### Safety decision"
    )

    # Alert metadata explains whether caregiver contact was prepared.
    alert_required = bool(
        alert_decision.get(
            "alert_required",
            False,
        )
    )

    st.write(
        "**Alert required:**",
        "Yes" if alert_required else "No",
    )

    st.write(
        "**Alert level:**",
        str(
            alert_decision.get(
                "alert_level",
                "none",
            )
        ).title(),
    )

    st.write(
        "**Alert reason:**",
        str(
            alert_decision.get(
                "alert_reason",
                "No alert reason.",
            )
        ),
    )

    action_state = (
        alert_decision.get(
            "action_state",
            {},
        )
        or {}
    )

    notification_sent = bool(
        action_state.get(
            "external_notification_sent",
            False,
        )
    )

    st.write(
        "**External notification:**",
        "Sent" if notification_sent else "Not sent",
    )

    # Show LangGraph spans when the backend attached them.
    trace_spans = list(
        developer_state.get(
            "langgraph_trace",
            [],
        )
    )

    if trace_spans:
        st.markdown(
            "### LangGraph execution"
        )

        for span in trace_spans:
            order = span.get(
                "order",
                "",
            )

            name = span.get(
                "name",
                span.get(
                    "node",
                    "Unknown node",
                ),
            )

            duration = span.get(
                "duration_ms",
                0,
            )

            label = f"{order}. {name} ({duration} ms)"

            with st.expander(
                label,
                expanded=False,
            ):
                st.json(
                    span.get(
                        "attributes",
                        {},
                    )
                )

    with st.expander(
        "Raw developer data",
        expanded=False,
    ):
        # Raw developer data is protected by the developer access gate.
        st.json(
            developer_state
        )


# Render model evaluation artifacts saved in the models folder.
def render_model_insights() -> None:
    """
    Display global model performance.
    """

    st.title(
        "AI Model Insights"
    )

    metrics_path = MODEL_REPORT_ROOT / "router_metrics.json"
    comparison_path = MODEL_REPORT_ROOT / "router_comparison.csv"
    confusion_path = MODEL_REPORT_ROOT / "confusion_matrix.png"
    chart_path = MODEL_REPORT_ROOT / "model_comparison.png"

    # Missing reports are normal before model training/evaluation runs.
    if not MODEL_REPORT_ROOT.exists():
        st.info(
            "No model report folder was found yet."
        )
        return

    shown_anything = False

    # Metrics JSON identifies the selected router model and score.
    if metrics_path.exists():
        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(
                file
            )

        first_column, second_column = st.columns(2)

        first_column.metric(
            "Selected model",
            str(
                metrics.get(
                    "best_model",
                    "Unknown",
                )
            ),
        )

        second_column.metric(
            "Macro F1",
            f"{float(metrics.get('best_macro_f1', 0)):.4f}",
        )

        st.caption(
            "The model was selected using "
            "the highest Macro F1 score."
        )
        shown_anything = True

    # Comparison CSV shows all evaluated model scores.
    if comparison_path.exists():
        st.subheader(
            "Model comparison"
        )

        comparison_data = pd.read_csv(
            comparison_path
        )

        st.dataframe(
            comparison_data,
            hide_index=True,
            use_container_width=True,
        )
        shown_anything = True

    # Optional chart image visualizes model performance.
    if chart_path.exists():
        st.subheader(
            "Performance chart"
        )

        st.image(
            str(chart_path),
            use_container_width=True,
        )
        shown_anything = True

    # Optional confusion matrix shows class-level mistakes.
    if confusion_path.exists():
        st.subheader(
            "Confusion matrix"
        )

        st.image(
            str(confusion_path),
            use_container_width=True,
        )

        st.caption(
            "Diagonal cells are correct predictions. "
            "Off diagonal cells show class confusion."
        )
        shown_anything = True

    # Explain which files the page expects when nothing is available.
    if not shown_anything:
        st.info(
            "No model insight files were found yet. "
            "Add router_metrics.json, router_comparison.csv, "
            "model_comparison.png, or confusion_matrix.png "
            "inside the models folder."
        )


# GET helper for backend API calls.
def api_get(path: str, timeout: int = 10) -> tuple[bool, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        # Return the exception to let callers decide the user-facing message.
        return False, exc


# POST helper for backend API calls.
def api_post(path: str, payload: dict[str, Any], timeout: int = 60) -> tuple[bool, Any]:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, exc


# Fetch public backend status for sidebar and readiness checks.
def get_status() -> tuple[bool, dict[str, Any]]:
    ok, data = api_get("/status", timeout=5)
    if ok and isinstance(data, dict):
        return True, data
    return False, {}


# Fetch recent logs for views that need them.
def get_logs() -> list[dict[str, Any]]:
    ok, data = api_get("/logs?limit=20")
    return data if ok and isinstance(data, list) else []


# Send one chat message to FastAPI and update Streamlit session state.
def send_to_backend(message: str, user_id: str = "demo_user") -> None:
    # Vietnamese note: Streamlit gui mot chat turn vao FastAPI gateway va giu conversation_id on dinh.
    # Save the user message immediately so the chat feels responsive.
    st.session_state.messages.append({"role": "user", "content": message})
    # Include recent chat history so the backend can answer follow-ups.
    client_history = [
        {"role": item["role"], "message": item["content"]}
        for item in st.session_state.messages[-10:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]

    # The backend does the reasoning and returns a structured response.
    with st.spinner("I'm thinking carefully about how to help..."):
        ok, data = api_post(
            "/chat",
            {
                "message": message,
                "user_id": user_id,
                "conversation_id": st.session_state.conversation_id,
                "client_history": client_history,
            },
            timeout=90,
        )

    if ok:
        # Store the full response for family and technical views.
        st.session_state.last_response = data
        st.session_state["latest_response"] = data
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data.get(
                    "final_message",
                    "I am here with you. Could you tell me a little more?",
                ),
                "data": data,
            }
        )
        st.session_state.answer_ready = True
    else:
        # Use a calm offline message when the local backend is unavailable.
        st.session_state.messages.append({"role": "assistant", "content": OFFLINE_MESSAGE})


# Render the elder-facing chat page.
def render_elder_view(status_ok: bool) -> None:
    # Vietnamese note: Elder view chi hien trai nghiem don gian; routing/trace nam trong view ky thuat.
    st.markdown(
        """
        <div class="eg-header">
          <div class="eg-title">ElderGuard</div>
          <div class="eg-subtitle">A simple AI care companion for everyday support.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="eg-card">
          <strong>Prototype safety note:</strong>
          ElderGuard is not a medical device and does not diagnose or prescribe.
          If you may be in immediate danger, call local emergency services or ask
          someone nearby for help now. Caregiver alerts in this demo are prepared
          as text only and are not automatically sent.
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_cols = st.columns([1, 1, 2])
    # Show backend readiness and emergency reminder controls.
    with status_cols[0]:
        st.markdown(
            f'<span class="status-badge">{"Care service ready" if status_ok else "Care service offline"}</span>',
            unsafe_allow_html=True,
        )
    with status_cols[1]:
        if st.button("Emergency help", use_container_width=True):
            st.warning("If you are in immediate danger, call local emergency services or ask someone nearby for help now.")

    st.markdown("#### Try a demo scenario")
    quick_cols = st.columns(len(QUICK_PROMPTS))
    # Quick prompts seed the chat with common test scenarios.
    for index, (label, prompt) in enumerate(QUICK_PROMPTS.items()):
        with quick_cols[index]:
            if st.button(label, use_container_width=True):
                send_to_backend(prompt)
                st.rerun()

    # Show a one-time confirmation after the backend response arrives.
    if st.session_state.get("answer_ready"):
        st.success("Your answer is ready.")
        st.session_state.answer_ready = False

    st.markdown("#### Chat with ElderGuard")

    # Replay the current chat transcript.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            data = message.get("data")

            if data:
                # Attach optional explanation and alert panels below responses.
                render_xai_explanation(data)
                render_caregiver_alert(data)

    # Send typed user input to the backend.
    user_text = st.chat_input("Type your message to ElderGuard...")
    if user_text:
        send_to_backend(user_text)
        st.rerun()


# Initialize chat transcript once per Streamlit session.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, I am ElderGuard. Tell me what is happening, and I will help with one simple next step.",
        }
    ]
# Store the latest backend response for family/developer views.
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "latest_response" not in st.session_state:
    st.session_state.latest_response = None
# Keep one conversation id so memory and follow-ups stay connected.
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = "default"
if "answer_ready" not in st.session_state:
    st.session_state.answer_ready = False
# Track which sidebar view is active.
if "interface_role" not in st.session_state:
    st.session_state.interface_role = "elder"

# Check backend status once for this render cycle.
status_ok, status_data = get_status()

# Sidebar chooses which app view to render.
with st.sidebar:
    st.title("ElderGuard")
    selected = st.radio(
        "View",
        ["Chat", "Family Summary", "Technical Trace", "AI Model Insights"],
        index=["elder", "family", "developer", "model_insights"].index(st.session_state.interface_role)
        if st.session_state.interface_role in {"elder", "family", "developer", "model_insights"}
        else 0,
    )
    st.session_state.interface_role = {
        "Chat": "elder",
        "Family Summary": "family",
        "Technical Trace": "developer",
        "AI Model Insights": "model_insights",
    }[selected]
    st.divider()
    st.write("Status")
    if status_ok:
        st.success("Connected")
    else:
        st.error("Offline")

# Render the selected view, applying access gates outside the elder chat.
if st.session_state.interface_role == "family":
    if require_view_code(
        "Caregiver View",
        "CAREGIVER_ACCESS_CODE",
    ):
        render_family_view(
            st.session_state.messages,
            st.session_state.last_response,
            [],
        )
elif st.session_state.interface_role == "developer":
    if require_view_code(
        "Developer Trace",
        "DEVELOPER_ACCESS_CODE",
    ):
        latest_response = st.session_state.get("latest_response")

        if latest_response is None:
            st.info(
                "Send a chat message first "
                "to create a technical trace."
            )
        else:
            render_technical_trace(
                latest_response
            )
elif st.session_state.interface_role == "model_insights":
    if require_view_code(
        "Developer Trace",
        "DEVELOPER_ACCESS_CODE",
    ):
        render_model_insights()
else:
    render_elder_view(status_ok)
