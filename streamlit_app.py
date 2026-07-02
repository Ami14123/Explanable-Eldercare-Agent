from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from app.ui.family_view import render_family_view
from app.ui.styles import apply_styles
from app.ui.technical_trace import render_trace_view


API_BASE_URL = "http://127.0.0.1:8000"
OFFLINE_MESSAGE = (
    "ElderGuard is not connected right now. Please ask your caregiver to start the local care service."
)

QUICK_PROMPTS = {
    "Juice help": "I want juice but I don't have money to buy",
    "Unknown email": "Someone emailed me but I don't know them, should I click the link?",
    "Dizzy": "I feel dizzy after standing up",
    "Medicine": "I forgot my blood pressure medicine",
    "Lonely": "I feel lonely because my friends are busy",
}


st.set_page_config(page_title="ElderGuard", layout="wide")
apply_styles()


def api_get(path: str, timeout: int = 10) -> tuple[bool, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, exc


def api_post(path: str, payload: dict[str, Any], timeout: int = 60) -> tuple[bool, Any]:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, exc


def get_status() -> tuple[bool, dict[str, Any]]:
    ok, data = api_get("/status", timeout=5)
    if ok and isinstance(data, dict):
        return True, data
    return False, {}


def get_logs() -> list[dict[str, Any]]:
    ok, data = api_get("/logs?limit=20")
    return data if ok and isinstance(data, list) else []


def send_to_backend(message: str, user_id: str = "demo_user") -> None:
    st.session_state.messages.append({"role": "user", "content": message})
    client_history = [
        {"role": item["role"], "message": item["content"]}
        for item in st.session_state.messages[-10:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]

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
        st.session_state.last_response = data
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
        st.session_state.messages.append({"role": "assistant", "content": OFFLINE_MESSAGE})


def render_elder_view(status_ok: bool) -> None:
    st.markdown(
        """
        <div class="eg-header">
          <div class="eg-title">ElderGuard</div>
          <div class="eg-subtitle">How can I help you today?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_cols = st.columns([1, 1, 3])
    with status_cols[0]:
        st.markdown(
            f'<span class="status-badge">{"Care service ready" if status_ok else "Care service offline"}</span>',
            unsafe_allow_html=True,
        )
    with status_cols[1]:
        if st.button("Emergency help", use_container_width=True):
            st.warning("If you are in immediate danger, call local emergency services or ask someone nearby for help now.")

    quick_cols = st.columns(len(QUICK_PROMPTS))
    for index, (label, prompt) in enumerate(QUICK_PROMPTS.items()):
        with quick_cols[index]:
            if st.button(label, use_container_width=True):
                send_to_backend(prompt)
                st.rerun()

    if st.session_state.get("answer_ready"):
        st.success("Your answer is ready.")
        st.session_state.answer_ready = False

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            data = message.get("data")
            if data and data.get("human_confirmation_required") and data.get("caregiver_alert"):
                st.markdown(
                    f'<div class="care-alert"><strong>Caregiver alert:</strong><br>{data["caregiver_alert"]}</div>',
                    unsafe_allow_html=True,
                )

    user_text = st.chat_input("Type your message to ElderGuard...")
    if user_text:
        send_to_backend(user_text)
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, I am ElderGuard. Tell me what is happening, and I will help with one simple next step.",
        }
    ]
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = "default"
if "answer_ready" not in st.session_state:
    st.session_state.answer_ready = False
if "interface_role" not in st.session_state:
    st.session_state.interface_role = "elder"

status_ok, status_data = get_status()

with st.sidebar:
    st.title("ElderGuard")
    selected = st.radio(
        "View",
        ["Chat", "Family Summary", "Technical Trace"],
        index=["elder", "family", "developer"].index(st.session_state.interface_role)
        if st.session_state.interface_role in {"elder", "family", "developer"}
        else 0,
    )
    st.session_state.interface_role = {
        "Chat": "elder",
        "Family Summary": "family",
        "Technical Trace": "developer",
    }[selected]
    st.divider()
    st.write("Status")
    if status_ok:
        st.success("Connected")
    else:
        st.error("Offline")

if st.session_state.interface_role == "family":
    render_family_view(st.session_state.messages, st.session_state.last_response, get_logs())
elif st.session_state.interface_role == "developer":
    render_trace_view(st.session_state.last_response, status_data)
else:
    render_elder_view(status_ok)
