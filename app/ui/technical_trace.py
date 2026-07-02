from __future__ import annotations

from typing import Any

import streamlit as st

from app.ui.sanitization import sanitize_developer_state


def _status_class(status: str) -> str:
    status = status if status in {"success", "warning", "error", "skipped", "running"} else "success"
    return f"trace-{status}"


def render_trace_view(last_response: dict[str, Any] | None, status: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="eg-header">
          <div class="eg-title">Technical Trace</div>
          <div class="eg-subtitle">Observability for the local LangGraph workflow.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not last_response:
        st.write("Run a chat message first to create a trace.")
        return

    developer = sanitize_developer_state(last_response.get("developer_state", {}))
    trace = developer.get("trace") or {}
    spans = trace.get("spans") or developer.get("langgraph_trace", [])
    guardrail_issues = developer.get("guardrail_issues", [])

    cols = st.columns(4)
    cols[0].metric("Total duration", f"{trace.get('total_duration_ms', 0)} ms")
    cols[1].metric("Selected specialist", trace.get("selected_agent") or developer.get("llm_caller", ""))
    cols[2].metric("Risk level", trace.get("risk") or last_response.get("overall_risk", "low"))
    cols[3].metric("LLM calls", trace.get("llm_calls_this_turn", developer.get("llm_calls_this_turn", 0)))

    cols = st.columns(4)
    cols[0].metric("OpenRouter model", trace.get("llm_model") or status.get("llm_model", "none"))
    cols[1].metric("RAG sources", len(trace.get("rag_sources", []) or []))
    cols[2].metric("Alert status", trace.get("alert_status", "none"))
    cols[3].metric("Guardrail", "warning" if guardrail_issues else "success")

    st.markdown('<div class="trace-shell">', unsafe_allow_html=True)
    for span in spans:
        name = span.get("name") or span.get("node", "unknown")
        status_label = span.get("status", "success")
        duration = span.get("duration_ms", 0)
        order = span.get("order", "")
        st.markdown(
            f"""
            <div class="trace-row">
              <div>{order}</div>
              <div>{name}</div>
              <div>{duration} ms</div>
              <div class="{_status_class(str(status_label))}">{status_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Details: {name}", expanded=False):
            st.json(sanitize_developer_state(span.get("attributes", span)))
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Output guardrail")
    st.json(
        {
            "guardrail_issues": developer.get("guardrail_issues", []),
            "response_before_guardrail": developer.get("response_before_guardrail", ""),
            "response_after_guardrail": developer.get("response_after_guardrail", ""),
            "fallback_used": developer.get("fallback_used", False),
            "fallback_name": developer.get("fallback_name", ""),
            "final_message_source": developer.get("final_message_source", ""),
        }
    )

    st.subheader("Memory")
    st.json(
        {
            "memory_saved": developer.get("memory_saved", False),
            "classified_memories": developer.get("classified_memories", []),
            "memory_policy": developer.get("memory_policy", ""),
        }
    )

    with st.expander("Advanced technical data", expanded=False):
        st.json(sanitize_developer_state(last_response))
