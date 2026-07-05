from __future__ import annotations

import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.agents import call_openrouter_llm, get_llm_call_counter, reset_llm_call_counter
from app.care_graph import get_graph_reasoning
from app.config import get_settings
from app.memory import (
    get_user_profile,
    get_recent_conversation,
    get_relevant_care_events,
    get_user_summary,
    save_care_event,
    save_interaction,
    save_message,
    save_user_profile,
    update_user_summary,
)
from app.memory_classifier import classify_memory, profile_updates_from_memories
from app.ml_router import extract_latest_user_text, explain_ml_route
from app.schemas import ChatRequest, ChatResponse


# Main LangGraph workflow for routing, safety checks, response generation, and memory.
logger = logging.getLogger(__name__)

# Ordered node names used in developer traces and fallback responses.
LANGGRAPH_PATH = [
    "START",
    "memory_context_builder",
    "router",
    "broad_agent_reasoning",
    "alert_decision",
    "conversation_agent",
    "guardrails",
    "save_memory",
    "END",
]

# Broad specialists used by the simplified routing layer.
BROAD_AGENTS = {
    "safety_agent": "Fraud, scam, suspicious email, unsafe pressure, emergency/fall danger.",
    "health_daily_care_agent": "Health, medication, food, nutrition, mobility, and daily needs.",
    "emotional_social_agent": "Loneliness, stress, family support, reassurance.",
    "action_agent": "Write messages, checklists, reminders, caregiver notes, call scripts.",
}

# Alert levels are ordered so higher-risk alerts are never downgraded.
ALERT_LEVEL_ORDER = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}

# Concrete replacement text used when fraud responses are too vague.
CONCRETE_FRAUD_VERIFICATION_GUIDANCE = (
    "Do not rely on how convincing the caller sounds. End the call and verify the claim using an official phone number "
    "that you find independently. Do not use a phone number, link, or contact method provided by the caller. Do not "
    "share passwords, one time codes, banking information, or personal information."
)

# Phrases that are too vague for scam or fraud safety guidance.
VAGUE_SAFETY_PHRASES = [
    "trust your instincts",
    "trust your instinct",
    "follow your gut",
    "use your best judgment",
    "listen to your intuition",
]

# Phrases that imply caregiver contact and must match alert policy.
CAREGIVER_ACTION_PHRASES = [
    "call your caregiver",
    "contact your caregiver",
    "alert your caregiver",
    "notify your caregiver",
    "call your family",
    "contact your family",
    "call my caregiver",
    "contact my caregiver",
    "alert my caregiver",
    "notify my caregiver",
]


# Shared state passed between LangGraph nodes.
class ElderGuardState(TypedDict, total=False):
    request: ChatRequest
    memory_context: dict[str, Any]
    router_decision: dict[str, Any]
    broad_agent_outputs: dict[str, dict[str, Any]]
    alert_decision: dict[str, Any]
    coordinated: dict[str, Any]
    final_response: ChatResponse
    raw_json: dict[str, Any]
    langgraph_trace: list[dict[str, Any]]
    trace_id: str
    trace_started_perf: float
    trace_last_perf: float


# Append one trace span and merge node updates into state.
def _trace(state: ElderGuardState, node: str, **updates: Any) -> ElderGuardState:
    next_state: ElderGuardState = {**state, **updates}
    # Measure elapsed time since the previous node.
    trace = list(state.get("langgraph_trace", []))
    now = time.perf_counter()
    last = float(state.get("trace_last_perf", state.get("trace_started_perf", now)))
    duration_ms = max(0.0, (now - last) * 1000)
    attributes = {
        "active_topic": next_state.get("router_decision", {}).get("active_topic", ""),
        "activated_agents": next_state.get("router_decision", {}).get("activated_agents", []),
        "llm_calls_this_turn": get_llm_call_counter(),
        "final_message_source": next_state.get("coordinated", {}).get("final_message_source", ""),
    }
    # Add node-specific attributes so each trace row explains its own output.
    if node == "router":
        attributes["router_decision"] = next_state.get("router_decision", {})
    elif node == "broad_agent_reasoning":
        attributes["selected_agent"] = next_state.get("router_decision", {}).get("selected_agent", "")
        attributes["broad_agent_outputs"] = next_state.get("broad_agent_outputs", {})
    elif node == "alert_decision":
        attributes["alert_decision"] = next_state.get("alert_decision", {})
    elif node == "conversation_agent":
        attributes.update(
            {
                "llm_provider": next_state.get("coordinated", {}).get("llm_provider", ""),
                "llm_model": next_state.get("coordinated", {}).get("llm_model", ""),
                "llm_caller": next_state.get("coordinated", {}).get("llm_caller", ""),
                "prompt_name": next_state.get("coordinated", {}).get("prompt_name", ""),
                "prompt_version": next_state.get("coordinated", {}).get("prompt_version", ""),
                "llm_error": next_state.get("coordinated", {}).get("llm_error", {}),
            }
        )
    elif node == "guardrails":
        raw = next_state.get("raw_json", {})
        attributes.update(
            {
                "guardrail_issues": raw.get("guardrail_issues", []),
                "fallback_used": raw.get("fallback_used", False),
                "fallback_name": raw.get("fallback_name", ""),
                "response_before_guardrail": raw.get("response_before_guardrail", ""),
                "response_after_guardrail": raw.get("response_after_guardrail", ""),
            }
        )
    trace.append(
        {
            "name": node,
            "node": node,
            "order": len(trace) + 1,
            "status": "error"
            if next_state.get("coordinated", {}).get("llm_error")
            else "warning"
            if node == "guardrails" and next_state.get("raw_json", {}).get("guardrail_issues")
            else "success",
            "duration_ms": round(duration_ms, 2),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes,
            "active_topic": next_state.get("router_decision", {}).get("active_topic", ""),
            "activated_agents": next_state.get("router_decision", {}).get("activated_agents", []),
            "llm_calls_this_turn": get_llm_call_counter(),
            "final_message_source": next_state.get("coordinated", {}).get("final_message_source", ""),
        }
    )
    next_state["langgraph_trace"] = trace
    next_state["trace_last_perf"] = now
    return next_state


# Check whether any trigger term appears in lowercase text.
def _contains(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


# Return the highest risk level from several candidates.
def _risk_max(*risks: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return max((risk if risk in order else "low" for risk in risks), key=lambda item: order[item])


# Create the default alert decision object.
def _alert_template() -> dict[str, Any]:
    return {
        "alert_required": False,
        "alert_type": "none",
        "alert_level": "low",
        "alert_reason": "",
        "recommended_contact": "",
        "human_confirmation_required": False,
        "user_requested_contact": False,
        "caregiver_message": "",
        "action_state": {},
    }


# Update an alert only when the new alert is not lower priority.
def _set_alert_if_not_downgrade(
    alert: dict[str, Any],
    *,
    alert_type: str,
    level: str,
    reason: str,
    contact: str,
    caregiver_message: str,
    user_requested_contact: bool = False,
) -> None:
    current_score = ALERT_LEVEL_ORDER.get(str(alert.get("alert_level", "none")), 0)
    next_score = ALERT_LEVEL_ORDER.get(level, 0)
    # Preserve stronger existing alerts while still recording user contact intent.
    if next_score < current_score:
        if user_requested_contact:
            alert["user_requested_contact"] = True
            alert["human_confirmation_required"] = False
        return
    # Store the prepared alert and explain that no external notification was sent.
    alert.update(
        {
            "alert_required": True,
            "alert_type": alert_type,
            "alert_level": level,
            "alert_reason": reason,
            "recommended_contact": contact,
            "human_confirmation_required": not user_requested_contact,
            "user_requested_contact": bool(user_requested_contact),
            "caregiver_message": caregiver_message,
            "action_state": {
                "prepared": True,
                "external_notification_sent": False,
                "reason": "No verified caregiver contact/action tool is configured.",
            },
        }
    )


# Decide whether a message should prepare a caregiver or emergency alert.
def evaluate_alert_decision(message: str) -> dict[str, Any]:
    text = message.lower()
    alert = _alert_template()

    # Detect health and caregiver-request signals first.
    dizzy_now = _contains(text, ["dizzy", "dizziness", "lightheaded", "light headed"])
    imminent_fall = _contains(
        text,
        [
            "about to fall",
            "going to fall",
            "might fall",
            "may fall",
            "feel like i will fall",
            "losing my balance",
            "cannot keep my balance",
            "can't keep my balance",
            "unsteady",
        ],
    )
    explicit_caregiver_request = _contains(
        text,
        [
            "call my caregiver",
            "contact my caregiver",
            "alert my caregiver",
            "notify my caregiver",
            "get my caregiver",
            "call my family",
            "contact my family",
            "please call my caregiver",
        ],
    )
    emergency_red_flag = _contains(
        text,
        [
            "chest pain",
            "trouble breathing",
            "can't breathe",
            "fainting",
            "fainted",
            "fall",
            "fell",
            "confusion",
            "confused",
            "hurt myself",
            "kill myself",
            "end my life",
            "suicide",
        ],
    )

    # Suspicious messages with sensitive requests can require safety escalation.
    suspicious_email = _contains(text, ["email", "emailed", "link", "click"])
    fraud_action = _contains(text, ["money", "pay", "transfer", "login", "log in", "password", "download", "bank", "otp"])
    if suspicious_email and fraud_action:
        level = "high" if _contains(text, ["password", "otp", "transfer", "bank"]) else "medium"
        _set_alert_if_not_downgrade(
            alert,
            alert_type="fraud",
            level=level,
            reason="Suspicious message includes a link or email plus a sensitive request.",
            contact="trusted family member or official organization contact",
            caregiver_message="Please check this message with the user before they click links, log in, download files, or send money.",
        )

    # Expired food plus symptoms is a health concern, not just daily support.
    if _contains(text, ["expired food", "expired", "old food"]) and _contains(text, ["stomach pain", "stomach hurts", "belly pain", "vomit", "nausea"]):
        _set_alert_if_not_downgrade(
            alert,
            alert_type="health",
            level="medium",
            reason="Expired food with stomach symptoms may need health attention.",
            contact="caregiver or healthcare professional",
            caregiver_message="The user reported expired food and stomach symptoms. Please check on them and consider medical advice if symptoms continue.",
        )

    # Emergency or fall risk receives the strongest alert.
    if imminent_fall or emergency_red_flag:
        reason = "The user reports dizziness and an imminent risk of falling." if dizzy_now and imminent_fall else "The user reported an emergency safety risk."
        _set_alert_if_not_downgrade(
            alert,
            alert_type="emergency",
            level="high",
            reason=reason,
            contact="caregiver or nearby trusted person",
            user_requested_contact=explicit_caregiver_request,
            caregiver_message="The user feels dizzy and believes they may fall. Please check on them immediately and help them remain seated or lying down safely.",
        )
    elif explicit_caregiver_request:
        _set_alert_if_not_downgrade(
            alert,
            alert_type="caregiver",
            level="medium",
            reason="The user explicitly asked to contact a caregiver or family member.",
            contact="caregiver or family",
            user_requested_contact=True,
            caregiver_message="The user requested caregiver or family contact. Please check what help they need.",
        )
    elif dizzy_now:
        alert["alert_reason"] = "Dizziness mentioned without imminent fall or emergency red flags."

    # Medication uncertainty should involve a caregiver or clinician.
    if _contains(text, ["medicine", "medication", "pill", "dose", "dosage", "forgot", "missed"]) and _contains(text, ["unsure", "forgot", "missed", "don't know", "not sure"]):
        _set_alert_if_not_downgrade(
            alert,
            alert_type="caregiver",
            level="medium",
            reason="Medication uncertainty may need confirmation.",
            contact="caregiver, pharmacist, doctor, or clinic",
            caregiver_message="The user is unsure about medication. Please help confirm the label, schedule, or pharmacist instructions.",
        )

    # Self-harm language always requires immediate human support.
    if _contains(text, ["kill myself", "hurt myself", "end my life", "suicide"]):
        _set_alert_if_not_downgrade(
            alert,
            alert_type="emergency",
            level="high",
            reason="Self-harm language requires immediate human support.",
            contact="emergency services, crisis line, or trusted nearby person",
            caregiver_message="The user used self-harm language. Please get immediate human support and do not leave them alone.",
        )

    # Keep raw signal booleans for explainability panels.
    alert["signals"] = {
        "dizzy_now": dizzy_now,
        "imminent_fall": imminent_fall,
        "explicit_caregiver_request": explicit_caregiver_request,
        "emergency_red_flag": emergency_red_flag,
    }
    return alert


# Apply final response safety checks before the message is saved or returned.
def apply_output_guardrails(
    response_text: str,
    *,
    alert_decision: dict[str, Any],
    router_decision: dict[str, Any],
) -> dict[str, Any]:
    # Track guardrail findings and whether a safe replacement is needed.
    issues: list[str] = []
    fallback_used = False
    fallback_name = ""
    final = response_text
    lower = response_text.lower()
    active_topic = str(router_decision.get("active_topic", "")).lower()
    agents = " ".join(router_decision.get("activated_agents", [])).lower()
    # Fraud context can come from topic, agent, or detected signal.
    fraud_context = "safety" in active_topic or "fraud" in active_topic or "safety_agent" in agents or any(
        signal in {"email", "link", "bank", "password", "money", "transfer", "otp", "police"}
        for signal in router_decision.get("detected_signals", [])
    )

    # Replace vague fraud advice with concrete verification guidance.
    if fraud_context and any(phrase in lower for phrase in VAGUE_SAFETY_PHRASES):
        issues.append("vague_safety_advice")
        final = CONCRETE_FRAUD_VERIFICATION_GUIDANCE
        fallback_used = True
        fallback_name = "concrete_fraud_verification_guidance"

    # If text promises caregiver contact without an alert, replace it safely.
    caregiver_action = any(phrase in lower for phrase in CAREGIVER_ACTION_PHRASES)
    if caregiver_action and not alert_decision.get("alert_required", False):
        issues.append("alert_decision_response_mismatch")
        final = (
            "A caregiver contact may be helpful, but the alert policy did not mark this as an alert. "
            "If you feel unsafe or need help now, contact a trusted person or local emergency services. "
            "Otherwise, tell me what you want the caregiver to know and I can help prepare a message."
        )
        fallback_used = True
        fallback_name = "caregiver_alert_mismatch_safe_response"

    # Record unsafe instructions so the trace can flag the response.
    if any(term in lower for term in ["take an extra dose", "double your dose", "change your dose"]):
        issues.append("unsafe_medication_instruction")
    if any(term in lower for term in ["share your password", "send the money now", "click the unknown link"]):
        issues.append("unsafe_safety_instruction")

    # Return a small result object that guardrails_node can merge into ChatResponse.
    source = "output_guardrail_replacement" if fallback_used else ""
    return {
        "final_message": final,
        "guardrail_issues": issues,
        "fallback_used": fallback_used,
        "fallback_name": fallback_name,
        "final_message_source": source,
    }


# Normalize trace data for the technical trace UI.
def _normalized_trace(state: ElderGuardState) -> dict[str, Any]:
    # Sum node durations and gather the latest routing/alert metadata.
    spans = list(state.get("langgraph_trace", []))
    total_duration = round(sum(float(span.get("duration_ms", 0) or 0) for span in spans), 2)
    router = state.get("router_decision", {})
    alert = state.get("alert_decision", {})
    coordinated = state.get("coordinated", {})
    graph_context = state.get("memory_context", {}).get("graph_context", {})
    return {
        "trace_id": state.get("trace_id", ""),
        "total_duration_ms": total_duration,
        "selected_agent": router.get("selected_agent", ""),
        "risk": router.get("priority", "low"),
        "llm_calls_this_turn": coordinated.get("llm_calls_this_turn", get_llm_call_counter()),
        "llm_provider": coordinated.get("llm_provider", ""),
        "llm_model": coordinated.get("llm_model", ""),
        "llm_caller": coordinated.get("llm_caller", ""),
        "rag_sources": graph_context.get("detected_nodes", []),
        "alert_status": alert.get("alert_type", "none"),
        "guardrail_status": "warning"
        if state.get("raw_json", {}).get("guardrail_issues")
        else "success",
        "spans": spans,
    }


# Build context from memory and graph reasoning before routing.
def memory_context_builder_node(state: ElderGuardState) -> ElderGuardState:
    # Vietnamese note: Memory loading + GraphRAG chuan bi ngu canh truoc khi router chon agent.
    request = state["request"]
    # Prefer sanitized client history when provided by the API caller.
    client_history = [
        {"role": item.get("role", ""), "message": item.get("message") or item.get("content", "")}
        for item in request.client_history[-8:]
        if item.get("role") in {"user", "assistant"} and (item.get("message") or item.get("content"))
    ]
    stored_history = get_recent_conversation(request.user_id, request.conversation_id, limit=8)
    history = client_history if client_history else stored_history
    # Keep the previous assistant answer to detect follow-up questions.
    previous_assistant = ""
    for item in reversed(history):
        if item.get("role") == "assistant":
            previous_assistant = str(item.get("message", ""))
            break

    # Graph reasoning adds explainable safety paths from local rules.
    graph_context = get_graph_reasoning(request.message)
    memory_context = {
        "current_message": request.message,
        "recent_history": history[-6:],
        "previous_assistant_message": previous_assistant,
        "user_summary": get_user_summary(request.user_id),
        "user_profile": get_user_profile(request.user_id),
        "recent_care_events": get_relevant_care_events(request.user_id, limit=5),
        "graph_context": graph_context,
    }
    return _trace(state, "memory_context_builder", memory_context=memory_context)


# Rule-based router that prioritizes safety and multi-intent messages.
def rule_router_node(state: ElderGuardState) -> ElderGuardState:
    # Vietnamese note: Hybrid router bat dau bang luat an toan de tranh bo sot rui ro.
    message = state["request"].message
    text = message.lower()
    history_text = " ".join(
        str(item.get("message", "")).lower()
        for item in state.get("memory_context", {}).get("recent_history", [])[-4:]
    )
    prior_email_topic = "email" in history_text or "link" in history_text or "click" in history_text
    activated: list[str] = []
    topics: list[str] = []
    signals: list[str] = []
    ## issue: how to constantly update the terms with user query? the term database should be built up to date 
    safety_terms = ["email", "link", "click", "login", "password", "otp", "bank", "money", "transfer", "police", "scam", "fall", "fell", "chest pain", "can't breathe", "emergency"]
    health_terms = ["dizzy", "weak", "medicine", "medication", "pill", "dose", "hungry", "food", "banana", "juice", "nutrition", "empty stomach", "walk", "mobility", "pain", "fever"]
    emotional_terms = ["lonely","alone","sad","worried","worry","afraid","scared","fear","nervous","anxious","stress","stressed","miss","daughter","son","family","children","friend","support"]
    action_terms = ["write", "draft", "message", "checklist", "reminder", "script", "caregiver note", "what should i say"]

    # Safety rules run first so fraud or emergency terms are not missed.
    if _contains(text, safety_terms) or (
        prior_email_topic
        and _contains(text, ["daughter", "friend", "log in", "login", "money", "download", "password", "yes", "no"])
    ):
        activated.append("safety_agent")
        if "email" in text or "link" in text or "click" in text or prior_email_topic:
            topics.append("suspicious_email")
        elif "fall" in text or "fell" in text:
            topics.append("health_symptom")
        else:
            topics.append("safety")
        signals.extend([term for term in safety_terms if term in text][:5] or ["email_follow_up"])
    # Health and daily-care rules catch symptoms, food needs, and medication issues.
    if _contains(text, health_terms):
        activated.append("health_daily_care_agent")
        if _contains(text, ["medicine", "medication", "pill", "dose"]):
            topics.append("medication")
        elif _contains(text, ["hungry", "food", "banana", "juice", "nutrition", "empty stomach"]):
            topics.append("basic_need_support")
        elif _contains(text, ["dizzy", "weak", "pain", "fever", "walk", "mobility"]):
            topics.append("health_symptom")
        else:
            topics.append("health_daily_care")
        signals.extend([term for term in health_terms if term in text][:5])
    # Emotional rules cover loneliness and family-support language.
    if _contains(text, emotional_terms):
        activated.append("emotional_social_agent")
        topics.append("emotional_support" if _contains(text, ["lonely", "sad", "worried", "stress", "stressed"]) else "emotional_social")
        signals.extend([term for term in emotional_terms if term in text][:5])
    # Action rules route drafting/checklist requests to the action agent.
    if _contains(text, action_terms):
        activated.append("action_agent")
        topics.append("action")
        signals.extend([term for term in action_terms if term in text][:5])
    # Default to daily-care support for ordinary messages.
    if not activated:
        activated = ["health_daily_care_agent"]
        topics = ["general_daily_life"]

    # Risk is intentionally simple and explainable for the local router.
    high_risk = _contains(text, ["chest pain", "can't breathe", "fainted", "send money", "password", "otp", "fell"])
    medium_risk = _contains(text, ["dizzy", "medicine", "medication", "email", "link", "money", "hungry"])
    router_decision = {
        "active_topic": topics[0],
        "activated_agents": list(dict.fromkeys(activated)),
        "selected_agent": list(dict.fromkeys(activated))[0],
        "detected_signals": list(dict.fromkeys(signals)),
        "priority": "high" if high_risk else "medium" if medium_risk else "low",
        "reason": "Local router selected broad agents from latest user message.",
        "router_source": "rules",
    }
    return _trace(state, "router", router_decision=router_decision)


# Read the feature flag that enables the optional ML router.
def ml_router_is_enabled() -> bool:
    return os.getenv("USE_ML_ROUTER", "false").strip().lower() in {"1", "true", "yes", "on"}


# Map graph route names to general topic labels.
GRAPH_ROUTE_TO_TOPIC = {
    "safety_agent": "safety",
    "health_daily_care_agent": "health_daily_care",
    "emotional_social_agent": "emotional_support",
    "action_agent": "action",
}


def _update_router_state(
    rule_state: ElderGuardState,
    router_decision: dict[str, Any],
) -> ElderGuardState:
    """
    Update router_decision and keep the router trace consistent.
    """

    # Replace router_decision in state without disturbing earlier node outputs.
    updated_state: ElderGuardState = {
        **rule_state,
        "router_decision": router_decision,
    }

    trace = list(
        updated_state.get(
            "langgraph_trace",
            [],
        )
    )

    # Patch the latest trace span so it matches the final router choice.
    if trace:
        latest_trace = trace[-1]

        latest_trace["active_topic"] = (
            router_decision.get(
                "active_topic",
                "",
            )
        )

        latest_trace["activated_agents"] = (
            router_decision.get(
                "activated_agents",
                [],
            )
        )

        attributes = latest_trace.setdefault(
            "attributes",
            {},
        )

        attributes["router_decision"] = (
            router_decision
        )

        attributes["active_topic"] = (
            router_decision.get(
                "active_topic",
                "",
            )
        )

        attributes["selected_agent"] = (
            router_decision.get(
                "selected_agent",
                "",
            )
        )

        attributes["activated_agents"] = (
            router_decision.get(
                "activated_agents",
                [],
            )
        )

        attributes["router_source"] = (
            router_decision.get(
                "router_source",
                "",
            )
        )

        attributes["ml_candidate"] = (
            router_decision.get(
                "ml_candidate",
                "",
            )
        )

    updated_state["langgraph_trace"] = trace

    return updated_state


# Attach the human-readable and structured XAI routing explanation.
def _attach_xai_decision(
    router_decision: dict[str, Any],
    *,
    rule_route: str,
    final_route: str,
    router_source: str,
    explanation_text: str,
    ml_explanation: dict[str, Any] | None = None,
    error_message: str = "",
) -> dict[str, Any]:
    """
    Attach a structured explanation to router_decision.
    """

    # Use an empty object when the ML router did not run.
    ml_data = ml_explanation or {}

    # Store all routing candidates and explanation terms under one xai key.
    router_decision["xai"] = {
        "rule_candidate": rule_route,
        "ml_candidate": str(
            ml_data.get(
                "graph_route",
                "",
            )
        ),
        "model_label": str(
            ml_data.get(
                "model_label",
                "",
            )
        ),
        "alternative_model_label": str(
            ml_data.get(
                "alternative_model_label",
                "",
            )
        ),
        "alternative_graph_route": str(
            ml_data.get(
                "alternative_graph_route",
                "",
            )
        ),
        "final_route": final_route,
        "final_agents": list(
            router_decision.get(
                "activated_agents",
                [],
            )
        ),
        "router_source": router_source,
        "decision_margin": ml_data.get(
            "decision_margin"
        ),
        "margin_is_probability": False,
        "supporting_terms": list(
            ml_data.get(
                "supporting_terms",
                [],
            )
        ),
        "opposing_terms": list(
            ml_data.get(
                "opposing_terms",
                [],
            )
        ),
        "explanation_text": (
            explanation_text
        ),
        "error_message": error_message,
    }

    return router_decision


# Hybrid router that combines safety rules with optional ML predictions.
def router_node(
    state: ElderGuardState,
) -> ElderGuardState:
    """
    Explainable hybrid router.

    Rules handle:
    1. High risk situations
    2. Action requests
    3. Multiple agents
    4. Strong contextual evidence

    ML handles normal single agent routing.
    """

    # Vietnamese note: Rule-based override uu tien khi co rui ro cao hoac nhieu intent.
    # Always run the rule router first so safety evidence is available.
    rule_state = rule_router_node(
        state
    )

    router_decision = dict(
        rule_state.get(
            "router_decision",
            {},
        )
    )

    rule_agents = list(
        router_decision.get(
            "activated_agents",
            [],
        )
    )

    rule_route = str(
        router_decision.get(
            "selected_agent",
            "",
        )
    )

    priority = str(
        router_decision.get(
            "priority",
            "low",
        )
    )

    active_topic = str(
        router_decision.get(
            "active_topic",
            "",
        )
    )

    detected_signals = list(
        router_decision.get(
            "detected_signals",
            [],
        )
    )

    # If ML is disabled, keep rule routing and attach an XAI explanation.
    if not ml_router_is_enabled():
        router_decision[
            "router_source"
        ] = "rules"

        router_decision = (
            _attach_xai_decision(
                router_decision,
                rule_route=rule_route,
                final_route=rule_route,
                router_source="rules",
                explanation_text=(
                    "The ML router is disabled. "
                    "The local rule router selected "
                    "the final specialist."
                ),
            )
        )

        logger.info(
            "Router source=rules "
            "route=%s",
            rule_route,
        )

        return _update_router_state(
            rule_state,
            router_decision,
        )

    # Try ML routing only after rule evidence has been collected.
    try:
        user_text = (
            extract_latest_user_text(
                state
            )
        )

        ml_explanation = (
            explain_ml_route(
                user_text
            )
        )

        ml_route = str(
            ml_explanation[
                "graph_route"
            ]
        )

        router_decision[
            "ml_candidate"
        ] = ml_route

        # Rules win when safety, action, or multi-agent evidence is present.
        must_keep_rules = (
            priority == "high"
            or "action_agent"
            in rule_agents
            or len(rule_agents) > 1
        )

        if must_keep_rules:
            router_source = (
                "rules_override"
            )

            reason = (
                "Safety and workflow rules "
                "had priority because the message "
                "contained high risk, an action "
                "request, or multiple relevant agents."
            )

            router_decision[
                "router_source"
            ] = router_source

            router_decision[
                "reason"
            ] = reason

            router_decision = (
                _attach_xai_decision(
                    router_decision,
                    rule_route=rule_route,
                    final_route=rule_route,
                    router_source=(
                        router_source
                    ),
                    explanation_text=reason,
                    ml_explanation=(
                        ml_explanation
                    ),
                )
            )

            logger.info(
                "Router source=rules_override "
                "rule=%s ml=%s",
                rule_agents,
                ml_route,
            )

            return _update_router_state(
                rule_state,
                router_decision,
            )

        # Clear rule evidence wins over a conflicting ML-only route.
        rule_has_clear_evidence = (
            bool(detected_signals)
            and active_topic
            != "general_daily_life"
        )

        if (
            rule_has_clear_evidence
            and ml_route != rule_route
        ):
            router_source = (
                "rules_ml_disagreement"
            )

            reason = (
                "The ML model and rule router "
                "disagreed. The rule route was "
                "retained because it found clear "
                "message or conversation evidence."
            )

            router_decision[
                "router_source"
            ] = router_source

            router_decision[
                "reason"
            ] = reason

            router_decision = (
                _attach_xai_decision(
                    router_decision,
                    rule_route=rule_route,
                    final_route=rule_route,
                    router_source=(
                        router_source
                    ),
                    explanation_text=reason,
                    ml_explanation=(
                        ml_explanation
                    ),
                )
            )

            logger.warning(
                "Router disagreement "
                "rule=%s ml=%s signals=%s",
                rule_route,
                ml_route,
                detected_signals,
            )

            return _update_router_state(
                rule_state,
                router_decision,
            )

        # Low-risk single-intent messages can use the ML-selected route.
        router_decision[
            "selected_agent"
        ] = ml_route

        router_decision[
            "activated_agents"
        ] = [ml_route]

        router_decision[
            "ml_route"
        ] = ml_route

        if ml_route == rule_route:
            router_source = (
                "ml_confirmed"
            )

            reason = (
                "The ML model and rule router "
                "selected the same specialist."
            )

        else:
            router_source = "ml"

            router_decision[
                "active_topic"
            ] = GRAPH_ROUTE_TO_TOPIC.get(
                ml_route,
                "general_daily_life",
            )

            reason = (
                "The rule router found no clear "
                "signal, so the ML model selected "
                "the specialist."
            )

        # Store the final ML or ML-confirmed explanation.
        router_decision[
            "router_source"
        ] = router_source

        router_decision[
            "reason"
        ] = reason

        router_decision = (
            _attach_xai_decision(
                router_decision,
                rule_route=rule_route,
                final_route=ml_route,
                router_source=router_source,
                explanation_text=reason,
                ml_explanation=(
                    ml_explanation
                ),
            )
        )

        logger.info(
            "Router source=%s route=%s",
            router_source,
            ml_route,
        )

        return _update_router_state(
            rule_state,
            router_decision,
        )

    except Exception as error:
        # Any ML failure falls back to the already-computed rule route.
        router_source = (
            "rules_fallback"
        )

        reason = (
            "The ML router failed, so the "
            "rule router result was retained."
        )

        router_decision[
            "router_source"
        ] = router_source

        router_decision[
            "ml_router_error"
        ] = str(error)

        router_decision[
            "reason"
        ] = reason

        router_decision = (
            _attach_xai_decision(
                router_decision,
                rule_route=rule_route,
                final_route=rule_route,
                router_source=router_source,
                explanation_text=reason,
                error_message=str(error),
            )
        )

        logger.exception(
            "ML router failed. "
            "Using rule router fallback."
        )

        return _update_router_state(
            rule_state,
            router_decision,
        )

# Build a normalized specialist evidence object.
def _agent_result(agent: str, risk: str, signals: list[str], actions: list[str], avoid: list[str], missing: list[str]) -> dict[str, Any]:
    return {
        "agent": agent,
        "risk": risk,
        "signals": signals,
        "known_facts": [],
        "missing_facts": missing,
        "recommended_actions": actions,
        "actions_to_avoid": avoid,
        "confidence": 0.75,
    }


# Create structured evidence for each broad specialist selected by the router.
def broad_agent_reasoning_node(state: ElderGuardState) -> ElderGuardState:
    # Vietnamese note: Specialist reasoning chi tao bang chung co cau truc, khong noi truc tiep voi user.
    message = state["request"].message
    text = message.lower()
    router = state["router_decision"]
    outputs: dict[str, dict[str, Any]] = {}

    # Safety evidence focuses on fraud, scams, and emergency risks.
    if "safety_agent" in router["activated_agents"]:
        risk = "high" if _contains(text, ["password", "otp", "send money", "transfer", "chest pain", "can't breathe", "fell"]) else "medium"
        outputs["safety_agent"] = _agent_result(
            "safety_agent",
            risk,
            [s for s in router["detected_signals"] if s in text],
            ["pause before risky action", "verify using a separate trusted channel", "seek urgent human help for severe symptoms"],
            ["do not click unknown links", "do not share passwords or OTPs", "do not send money under pressure"],
            ["who sent the message", "whether the request was expected"],
        )
    # Health and daily-care evidence covers symptoms, food, and medication.
    if "health_daily_care_agent" in router["activated_agents"]:
        risk = "medium" if _contains(text, ["dizzy", "medicine", "medication", "fell", "pain"]) else "low"
        outputs["health_daily_care_agent"] = _agent_result(
            "health_daily_care_agent",
            risk,
            [s for s in router["detected_signals"] if s in text],
            ["answer the daily care question directly", "ask one practical follow-up if needed"],
            ["do not diagnose", "do not prescribe", "do not change medication dosage"],
            ["exact timing or available items if needed"],
        )
    # Emotional evidence keeps support gentle and scoped.
    if "emotional_social_agent" in router["activated_agents"]:
        outputs["emotional_social_agent"] = _agent_result(
            "emotional_social_agent",
            "low",
            [s for s in router["detected_signals"] if s in text],
            ["acknowledge the feeling", "suggest one small social support step"],
            ["do not pretend to be a therapist", "do not give unrelated safety warnings"],
            ["whether the user wants help contacting someone"],
        )
    # Action evidence tells the response node to fulfill clear requests.
    if "action_agent" in router["activated_agents"]:
        outputs["action_agent"] = _agent_result(
            "action_agent",
            "low",
            [s for s in router["detected_signals"] if s in text],
            ["produce the requested message, checklist, reminder, note, or call script"],
            ["do not ask again when the requested action is clear"],
            ["recipient or purpose if unclear"],
        )
    return _trace(state, "broad_agent_reasoning", broad_agent_outputs=outputs)


# Prepare an alert decision from the latest user message.
def alert_decision_node(state: ElderGuardState) -> ElderGuardState:
    # Vietnamese note: Alert decision chi chuan bi canh bao; prototype khong gui thong bao that.
    return _trace(state, "alert_decision", alert_decision=evaluate_alert_decision(state["request"].message))


# Generate a deterministic response when the app runs without a live LLM.
def _mock_conversation_reply(state: ElderGuardState) -> dict[str, Any]:
    message = state["request"].message
    text = message.lower()
    alert = state.get("alert_decision", {})
    # Alert-required messages get the safest response branch first.
    if alert.get("alert_required"):
        if alert.get("alert_type") == "health" and "dizzy" in text:
            final = "Please sit or lie down now. Since you feel dizzy after standing, I recommend asking someone nearby to check on you. If you have chest pain, trouble breathing, fainting, confusion, or you fall, call emergency services."
        elif alert.get("alert_type") == "fraud":
            final = "Please do not share your bank password, one-time code, account details, or any money information. A real bank or official service should not pressure you to give a password. End the call or message, then contact the bank using the official phone number or app. If you are unsure, ask a trusted family member to check it with you."
        else:
            final = f"This may need help from {alert.get('recommended_contact', 'a trusted person')}. {alert.get('caregiver_message', '')}"
    elif "banana" in text and "empty stomach" in text:
        final = "A banana is usually a gentle food for many people, but if your stomach feels upset, start with a small amount and some water. Do you feel hungry, nauseous, or dizzy right now?"
    elif "email" in text or "link" in text:
        final = "Do not click the link yet. Please check who sent it and whether you expected it before opening anything."
    elif "password" in text or "bank" in text or "otp" in text:
        final = "Please do not share your bank password, OTP, or account information. If someone is asking for those details, stop the conversation and use the official bank number or app to check. Would you like help writing a short message to a family member to verify it?"
    elif "dizzy" in text or "dizziness" in text or "lightheaded" in text:
        final = "I’m sorry you feel dizzy. Please sit or lie down somewhere safe and avoid walking alone for the moment. Did this happen after standing up, and do you also have chest pain, trouble breathing, fainting, confusion, or a fall?"
    elif "lonely" in text or "alone" in text or "sad" in text:
        final = "I’m sorry you’re feeling lonely. That can feel heavy, especially when friends or family are busy. You do not have to handle the feeling alone. Would you like me to help you write a short message asking someone to check in with you today?"
    elif "medicine" in text or "medication" in text or "pill" in text or "dose" in text:
        final = "If you are unsure about medicine, please do not take an extra dose or change the dosage on your own. Check the label if you can, and ask a pharmacist, doctor, clinic, or caregiver to confirm the safest next step."
    elif "hungry" in text:
        final = "I’m sorry you’re hungry. What simple food do you have nearby, such as bread, rice, soup, eggs, banana, or crackers?"
    else:
        final = "I’m here with you. Tell me one detail about what you need, and I’ll help with the next step."
    return {
        "final_message": final,
        "xai_simple": "Limited mock mode used because no selected LLM key is configured.",
        "follow_up_questions": [],
        "final_message_source": "mock_specialist",
        "active_mode": "Mock",
    }


# Generate the final natural-language answer using mock or live LLM mode.
def conversation_agent_node(state: ElderGuardState) -> ElderGuardState:
    # Vietnamese note: Conversation agent la noi duy nhat viet phan hoi tu nhien cho nguoi dung.
    settings = get_settings()
    memory_context = state["memory_context"]
    router = state["router_decision"]
    broad_outputs = state["broad_agent_outputs"]
    selected_agent = str(router.get("selected_agent", "health_daily_care_agent"))
    prompt_name = f"{selected_agent}_prompt"
    prompt_version = "v2"
    coordinated: dict[str, Any]

    # Mock mode keeps local tests deterministic and avoids network calls.
    if settings.llm_mode == "mock":
        coordinated = _mock_conversation_reply(state)
        coordinated["prompt_name"] = prompt_name
        coordinated["prompt_version"] = prompt_version
        coordinated["raw_llm_output"] = ""
    else:
        # The system prompt constrains the LLM to one safe JSON response.
        system = (
            "You are ElderGuard AI, one warm care companion. "
            f"You are responding as the selected specialist: {selected_agent}. "
            "Use the structured evidence silently and write one natural answer. "
            "Do not diagnose, prescribe, or use unrelated safety warnings. "
            "For fraud, scam, impersonation, suspicious calls, suspicious emails, or links: do not use vague advice such as "
            "'trust your instincts', 'follow your gut', or 'use your best judgment'. Always provide concrete verification steps. "
            "Ask at most one useful follow-up question. "
            "Return only JSON with final_message, xai_simple, follow_up_questions."
        )
        # The user payload gives the model structured evidence instead of raw internals.
        user_payload = {
            "current_message": state["request"].message,
            "recent_history": memory_context.get("recent_history", [])[-6:],
            "memory_context": {
                "user_summary": memory_context.get("user_summary", ""),
                "user_profile_confirmed_facts": memory_context.get("user_profile", {}),
                "previous_assistant_message": memory_context.get("previous_assistant_message", ""),
                "graph_context": memory_context.get("graph_context", {}),
            },
            "router_decision": router,
            "broad_agent_outputs": broad_outputs,
            "alert_decision": state.get("alert_decision", {}),
            "instruction": (
                "Include alert information in final_message only if alert_decision.alert_required is true. "
                "Only treat confirmed profile facts as truth. If a remembered fact is not confirmed, say 'I may be mistaken' or ask the user."
            ),
        }
        try:
            # This is the only live LLM call path for the conversation response.
            raw = call_openrouter_llm(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
                ],
                temperature=0.4,
            )
            data = _json_from_text(raw)
            # Keep only the JSON fields the rest of the graph expects.
            coordinated = {
                "final_message": str(data.get("final_message", "")).strip(),
                "xai_simple": str(data.get("xai_simple", "")).strip(),
                "follow_up_questions": list(data.get("follow_up_questions", []) or []),
                "final_message_source": "openrouter_specialist",
                "active_mode": settings.llm_mode,
                "raw_llm_output": raw,
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
            }
            if not coordinated["final_message"]:
                raise RuntimeError(f"Conversation Agent returned empty final_message. Raw output: {raw[:1000]}")
        except Exception as exc:
            # Save full provider errors locally but return only safe details.
            local_error = {
                "error_type": type(exc).__name__,
                "raw_error": str(exc),
                "traceback": traceback.format_exc(),
                "llm_provider": settings.llm_provider,
                "llm_model": settings.active_llm_model,
                "llm_mode": settings.llm_mode,
                "llm_calls_this_turn": get_llm_call_counter(),
            }
            safe_error = {
                "error_type": type(exc).__name__,
                "raw_error": str(exc),
                "llm_provider": settings.llm_provider,
                "llm_model": settings.active_llm_model,
                "llm_mode": settings.llm_mode,
                "llm_calls_this_turn": get_llm_call_counter(),
            }
            Path("data").mkdir(exist_ok=True)
            Path("data/conversation_agent_error.txt").write_text(json.dumps(local_error, ensure_ascii=False, indent=2), encoding="utf-8")
            coordinated = {
                "final_message": "I am having trouble generating a full response right now. Please try again in a moment.",
                "xai_simple": "The selected LLM provider failed. See Developer Console for the full error.",
                "follow_up_questions": [],
                "final_message_source": "safe_service_failure",
                "active_mode": "LLMError",
                "llm_error": safe_error,
                "raw_llm_output": "",
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
            }

    # Attach runtime metadata for the technical trace.
    coordinated["llm_calls_this_turn"] = get_llm_call_counter()
    coordinated["llm_provider"] = settings.llm_provider
    coordinated["llm_model"] = settings.active_llm_model
    coordinated["llm_mode"] = settings.llm_mode
    coordinated["llm_caller"] = selected_agent
    return _trace(state, "conversation_agent", coordinated=coordinated)


# Parse JSON even if a model wraps it with extra text.
def _json_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


# Convert the coordinated answer into a ChatResponse after safety checks.
def guardrails_node(state: ElderGuardState) -> ElderGuardState:
    # Vietnamese note: Guardrails kiem tra phan hoi cuoi truoc khi luu memory va tra ve UI.
    coordinated = dict(state["coordinated"])
    response_before_guardrail = str(coordinated.get("final_message", ""))
    # Run output guardrails before the response is exposed or saved.
    guardrail_result = apply_output_guardrails(
        response_before_guardrail,
        alert_decision=state.get("alert_decision", {}),
        router_decision=state.get("router_decision", {}),
    )
    final = str(guardrail_result["final_message"])
    issues: list[str] = list(guardrail_result["guardrail_issues"])
    llm_calls = int(coordinated.get("llm_calls_this_turn", get_llm_call_counter()) or 0)
    # Enforce the one-call-per-turn budget in the trace.
    if llm_calls > 1:
        issues.append("llm_call_budget_exceeded")
    response_after_guardrail = final
    fallback_used = bool(guardrail_result["fallback_used"])
    fallback_name = str(guardrail_result["fallback_name"])
    final_message_source = str(guardrail_result["final_message_source"] or coordinated.get("final_message_source", "response_coordinator"))
    coordinated["final_message"] = final
    coordinated["final_message_source"] = final_message_source

    # Combine broad-agent risks with router priority.
    risks = [output.get("risk", "low") for output in state.get("broad_agent_outputs", {}).values()]
    overall_risk = _risk_max(*risks, state.get("router_decision", {}).get("priority", "low"))
    care_plan = {
        "router_decision": state.get("router_decision", {}),
        "broad_agent_outputs": state.get("broad_agent_outputs", {}),
        "alert_decision": state.get("alert_decision", {}),
        "guardrail_issues": issues,
    }
    # Developer state is verbose because it powers trace and family views.
    developer_state = {
        "llm_provider": coordinated.get("llm_provider", ""),
        "llm_model": coordinated.get("llm_model", ""),
        "llm_mode": coordinated.get("llm_mode", ""),
        "llm_caller": coordinated.get("llm_caller", ""),
        "llm_calls_this_turn": llm_calls,
        "prompt_name": coordinated.get("prompt_name", ""),
        "prompt_version": coordinated.get("prompt_version", ""),
        "raw_llm_output": coordinated.get("raw_llm_output", ""),
        "response_before_guardrail": response_before_guardrail,
        "response_after_guardrail": response_after_guardrail,
        "guardrail_issues": issues,
        "fallback_used": fallback_used,
        "fallback_name": fallback_name,
        "final_message_source": final_message_source,
        "langgraph_path": LANGGRAPH_PATH,
        "langgraph_trace": state.get("langgraph_trace", []),
        "active_topic": state.get("router_decision", {}).get("active_topic", ""),
        "activated_agents": state.get("router_decision", {}).get("activated_agents", []),
        "router_decision": state.get("router_decision", {}),
        "alert_decision": state.get("alert_decision", {}),
        "llm_error": coordinated.get("llm_error", {}),
        "memory_policy": "Facts are classified before durable save. Only confirmed profile facts are treated as truth.",
        "guardrails": {"passed": not issues, "issues": issues},
    }
    # Build the public response model expected by API and Streamlit callers.
    response = ChatResponse(
        final_message=final,
        active_mode=coordinated.get("active_mode", "Mock"),
        final_message_source=final_message_source,
        current_topic=state.get("router_decision", {}).get("active_topic", ""),
        is_follow_up=bool(state.get("memory_context", {}).get("previous_assistant_message")),
        known_information=[],
        follow_up_questions=coordinated.get("follow_up_questions", []),
        specificity_score=75 if not issues else 45,
        missing_information=[],
        hypotheses=[],
        need_more_information=False,
        reasoning_summary="Simplified broad-agent workflow completed.",
        xai_simple=coordinated.get("xai_simple", ""),
        caregiver_summary="",
        activated_agents=state.get("router_decision", {}).get("activated_agents", []),
        overall_risk=overall_risk,  # type: ignore[arg-type]
        main_concern=state.get("router_decision", {}).get("active_topic", ""),
        advice="",
        caregiver_alert=(
            state.get("alert_decision", {}).get("caregiver_message", "")
            if state.get("alert_decision", {}).get("alert_required")
            else ""
        ),
        next_step=(
            f"Contact: {state.get('alert_decision', {}).get('recommended_contact', '')}"
            if state.get("alert_decision", {}).get("alert_required")
            else coordinated.get("follow_up_questions", [""])[0] if coordinated.get("follow_up_questions") else ""
        ),
        memory_id=0,
        human_confirmation_required=bool(state.get("alert_decision", {}).get("human_confirmation_required", overall_risk == "high")),
        confirmation_reason="High-risk safety or health issue." if overall_risk == "high" else "",
        routing_explanation=state.get("router_decision", {}).get("reason", ""),
        graph_reasoning=state.get("memory_context", {}).get("graph_context", {}),
        detected_signals=[
            {"signal": signal, "agent": ",".join(state.get("router_decision", {}).get("activated_agents", [])), "reason": "local router signal"}
            for signal in state.get("router_decision", {}).get("detected_signals", [])
        ],
        risk_scores={agent: {"low": 20, "medium": 60, "high": 90}.get(output.get("risk", "low"), 20) for agent, output in state.get("broad_agent_outputs", {}).items()},
        memory_summary={"user_summary": state.get("memory_context", {}).get("user_summary", "")},
        situation={"memory_context": state.get("memory_context", {})},
        care_plan=care_plan,
        developer_state=developer_state,
    )
    # Keep raw workflow details for persistence and trace artifacts.
    raw_json = {
        "memory_context": state.get("memory_context", {}),
        "router_decision": state.get("router_decision", {}),
        "broad_agent_outputs": state.get("broad_agent_outputs", {}),
        "alert_decision": state.get("alert_decision", {}),
        "coordinated": coordinated,
        "response_before_guardrail": response_before_guardrail,
        "response_after_guardrail": response_after_guardrail,
        "guardrail_issues": issues,
        "fallback_used": fallback_used,
        "fallback_name": fallback_name,
        "developer_state": developer_state,
    }
    return _trace(state, "guardrails", final_response=response, raw_json=raw_json)


# Persist messages, classified memories, logs, and trace data.
def save_memory_node(state: ElderGuardState) -> ElderGuardState:
    # Vietnamese note: Save memory phan loai thong tin truoc khi luu vao SQLite va technical trace.
    request = state["request"]
    response = state["final_response"]
    llm_error = response.developer_state.get("llm_error", {})
    # Classify the user message before saving anything durable.
    classified_memories = classify_memory(
        request.message,
        final_message_source=response.final_message_source,
        llm_error=llm_error,
        router_decision=state.get("router_decision", {}),
        alert_decision=state.get("alert_decision", {}),
    )
    # Always save the user message, but skip assistant text on LLM errors.
    save_message(request.user_id, request.conversation_id, "user", request.message)
    if not llm_error:
        save_message(request.user_id, request.conversation_id, "assistant", response.final_message)

    # Save only memory items that the classifier allows.
    for memory in classified_memories:
        if memory.get("memory_type") == "do_not_save":
            continue
        save_care_event(
            request.user_id,
            request.conversation_id,
            memory.get("memory_type", "episodic_memory"),
            {
                "classified_memory": memory,
                "message": request.message,
                "activated_agents": response.activated_agents,
                "overall_risk": response.overall_risk,
            },
        )

    # Confirmed long-term memories update the durable user profile.
    profile_updates = profile_updates_from_memories(classified_memories)
    if profile_updates and not llm_error:
        profile = get_user_profile(request.user_id)
        for key, values in profile_updates.items():
            profile.setdefault(key, [])
            profile[key].extend(values)
        save_user_profile(request.user_id, profile)

    # Write trace artifacts and the admin interaction log.
    raw_json = {**state.get("raw_json", {}), "langgraph_trace": state.get("langgraph_trace", [])}
    raw_json["classified_memories"] = classified_memories
    _write_trace_artifact(request, raw_json)
    response.memory_id = save_interaction(request, response, raw_json)
    response.developer_state["classified_memories"] = classified_memories
    if not llm_error:
        # Update the compact text summary with confirmed durable facts.
        summary_items = [
            f"{item['memory_type']}: {item['content']}"
            for item in classified_memories
            if item.get("memory_type") in {"episodic_memory", "long_term_profile"} and item.get("confirmed")
        ]
        if summary_items:
            update_user_summary(request.user_id, " ".join(summary_items)[:500])
    final_state = _trace(state, "save_memory", final_response=response)
    # Attach final trace and memory status back onto the response.
    response.developer_state["langgraph_trace"] = final_state.get("langgraph_trace", [])
    response.developer_state["trace"] = _normalized_trace(final_state)
    response.developer_state["memory_saved"] = True
    return final_state


# Write the latest trace to data/traces for local debugging.
def _write_trace_artifact(request: ChatRequest, raw_json: dict[str, Any]) -> None:
    # Vietnamese note: Technical trace giup developer giai thich workflow, khong hien o elder view.
    trace_dir = Path("data/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_user = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in request.user_id)
    safe_conversation = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in request.conversation_id)
    # Store both a timestamped file and last_trace.json for quick inspection.
    payload = {
        "timestamp": stamp,
        "user_id": request.user_id,
        "conversation_id": request.conversation_id,
        "message": request.message,
        "trace": raw_json,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    (trace_dir / f"{stamp}_{safe_user}_{safe_conversation}.json").write_text(text, encoding="utf-8")
    (trace_dir / "last_trace.json").write_text(text, encoding="utf-8")


# Build and compile the LangGraph workflow.
def build_graph():
    graph = StateGraph(ElderGuardState)
    # Register each node by the name used in the trace.
    graph.add_node("memory_context_builder", memory_context_builder_node)
    graph.add_node("router", router_node)
    graph.add_node("broad_agent_reasoning", broad_agent_reasoning_node)
    graph.add_node("alert_decision_step", alert_decision_node)
    graph.add_node("conversation_agent", conversation_agent_node)
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("save_memory", save_memory_node)

    # Wire the graph in the fixed workflow order.
    graph.set_entry_point("memory_context_builder")
    graph.add_edge("memory_context_builder", "router")
    graph.add_edge("router", "broad_agent_reasoning")
    graph.add_edge("broad_agent_reasoning", "alert_decision_step")
    graph.add_edge("alert_decision_step", "conversation_agent")
    graph.add_edge("conversation_agent", "guardrails")
    graph.add_edge("guardrails", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()


# Compile once at import time so requests can invoke the graph quickly.
elderguard_graph = build_graph()


# Run one ElderGuard workflow turn and return the final response.
def run_elderguard_workflow(request: ChatRequest) -> ChatResponse:
    # Reset per-turn LLM accounting before the graph starts.
    reset_llm_call_counter()
    try:
        # Seed trace timing fields and invoke the compiled graph.
        start = time.perf_counter()
        result = elderguard_graph.invoke(
            {
                "request": request,
                "langgraph_trace": [],
                "trace_id": str(uuid4()),
                "trace_started_perf": start,
                "trace_last_perf": start,
            },
            {"recursion_limit": 15},
        )
        return result["final_response"]
    except Exception as exc:
        # If the graph itself fails, return a safe response and log details locally.
        try:
            settings = get_settings()
            llm_mode = settings.llm_mode
            llm_provider = settings.llm_provider
            llm_model = settings.active_llm_model
        except Exception:
            llm_mode = "live"
            llm_provider = "openrouter"
            llm_model = "unknown"
        text = request.message.lower()
        emergency_hint = ""
        # Preserve emergency guidance even during backend failures.
        if any(term in text for term in ["chest pain", "can't breathe", "trouble breathing", "fainted", "confusion", "hurt myself", "kill myself"]):
            emergency_hint = " If this may be urgent, please contact emergency services or a trusted nearby person now."
        local_error = {
            "error_type": type(exc).__name__,
            "raw_error": str(exc),
            "traceback": traceback.format_exc(),
            "llm_calls_this_turn": get_llm_call_counter(),
        }
        safe_error = {
            "error_type": type(exc).__name__,
            "raw_error": str(exc),
            "llm_calls_this_turn": get_llm_call_counter(),
        }
        Path("data").mkdir(exist_ok=True)
        Path("data/last_chat_error.txt").write_text(json.dumps(local_error, ensure_ascii=False, indent=2), encoding="utf-8")
        # Return a full ChatResponse so callers do not need a separate error shape.
        return ChatResponse(
            final_message=f"ElderGuard is having a service problem right now. Please try again in a moment.{emergency_hint}",
            active_mode="Error",
            final_message_source="workflow_error",
            activated_agents=[],
            overall_risk="medium",
            main_concern="Backend workflow error",
            advice="",
            caregiver_alert="",
            next_step="",
            memory_id=0,
            developer_state={
                "llm_mode": llm_mode,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "llm_calls_this_turn": get_llm_call_counter(),
                "langgraph_path": LANGGRAPH_PATH,
                "active_topic": "",
                "activated_agents": [],
                "router_decision": {},
                "llm_error": safe_error,
            },
        )
