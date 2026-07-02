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
from app.ml_router import extract_latest_user_text, predict_route
from app.schemas import ChatRequest, ChatResponse


logger = logging.getLogger(__name__)

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

BROAD_AGENTS = {
    "safety_agent": "Fraud, scam, suspicious email, unsafe pressure, emergency/fall danger.",
    "health_daily_care_agent": "Health, medication, food, nutrition, mobility, and daily needs.",
    "emotional_social_agent": "Loneliness, stress, family support, reassurance.",
    "action_agent": "Write messages, checklists, reminders, caregiver notes, call scripts.",
}

ALERT_LEVEL_ORDER = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}

CONCRETE_FRAUD_VERIFICATION_GUIDANCE = (
    "Do not rely on how convincing the caller sounds. End the call and verify the claim using an official phone number "
    "that you find independently. Do not use a phone number, link, or contact method provided by the caller. Do not "
    "share passwords, one time codes, banking information, or personal information."
)

VAGUE_SAFETY_PHRASES = [
    "trust your instincts",
    "trust your instinct",
    "follow your gut",
    "use your best judgment",
    "listen to your intuition",
]

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


def _trace(state: ElderGuardState, node: str, **updates: Any) -> ElderGuardState:
    next_state: ElderGuardState = {**state, **updates}
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


def _contains(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _risk_max(*risks: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return max((risk if risk in order else "low" for risk in risks), key=lambda item: order[item])


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
    if next_score < current_score:
        if user_requested_contact:
            alert["user_requested_contact"] = True
            alert["human_confirmation_required"] = False
        return
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


def evaluate_alert_decision(message: str) -> dict[str, Any]:
    text = message.lower()
    alert = _alert_template()

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

    if _contains(text, ["expired food", "expired", "old food"]) and _contains(text, ["stomach pain", "stomach hurts", "belly pain", "vomit", "nausea"]):
        _set_alert_if_not_downgrade(
            alert,
            alert_type="health",
            level="medium",
            reason="Expired food with stomach symptoms may need health attention.",
            contact="caregiver or healthcare professional",
            caregiver_message="The user reported expired food and stomach symptoms. Please check on them and consider medical advice if symptoms continue.",
        )

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

    if _contains(text, ["medicine", "medication", "pill", "dose", "dosage", "forgot", "missed"]) and _contains(text, ["unsure", "forgot", "missed", "don't know", "not sure"]):
        _set_alert_if_not_downgrade(
            alert,
            alert_type="caregiver",
            level="medium",
            reason="Medication uncertainty may need confirmation.",
            contact="caregiver, pharmacist, doctor, or clinic",
            caregiver_message="The user is unsure about medication. Please help confirm the label, schedule, or pharmacist instructions.",
        )

    if _contains(text, ["kill myself", "hurt myself", "end my life", "suicide"]):
        _set_alert_if_not_downgrade(
            alert,
            alert_type="emergency",
            level="high",
            reason="Self-harm language requires immediate human support.",
            contact="emergency services, crisis line, or trusted nearby person",
            caregiver_message="The user used self-harm language. Please get immediate human support and do not leave them alone.",
        )

    alert["signals"] = {
        "dizzy_now": dizzy_now,
        "imminent_fall": imminent_fall,
        "explicit_caregiver_request": explicit_caregiver_request,
        "emergency_red_flag": emergency_red_flag,
    }
    return alert


def apply_output_guardrails(
    response_text: str,
    *,
    alert_decision: dict[str, Any],
    router_decision: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    fallback_used = False
    fallback_name = ""
    final = response_text
    lower = response_text.lower()
    active_topic = str(router_decision.get("active_topic", "")).lower()
    agents = " ".join(router_decision.get("activated_agents", [])).lower()
    fraud_context = "safety" in active_topic or "fraud" in active_topic or "safety_agent" in agents or any(
        signal in {"email", "link", "bank", "password", "money", "transfer", "otp", "police"}
        for signal in router_decision.get("detected_signals", [])
    )

    if fraud_context and any(phrase in lower for phrase in VAGUE_SAFETY_PHRASES):
        issues.append("vague_safety_advice")
        final = CONCRETE_FRAUD_VERIFICATION_GUIDANCE
        fallback_used = True
        fallback_name = "concrete_fraud_verification_guidance"

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

    if any(term in lower for term in ["take an extra dose", "double your dose", "change your dose"]):
        issues.append("unsafe_medication_instruction")
    if any(term in lower for term in ["share your password", "send the money now", "click the unknown link"]):
        issues.append("unsafe_safety_instruction")

    source = "output_guardrail_replacement" if fallback_used else ""
    return {
        "final_message": final,
        "guardrail_issues": issues,
        "fallback_used": fallback_used,
        "fallback_name": fallback_name,
        "final_message_source": source,
    }


def _normalized_trace(state: ElderGuardState) -> dict[str, Any]:
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


def memory_context_builder_node(state: ElderGuardState) -> ElderGuardState:
    request = state["request"]
    client_history = [
        {"role": item.get("role", ""), "message": item.get("message") or item.get("content", "")}
        for item in request.client_history[-8:]
        if item.get("role") in {"user", "assistant"} and (item.get("message") or item.get("content"))
    ]
    stored_history = get_recent_conversation(request.user_id, request.conversation_id, limit=8)
    history = client_history if client_history else stored_history
    previous_assistant = ""
    for item in reversed(history):
        if item.get("role") == "assistant":
            previous_assistant = str(item.get("message", ""))
            break

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


def rule_router_node(state: ElderGuardState) -> ElderGuardState:
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

    safety_terms = ["email", "link", "click", "login", "password", "otp", "bank", "money", "transfer", "police", "scam", "fall", "fell", "chest pain", "can't breathe", "emergency"]
    health_terms = ["dizzy", "weak", "medicine", "medication", "pill", "dose", "hungry", "food", "banana", "juice", "nutrition", "empty stomach", "walk", "mobility", "pain", "fever"]
    emotional_terms = ["lonely", "sad", "worried", "stress", "stressed", "family", "children", "friend", "support"]
    action_terms = ["write", "draft", "message", "checklist", "reminder", "script", "caregiver note", "what should i say"]

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
    if _contains(text, emotional_terms):
        activated.append("emotional_social_agent")
        topics.append("emotional_support" if _contains(text, ["lonely", "sad", "worried", "stress", "stressed"]) else "emotional_social")
        signals.extend([term for term in emotional_terms if term in text][:5])
    if _contains(text, action_terms):
        activated.append("action_agent")
        topics.append("action")
        signals.extend([term for term in action_terms if term in text][:5])
    if not activated:
        activated = ["health_daily_care_agent"]
        topics = ["general_daily_life"]

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


def ml_router_is_enabled() -> bool:
    return os.getenv("USE_ML_ROUTER", "false").strip().lower() in {"1", "true", "yes", "on"}


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


def router_node(
    state: ElderGuardState,
) -> ElderGuardState:
    """
    Safe hybrid router.

    Rules remain responsible for:
    1. High risk situations
    2. Action requests
    3. Multi agent requests
    4. Context dependent follow up messages

    ML handles normal single agent routing.
    """

    rule_state = rule_router_node(
        state
    )

    router_decision = dict(
        rule_state.get(
            "router_decision",
            {},
        )
    )

    if not ml_router_is_enabled():
        logger.info(
            "Router source=rules"
        )

        return rule_state

    try:
        user_text = extract_latest_user_text(
            state
        )

        ml_route = predict_route(
            user_text
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

        router_decision["ml_candidate"] = (
            ml_route
        )

        must_keep_rules = (
            priority == "high"
            or "action_agent" in rule_agents
            or len(rule_agents) > 1
        )

        if must_keep_rules:
            router_decision["router_source"] = (
                "rules_override"
            )

            router_decision["reason"] = (
                "Rules were kept because the message "
                "contains a high risk situation, "
                "an action request, or multiple agents."
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

        rule_has_clear_evidence = (
            bool(detected_signals)
            and active_topic
            != "general_daily_life"
        )

        if (
            rule_has_clear_evidence
            and ml_route != rule_route
        ):
            router_decision["router_source"] = (
                "rules_ml_disagreement"
            )

            router_decision["reason"] = (
                "ML and rule routing disagreed. "
                "The rule route was kept because "
                "the rule router found clear evidence."
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

        router_decision["selected_agent"] = (
            ml_route
        )

        router_decision["activated_agents"] = [
            ml_route
        ]

        router_decision["ml_route"] = (
            ml_route
        )

        if ml_route == rule_route:
            router_decision["router_source"] = (
                "ml_confirmed"
            )

            router_decision["reason"] = (
                "ML and rule routing selected "
                "the same specialist."
            )

        else:
            router_decision["router_source"] = (
                "ml"
            )

            router_decision["active_topic"] = (
                GRAPH_ROUTE_TO_TOPIC.get(
                    ml_route,
                    "general_daily_life",
                )
            )

            router_decision["reason"] = (
                "The rule router had no clear signal, "
                "so the ML router selected the specialist."
            )

        logger.info(
            "Router source=%s route=%s",
            router_decision[
                "router_source"
            ],
            ml_route,
        )

        return _update_router_state(
            rule_state,
            router_decision,
        )

    except Exception as error:
        logger.exception(
            "ML router failed. "
            "Using rule router fallback."
        )

        router_decision["router_source"] = (
            "rules_fallback"
        )

        router_decision["ml_router_error"] = (
            str(error)
        )

        router_decision["reason"] = (
            "The ML router failed, so the "
            "rule router result was retained."
        )

        return _update_router_state(
            rule_state,
            router_decision,
        )

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


def broad_agent_reasoning_node(state: ElderGuardState) -> ElderGuardState:
    message = state["request"].message
    text = message.lower()
    router = state["router_decision"]
    outputs: dict[str, dict[str, Any]] = {}

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
    if "emotional_social_agent" in router["activated_agents"]:
        outputs["emotional_social_agent"] = _agent_result(
            "emotional_social_agent",
            "low",
            [s for s in router["detected_signals"] if s in text],
            ["acknowledge the feeling", "suggest one small social support step"],
            ["do not pretend to be a therapist", "do not give unrelated safety warnings"],
            ["whether the user wants help contacting someone"],
        )
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


def alert_decision_node(state: ElderGuardState) -> ElderGuardState:
    return _trace(state, "alert_decision", alert_decision=evaluate_alert_decision(state["request"].message))


def _mock_conversation_reply(state: ElderGuardState) -> dict[str, Any]:
    message = state["request"].message
    text = message.lower()
    alert = state.get("alert_decision", {})
    if alert.get("alert_required"):
        if alert.get("alert_type") == "health" and "dizzy" in text:
            final = "Please sit or lie down now. Since you feel dizzy after standing, I recommend asking someone nearby to check on you. If you have chest pain, trouble breathing, fainting, confusion, or you fall, call emergency services."
        elif alert.get("alert_type") == "fraud":
            final = "Please do not click the link or share any login, password, bank, or money information yet. Ask a trusted person to check it first, or contact the organization using an official number or website."
        else:
            final = f"This may need help from {alert.get('recommended_contact', 'a trusted person')}. {alert.get('caregiver_message', '')}"
    elif "banana" in text and "empty stomach" in text:
        final = "A banana is usually a gentle food for many people, but if your stomach feels upset, start with a small amount and some water. Do you feel hungry, nauseous, or dizzy right now?"
    elif "email" in text or "link" in text:
        final = "Do not click the link yet. Please check who sent it and whether you expected it before opening anything."
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


def conversation_agent_node(state: ElderGuardState) -> ElderGuardState:
    settings = get_settings()
    memory_context = state["memory_context"]
    router = state["router_decision"]
    broad_outputs = state["broad_agent_outputs"]
    selected_agent = str(router.get("selected_agent", "health_daily_care_agent"))
    prompt_name = f"{selected_agent}_prompt"
    prompt_version = "v2"
    coordinated: dict[str, Any]

    if settings.llm_mode == "mock":
        coordinated = _mock_conversation_reply(state)
        coordinated["prompt_name"] = prompt_name
        coordinated["prompt_version"] = prompt_version
        coordinated["raw_llm_output"] = ""
    else:
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
            raw = call_openrouter_llm(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
                ],
                temperature=0.4,
            )
            data = _json_from_text(raw)
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
            error = {
                "error_type": type(exc).__name__,
                "raw_error": str(exc),
                "traceback": traceback.format_exc(),
                "llm_provider": settings.llm_provider,
                "llm_model": settings.active_llm_model,
                "llm_mode": settings.llm_mode,
                "llm_calls_this_turn": get_llm_call_counter(),
            }
            Path("data").mkdir(exist_ok=True)
            Path("data/conversation_agent_error.txt").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
            coordinated = {
                "final_message": f"Conversation Agent error: {type(exc).__name__}: {exc}",
                "xai_simple": "The selected LLM provider failed. See Developer Console for the full error.",
                "follow_up_questions": [],
                "final_message_source": "safe_service_failure",
                "active_mode": "LLMError",
                "llm_error": error,
                "raw_llm_output": "",
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
            }

    coordinated["llm_calls_this_turn"] = get_llm_call_counter()
    coordinated["llm_provider"] = settings.llm_provider
    coordinated["llm_model"] = settings.active_llm_model
    coordinated["llm_mode"] = settings.llm_mode
    coordinated["llm_caller"] = selected_agent
    return _trace(state, "conversation_agent", coordinated=coordinated)


def _json_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def guardrails_node(state: ElderGuardState) -> ElderGuardState:
    coordinated = dict(state["coordinated"])
    response_before_guardrail = str(coordinated.get("final_message", ""))
    guardrail_result = apply_output_guardrails(
        response_before_guardrail,
        alert_decision=state.get("alert_decision", {}),
        router_decision=state.get("router_decision", {}),
    )
    final = str(guardrail_result["final_message"])
    issues: list[str] = list(guardrail_result["guardrail_issues"])
    llm_calls = int(coordinated.get("llm_calls_this_turn", get_llm_call_counter()) or 0)
    if llm_calls > 1:
        issues.append("llm_call_budget_exceeded")
    response_after_guardrail = final
    fallback_used = bool(guardrail_result["fallback_used"])
    fallback_name = str(guardrail_result["fallback_name"])
    final_message_source = str(guardrail_result["final_message_source"] or coordinated.get("final_message_source", "response_coordinator"))
    coordinated["final_message"] = final
    coordinated["final_message_source"] = final_message_source

    risks = [output.get("risk", "low") for output in state.get("broad_agent_outputs", {}).values()]
    overall_risk = _risk_max(*risks, state.get("router_decision", {}).get("priority", "low"))
    care_plan = {
        "router_decision": state.get("router_decision", {}),
        "broad_agent_outputs": state.get("broad_agent_outputs", {}),
        "alert_decision": state.get("alert_decision", {}),
        "guardrail_issues": issues,
    }
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


def save_memory_node(state: ElderGuardState) -> ElderGuardState:
    request = state["request"]
    response = state["final_response"]
    llm_error = response.developer_state.get("llm_error", {})
    classified_memories = classify_memory(
        request.message,
        final_message_source=response.final_message_source,
        llm_error=llm_error,
        router_decision=state.get("router_decision", {}),
        alert_decision=state.get("alert_decision", {}),
    )
    save_message(request.user_id, request.conversation_id, "user", request.message)
    if not llm_error:
        save_message(request.user_id, request.conversation_id, "assistant", response.final_message)

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

    profile_updates = profile_updates_from_memories(classified_memories)
    if profile_updates and not llm_error:
        profile = get_user_profile(request.user_id)
        for key, values in profile_updates.items():
            profile.setdefault(key, [])
            profile[key].extend(values)
        save_user_profile(request.user_id, profile)

    raw_json = {**state.get("raw_json", {}), "langgraph_trace": state.get("langgraph_trace", [])}
    raw_json["classified_memories"] = classified_memories
    _write_trace_artifact(request, raw_json)
    response.memory_id = save_interaction(request, response, raw_json)
    response.developer_state["classified_memories"] = classified_memories
    if not llm_error:
        summary_items = [
            f"{item['memory_type']}: {item['content']}"
            for item in classified_memories
            if item.get("memory_type") in {"episodic_memory", "long_term_profile"} and item.get("confirmed")
        ]
        if summary_items:
            update_user_summary(request.user_id, " ".join(summary_items)[:500])
    final_state = _trace(state, "save_memory", final_response=response)
    response.developer_state["langgraph_trace"] = final_state.get("langgraph_trace", [])
    response.developer_state["trace"] = _normalized_trace(final_state)
    response.developer_state["memory_saved"] = True
    return final_state


def _write_trace_artifact(request: ChatRequest, raw_json: dict[str, Any]) -> None:
    trace_dir = Path("data/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_user = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in request.user_id)
    safe_conversation = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in request.conversation_id)
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


def build_graph():
    graph = StateGraph(ElderGuardState)
    graph.add_node("memory_context_builder", memory_context_builder_node)
    graph.add_node("router", router_node)
    graph.add_node("broad_agent_reasoning", broad_agent_reasoning_node)
    graph.add_node("alert_decision_step", alert_decision_node)
    graph.add_node("conversation_agent", conversation_agent_node)
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("memory_context_builder")
    graph.add_edge("memory_context_builder", "router")
    graph.add_edge("router", "broad_agent_reasoning")
    graph.add_edge("broad_agent_reasoning", "alert_decision_step")
    graph.add_edge("alert_decision_step", "conversation_agent")
    graph.add_edge("conversation_agent", "guardrails")
    graph.add_edge("guardrails", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()


elderguard_graph = build_graph()


def run_elderguard_workflow(request: ChatRequest) -> ChatResponse:
    reset_llm_call_counter()
    try:
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
        if any(term in text for term in ["chest pain", "can't breathe", "trouble breathing", "fainted", "confusion", "hurt myself", "kill myself"]):
            emergency_hint = " If this may be urgent, please contact emergency services or a trusted nearby person now."
        error = {
            "error_type": type(exc).__name__,
            "raw_error": str(exc),
            "traceback": traceback.format_exc(),
            "llm_calls_this_turn": get_llm_call_counter(),
        }
        Path("data").mkdir(exist_ok=True)
        Path("data/last_chat_error.txt").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
        return ChatResponse(
            final_message=f"Service configuration error: {type(exc).__name__}: {exc}.{emergency_hint}",
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
                "llm_error": error,
            },
        )
