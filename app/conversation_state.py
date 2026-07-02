from __future__ import annotations

import re
from typing import Any

DEFAULT_CONVERSATION_STATE: dict[str, Any] = {
    "active_topic": "general_chat",
    "topic_status": "active",
    "user_goal": "answer the user naturally",
    "pending_action": "",
    "pending_question": "",
    "pending_answer_type": "",
    "known_facts": {},
    "missing_facts": [],
    "last_recommendation": "",
    "impossible_actions": [],
    "last_user_intent": "",
    "last_assistant_intent": "",
}

TOPIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "fraud_safety": {
        "terms": ["email", "emailed", "link", "click", "login", "log in", "sign in", "password", "bank", "otp", "transfer", "send money", "lend money", "gift card", "crypto", "police", "download", "account"],
        "agents": ["fraud_agent", "technology_agent"],
        "goal": "decide whether a message, link, or money request is safe",
    },
    "food_support": {
        "terms": ["hungry", "food", "fridge", "refrigerator", "meal", "eat", "cook", "lunch", "dinner", "breakfast", "rice", "bread", "soup", "egg", "eggs", "juice", "water", "drink", "groceries", "buy food", "no food", "banana", "nutrition", "empty stomach"],
        "agents": ["basic_needs_agent", "nutrition_agent"],
        "goal": "get practical food or drink support",
    },
    "medication_support": {
        "terms": ["medicine", "medication", "pill", "dose", "dosage", "prescription", "missed", "forgot", "pharmacist"],
        "agents": ["medication_agent"],
        "goal": "handle medication uncertainty safely",
    },
    "health_support": {
        "terms": ["dizzy", "dizziness", "weak", "weakness", "pain", "chest pain", "breathing", "fainted", "fall", "fell", "fever", "injured"],
        "agents": ["health_agent", "mobility_agent"],
        "goal": "understand immediate health and safety needs",
    },
    "emotional_support": {
        "terms": ["lonely", "sad", "alone", "worried", "stress", "stressed", "scared", "upset", "friends", "children are busy"],
        "agents": ["companion_agent"],
        "goal": "feel heard and choose one supportive next step",
    },
    "caregiver_coordination": {
        "terms": ["daughter", "son", "family", "caregiver", "neighbor", "trusted person", "call my", "text my", "message my"],
        "agents": ["caregiver_support_agent"],
        "goal": "reach or update a trusted support person",
    },
    "technology_help": {
        "terms": ["phone", "app", "website", "computer", "wifi", "internet", "settings", "text message"],
        "agents": ["technology_agent"],
        "goal": "use a device, app, website, or message safely",
    },
    "general_chat": {"terms": [], "agents": [], "goal": "answer the user naturally"},
}

ACTION_PATTERNS: dict[str, list[str]] = {
    "message_draft": ["write a message", "draft a text", "write text", "message for", "message to", "send message", "help me write", "text my", "write my", "a message for"],
    "checklist": ["checklist", "list of steps", "steps to"],
    "call_script": ["call script", "what should i say", "script"],
    "caregiver_summary": ["summary for caregiver", "tell my caregiver", "caregiver summary"],
}

YES_WORDS = {"yes", "yeah", "yep", "ok", "okay", "sure", "please", "yes please"}
NO_WORDS = {"no", "nope", "not", "don't", "do not"}


def normalize_state(state: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(DEFAULT_CONVERSATION_STATE)
    if state:
        normalized.update(state)
    normalized["known_facts"] = dict(normalized.get("known_facts") or {})
    normalized["missing_facts"] = list(normalized.get("missing_facts") or [])
    normalized["impossible_actions"] = list(normalized.get("impossible_actions") or [])
    if normalized.get("active_topic") not in TOPIC_DEFINITIONS:
        normalized["active_topic"] = "general_chat"
    if normalized.get("topic_status") not in {"active", "resolved", "replaced"}:
        normalized["topic_status"] = "active"
    return normalized


def _term_match(text: str, term: str) -> bool:
    term = term.lower()
    if " " in term:
        return term in text
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def detect_topic(message: str) -> str:
    text = message.lower()
    if _has_fraud_signal(text):
        return "fraud_safety"
    scored: list[tuple[int, str]] = []
    for topic, data in TOPIC_DEFINITIONS.items():
        if topic == "general_chat":
            continue
        score = sum(1 for term in data["terms"] if _term_match(text, term))
        if score:
            scored.append((score, topic))
    if not scored:
        return "general_chat"
    # Prefer concrete need topics over caregiver_coordination when both match "daughter".
    scored.sort(key=lambda item: (item[0], item[1] != "caregiver_coordination"), reverse=True)
    return scored[0][1]


def detect_action_request(message: str) -> str:
    text = message.lower()
    for action, terms in ACTION_PATTERNS.items():
        if any(term in text for term in terms):
            return action
    return ""


def classify_latest_user_intent(message: str, detected_topic: str = "") -> str:
    """Classify the latest user message without letting old topic memory dominate."""
    text = message.lower()
    if detect_action_request(message):
        return detect_action_request(message)
    if any(term in text for term in ["email", "emailed", "link", "click", "unknown sender", "don't know them", "do not know them"]):
        return "suspicious_email"
    if any(term in text for term in ["otp", "bank", "password", "transfer", "send money", "lend money", "gift card", "crypto", "police"]):
        return "fraud_safety"
    if any(term in text for term in ["medicine", "medication", "pill", "dose", "dosage", "prescription", "missed", "forgot"]):
        return "medication_uncertainty"
    if any(term in text for term in ["dizzy", "dizziness", "weak", "pain", "chest pain", "breathing", "fainted", "fall", "fell", "fever"]):
        return "health_symptom"
    if any(term in text for term in ["lonely", "sad", "alone", "worried", "stress", "stressed", "scared", "upset"]):
        return "emotional_support"
    if any(term in text for term in ["good for", "empty stomach", "healthy", "nutrition", "nutritious", "can i eat", "should i eat", "banana"]):
        if any(food in text for food in ["banana", "juice", "bread", "rice", "egg", "eggs", "food", "meal", "eat"]):
            return "nutrition_question"
    if any(term in text for term in ["hungry", "no food", "groceries", "buy food", "thirsty", "drink"]):
        return "basic_food_need"
    if detected_topic and detected_topic != "general_chat":
        return detected_topic
    return "general_chat"


def _is_yes(text: str) -> bool:
    cleaned = text.strip().lower().strip(" .,!?")
    return cleaned in YES_WORDS or cleaned.startswith("yes,") or cleaned.startswith("yes ")


def _is_no(text: str) -> bool:
    cleaned = text.strip().lower().strip(" .,!?")
    return cleaned in NO_WORDS or cleaned.startswith("no,") or cleaned.startswith("no ")


def _is_correction(text: str) -> bool:
    return any(text.startswith(prefix) for prefix in ["no,", "no ", "actually", "i mean", "not that", "but "])



def _has_fraud_signal(text: str) -> bool:
    explicit_terms = [
        "email",
        "emailed",
        "link",
        "click",
        "unknown sender",
        "don't know them",
        "do not know them",
        "someone emailed",
    ]
    return any(term in text for term in explicit_terms) or any(
        _term_match(text, term) for term in TOPIC_DEFINITIONS["fraud_safety"]["terms"]
    )

def classify_transition(message: str, state: dict[str, Any]) -> dict[str, Any]:
    state = normalize_state(state)
    text = message.lower().strip()
    detected_topic = detect_topic(message)
    requested_action = detect_action_request(message)
    current_topic = state.get("active_topic", "general_chat")
    pending_question = state.get("pending_question", "")
    pending_action = state.get("pending_action", "")

    if requested_action:
        transition = "request_action"
    elif pending_question and _is_correction(text) and any(term in text for term in ["have", "is", "it", "they"]):
        transition = "answer_pending_question"
    elif _is_correction(text):
        transition = "correct_previous_assumption"
    elif pending_action and any(term in text for term in ["cannot", "can't", "cant"]):
        transition = "reject_action"
    # Check for pending question/action answers BEFORE topic switching
    elif (_is_yes(text) or _is_no(text)) and (pending_action or pending_question):
        transition = "confirm_action" if _is_yes(text) and pending_action else "answer_pending_question"
    elif (
        pending_question
        and detected_topic != "general_chat"
        and detected_topic != current_topic
        and not (
            current_topic == "fraud_safety"
            and state.get("pending_answer_type") == "sender_identity"
            and detected_topic == "caregiver_coordination"
        )
    ):
        transition = "start_new_topic"
    elif pending_question and len(text.split()) <= 20:
        # Short response to a pending question - stay in current topic
        transition = "answer_pending_question"
        detected_topic = current_topic  # Keep current topic when answering pending question
    elif detected_topic != "general_chat" and detected_topic != current_topic:
        # Topic switch only if no pending question
        transition = "start_new_topic"
    elif (
        detected_topic == "general_chat"
        and current_topic != "general_chat"
        and not pending_question
        and not pending_action
        and not _has_fraud_signal(text)
    ):
        transition = "start_new_topic"
        detected_topic = "general_chat"
    elif detected_topic == current_topic and current_topic != "general_chat":
        transition = "continue_current_topic"
    else:
        transition = "casual_chat" if current_topic == "general_chat" else "continue_current_topic"

    latest_user_intent = classify_latest_user_intent(message, detected_topic)
    return {
        "transition": transition,
        "detected_topic": detected_topic,
        "requested_action": requested_action,
        "latest_user_intent": latest_user_intent,
        "message_intent": latest_user_intent,
        "reason": f"detected_topic={detected_topic}; current_topic={current_topic}; pending_question={bool(pending_question)}; pending_action={pending_action}",
    }


def _intent_label(message: str, action: str, topic: str, state: dict[str, Any]) -> str:
    if action:
        return action
    text = message.lower()
    if _is_yes(text) and state.get("pending_action"):
        return "confirm_action"
    if topic == "food_support" and any(t in text for t in ["no food", "fridge", "hungry", "buy food"]):
        return "food_need"
    if topic == "fraud_safety":
        return "suspicious_message_check"
    return topic


def update_conversation_state(previous_state: dict[str, Any] | None, message: str, transition: dict[str, Any], assistant_message: str = "") -> dict[str, Any]:
    state = normalize_state(previous_state)
    detected_topic = transition.get("detected_topic") or "general_chat"
    transition_name = transition.get("transition", "casual_chat")
    requested_action = transition.get("requested_action", "")

    if transition_name == "start_new_topic":
        state["active_topic"] = detected_topic
        state["topic_status"] = "replaced"
        state["known_facts"] = {}
        state["missing_facts"] = []
        state["pending_action"] = ""
        state["pending_question"] = ""
        state["pending_answer_type"] = ""
    elif transition_name in {"continue_current_topic", "answer_pending_question", "correct_previous_assumption", "request_action", "confirm_action"}:
        # Keep current topic for yes/no and follow-up answers unless the message clearly names a new topic.
        if detected_topic != "general_chat" and transition_name not in {"answer_pending_question", "confirm_action"}:
            state["active_topic"] = detected_topic
        state["topic_status"] = "active"
    elif transition_name == "casual_chat" and state.get("active_topic") == "general_chat":
        state["topic_status"] = "active"

    topic = state.get("active_topic", "general_chat")
    state["user_goal"] = TOPIC_DEFINITIONS.get(topic, TOPIC_DEFINITIONS["general_chat"])["goal"]
    state["last_user_intent"] = transition.get("message_intent", "")

    _merge_known_facts(state, message, transition)

    if transition_name == "reject_action":
        rejected = state.get("pending_action") or state.get("last_assistant_intent") or "previous suggestion"
        if rejected not in state["impossible_actions"]:
            state["impossible_actions"].append(rejected)

    # If the user confirms an action and provides recipient details, keep the action request.
    if requested_action:
        state["pending_action"] = requested_action
    elif transition_name == "confirm_action" and state.get("pending_action"):
        pass
    elif transition_name == "request_action":
        state["pending_action"] = requested_action
    else:
        state["pending_action"] = ""

    state["missing_facts"] = _missing_facts_for_topic(topic, state["known_facts"], state.get("pending_action", ""))
    state["pending_question"] = _next_pending_question(topic, state["missing_facts"], state.get("pending_action", ""))
    state["pending_answer_type"] = _answer_type_for_question(state["pending_question"])
    if assistant_message:
        state["last_recommendation"] = assistant_message[:500]
    return state


def _merge_known_facts(state: dict[str, Any], message: str, transition: dict[str, Any]) -> None:
    facts = state.setdefault("known_facts", {})
    text = message.lower().strip()
    topic = state.get("active_topic", "general_chat")
    transition_name = transition.get("transition", "")
    if transition_name == "answer_pending_question" and state.get("pending_answer_type"):
        facts[state["pending_answer_type"]] = _normalize_pending_answer(state["pending_answer_type"], message)

    if topic == "fraud_safety":
        if "email" in text or "emailed" in text or "link" in text or "click" in text:
            facts.setdefault("channel", "email/link")
            if any(t in text for t in ["don't know", "do not know", "unknown", "not sure", "someone"]):
                facts["sender_identity"] = "unknown"
        if "daughter" in text and "friend" in text:
            facts["sender_identity"] = "daughter's friend"
        elif transition_name == "answer_pending_question" and state.get("pending_answer_type") == "sender_identity":
            facts["sender_identity"] = _normalize_pending_answer("sender_identity", message)
        if any(t in text for t in ["log in", "login", "sign in", "password"]):
            facts["asks_for_login"] = True
        if any(t in text for t in ["send money", "lend money", "transfer", "pay", "payment", "money"]):
            facts["asks_for_money"] = True
        if any(t in text for t in ["download", "file"]):
            facts["asks_for_download"] = True
        if any(t in text for t in ["urgent", "every minute", "lose money", "waiting"]):
            facts["urgency_pressure"] = True
    elif topic == "food_support":
        if "hungry" in text:
            facts["hungry"] = True
        if "no food" in text or "don't have any food" in text or "dont have any food" in text:
            facts["food_available"] = "none in fridge"
        elif any(t in text for t in ["yes", "have", "bread", "rice", "soup", "eggs", "crackers", "banana", "yogurt", "leftovers"]):
            # User confirmed they have food
            available = []
            for food in ["bread", "rice", "soup", "eggs", "crackers", "banana", "yogurt", "leftovers"]:
                if food in text:
                    available.append(food)
            if available:
                facts["food_available"] = ", ".join(available)
            elif "yes" in text and not facts.get("food_available"):
                facts["food_available"] = "some food"
        if "juice" in text:
            facts["drink_available"] = "juice"
        if "daughter" in text:
            facts["recipient"] = "daughter"
        if any(t in text for t in ["buy food", "groceries", "buy"]):
            facts["needed_item"] = "food/groceries"
    elif topic == "medication_support":
        if "forgot" in text or "missed" in text:
            facts["missed_or_unsure"] = True
        facts.setdefault("details", message.strip())
    elif topic == "health_support":
        for symptom in ["dizzy", "weak", "pain", "chest pain", "breathing", "fall", "fell", "fever"]:
            if symptom in text:
                facts.setdefault("symptoms", [])
                if symptom not in facts["symptoms"]:
                    facts["symptoms"].append(symptom)
    elif topic == "emotional_support":
        facts["emotion_detail"] = message.strip()
    elif topic == "caregiver_coordination":
        if "daughter" in text:
            facts["recipient"] = "daughter"


def _missing_facts_for_topic(topic: str, facts: dict[str, Any], action: str = "") -> list[str]:
    if action == "message_draft":
        return [] if facts.get("recipient") or topic == "food_support" else ["who should receive the message"]
    if topic == "fraud_safety":
        missing = []
        if not facts.get("sender_identity") or facts.get("sender_identity") == "unknown":
            missing.append("sender identity")
        if not (facts.get("asks_for_login") or facts.get("asks_for_money") or facts.get("asks_for_download")):
            missing.append("whether the message asks for login, money, download, or private information")
        return missing
    if topic == "food_support":
        if facts.get("food_available") == "none in fridge":
            return ["who can help buy or bring food"] if not facts.get("recipient") else []
        if facts.get("food_available") and facts.get("food_available") != "none in fridge":
            return []
        return [] if facts.get("hungry") else ["what food or drink is needed"]
    if topic == "medication_support":
        return ["medicine name", "usual dose time"]
    if topic == "health_support":
        return ["whether there are red flag symptoms", "whether the user is alone"]
    return []


def _next_pending_question(topic: str, missing: list[str], action: str = "") -> str:
    if action == "message_draft":
        return ""
    if not missing:
        return ""
    if topic == "fraud_safety":
        if "sender identity" in missing:
            return "Who sent it, and were you expecting it?"
        if "whether the message asks for login, money, download, or private information" in missing:
            return "Does the email ask you to log in, send money, download a file, or share private information?"
        return ""
    if topic == "food_support":
        return "What simple food or drink do you have nearby?"
    if topic == "medication_support":
        return "What medicine is it, and when were you supposed to take it?"
    if topic == "health_support":
        return "Do you have chest pain, trouble breathing, fainting, or did you fall?"
    return "Can you tell me one more detail?"


def _answer_type_for_question(question: str) -> str:
    q = question.lower()
    if "who sent" in q:
        return "sender_identity"
    if "log in" in q or "send money" in q:
        return "requested_sensitive_action"
    if "who could" in q:
        return "support_person"
    if "medicine" in q:
        return "medication_detail"
    if "chest pain" in q:
        return "red_flag_detail"
    return "detail"



def _normalize_pending_answer(answer_type: str, message: str) -> Any:
    cleaned = message.strip().strip(" .''\"")
    lowered = cleaned.lower()
    if answer_type == "sender_identity":
        if any(term in lowered for term in ["don't know", "dont know", "do not know", "unknown", "not sure", "someone"]):
            return "unknown"
        return cleaned
    return cleaned
def agents_for_topic(topic: str) -> list[str]:
    return list(TOPIC_DEFINITIONS.get(topic, TOPIC_DEFINITIONS["general_chat"])["agents"])


def fulfill_requested_action(action: str, state: dict[str, Any], message: str) -> dict[str, Any]:
    if not action:
        return {}
    facts = state.get("known_facts", {})
    topic = state.get("active_topic", "general_chat")
    recipient = facts.get("recipient") or _recipient_from_text(message) or "someone you trust"
    if action == "message_draft":
        if topic == "food_support":
            content = (
                f"Of course. You can send this to your {recipient}:\n\n"
                "Hi, I donâ€™t have any food at home right now. Could you please help me buy or bring some simple food today, like bread, rice, soup, eggs, fruit, or anything easy to eat? I would really appreciate it."
            )
        elif topic == "fraud_safety":
            content = (
                f"You can send this to your {recipient}:\n\n"
                "Hi, I received a message that asks me to click a link or take action. Can you please check whether it is real before I do anything?"
            )
        else:
            content = f"You can send this to {recipient}:\n\nHi, I need some help today. Could you please check on me when you can?"
    elif action == "checklist":
        content = "1. Pause.\n2. Check what is known.\n3. Avoid risky action.\n4. Ask a trusted person if money, health, or safety is involved.\n5. Take one small safe next step."
    elif action == "call_script":
        content = f"You can say: Hi, I need help with {topic.replace('_', ' ')}. Can you help me check what I should do next?"
    elif action == "caregiver_summary":
        content = f"Caregiver summary: topic={topic}; known facts={facts}; goal={state.get('user_goal', '')}."
    else:
        content = "I can help with that. Please tell me what you want the message to say."
    return {"action_type": action, "content": content, "topic": topic, "should_fulfill_now": True}


def _recipient_from_text(message: str) -> str:
    text = message.lower()
    for person in ["daughter", "son", "caregiver", "neighbor", "friend", "family"]:
        if person in text:
            return person
    return ""











