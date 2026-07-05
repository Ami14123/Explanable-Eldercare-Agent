# Rule-based care knowledge graph used for explainable routing and safety notes.
from __future__ import annotations

import re
from typing import Any


# Each node maps keywords to an agent, reasoning path, actions, and safety rules.
CARE_GRAPH = {
    # Fraud and link-safety nodes catch suspicious email or money-pressure patterns.
    "suspicious_email": {
        "keywords": ["email", "emailed", "unknown email", "don't know them", "do not know them"],
        "agent": "fraud_agent",
        "reason": "email sender and link safety need investigation",
        "pathway": ["suspicious_email", "sender_identity", "expected_or_not", "link_destination", "action_decision"],
        "actions": ["Do not click yet.", "Ask who sent it and whether it was expected."],
        "safety_rules": ["Verify sender before clicking.", "Check whether the link asks for login, payment, download, or personal information."],
        "still_missing": ["sender identity", "whether expected", "whether link asks for login/payment/download/personal information"],
        "recommended_next_question": "Who sent it, and were you expecting this email?",
    },
    "unknown_link": {
        "keywords": ["link", "click the link", "clicking the link"],
        "agent": "fraud_agent",
        "reason": "unknown or unverified link risk",
        "pathway": ["unknown_link", "do_not_click_yet", "verify_sender", "inspect_domain"],
        "actions": ["Do not click the link yet.", "Inspect the sender and link destination without opening it."],
        "safety_rules": ["Unknown links should not be clicked until verified."],
        "still_missing": ["link destination", "sender identity"],
        "recommended_next_question": "Can you see the website address without clicking the link?",
    },
    "daughter_friend_sender": {
        "keywords": ["daughter's friend", "daughter friend", "my daughter's friend", "daughters friend"],
        "agent": "fraud_agent",
        "reason": "known-sounding sender still needs verification",
        "pathway": ["daughter_friend_sender", "still_verify_with_daughter", "check_for_login_payment_download"],
        "actions": ["Ask daughter whether her friend really sent it.", "Do not click until verified."],
        "safety_rules": ["A familiar name can still be spoofed.", "Verify through a separate trusted channel."],
        "still_missing": ["whether daughter confirms sender", "whether email asks for login/payment/download/personal information"],
        "recommended_next_question": "Does the email ask you to log in, pay money, download a file, or share personal information?",
    },
    # Health nodes catch dizziness, weakness, and fall-related safety risks.
    "dizziness": {
        "keywords": ["dizzy", "dizziness", "lightheaded", "weak", "weakness"],
        "agent": "health_agent",
        "reason": "possible fall or health risk",
        "pathway": ["dizziness", "stand_or_sit", "fall_risk", "red_flag_questions"],
        "actions": [
            "Sit or lie down somewhere safe.",
            "Avoid walking alone while dizzy or weak.",
            "Contact emergency services or a healthcare professional if symptoms worsen.",
        ],
        "safety_rules": [
            "Do not diagnose the cause of dizziness.",
            "Ask red-flag questions about chest pain, breathing trouble, fainting, fall, or being alone.",
        ],
        "still_missing": ["chest pain", "trouble breathing", "fainting or fall", "whether user is alone"],
        "recommended_next_question": "Do you have chest pain, trouble breathing, fainting, or did you fall?",
    },
    "near_fall": {
        "keywords": ["almost falling", "near fall", "fall", "fell"],
        "agent": "health_agent",
        "reason": "possible fall risk",
        "pathway": ["near_fall", "fall_risk", "remove_hazards", "caregiver_check_in"],
        "actions": [
            "Move away from hazards and sit safely.",
            "Ask a caregiver or trusted person to check the home environment.",
        ],
        "safety_rules": [
            "Treat near-fall details as a safety risk.",
            "Recommend a nearby person or caregiver if the user is unsafe walking alone.",
        ],
    },
    # Medication nodes catch missed-dose and dosage uncertainty.
    "missed_medication": {
        "keywords": ["forgot", "missed", "medicine", "medication", "pill", "dose", "dosage"],
        "agent": "medication_agent",
        "reason": "possible dosage confusion or missed medication",
        "pathway": ["missed_medicine", "medicine_name", "dose_timing", "do_not_double_dose", "pharmacist_or_doctor"],
        "actions": [
            "Do not take an extra dose or change dosage on your own.",
            "Check the prescription label or medication log.",
            "Contact a pharmacist, doctor, or clinic if unsure.",
        ],
        "safety_rules": [
            "Never tell the user to double dose.",
            "Medication uncertainty should be confirmed by a pharmacist, doctor, or clinic.",
        ],
        "still_missing": ["medicine name", "usual dose time", "label missed-dose instructions"],
        "recommended_next_question": "What medicine is it, and when were you supposed to take it?",
    },
    # Daily care nodes cover food, drink, money, and transport support needs.
    "daily_drink_or_food_need": {
        "keywords": ["juice", "water", "drink", "food", "hungry", "groceries", "grocery"],
        "agent": "basic_needs_agent",
        "reason": "practical food or drink support need",
        "pathway": ["hunger", "available_food", "drink_only", "suggest_simple_food"],
        "actions": [
            "Check whether the need is urgent right now.",
            "Use a simple safe alternative if available.",
            "Ask a trusted person for help with groceries or a drink if needed.",
        ],
        "safety_rules": [
            "If the user is hungry, juice may help thirst but does not replace food.",
            "Ask what simple food is available before suggesting outside help.",
        ],
        "still_missing": ["what food is available"],
        "recommended_next_question": "Do you have something simple to eat, like bread, rice, soup, crackers, banana, or yogurt?",
    },
    "juice_only": {
        "keywords": ["juice in the fridge", "only juice", "have juice"],
        "agent": "basic_needs_agent",
        "reason": "juice may not be enough for hunger",
        "pathway": ["juice_only", "not_enough_for_hunger", "avoid_only_sweet_drink_on_empty_stomach"],
        "actions": ["Explain juice is a drink, not a full food.", "Ask about simple food options."],
        "safety_rules": ["If hungry, suggest simple food instead of only sweet drink."],
        "still_missing": ["whether any simple food is available"],
        "recommended_next_question": "Do you have bread, rice, crackers, soup, banana, or yogurt?",
    },
    "small_money_need": {
        "keywords": ["money", "buy", "pay", "cannot afford", "don't have money"],
        "agent": "basic_needs_agent",
        "reason": "practical financial support for a small daily need",
        "pathway": ["small_money_need", "financial_support", "ask_trusted_person", "avoid_pressure"],
        "actions": [
            "Clarify what the user needs to buy.",
            "Suggest asking a trusted family member, caregiver, neighbor, or local support person.",
        ],
        "safety_rules": [
            "Offer practical support before escalating to safety warnings.",
            "Do not imply a medical emergency for ordinary daily needs.",
        ],
    },
    "transport_need": {
        "keywords": ["transport", "ride", "bus", "taxi", "appointment"],
        "agent": "basic_needs_agent",
        "reason": "practical transport support need",
        "pathway": ["transport_need", "daily_life_support", "safe_ride", "confirm_helper"],
        "actions": [
            "Clarify where the user needs to go.",
            "Suggest arranging help from a trusted person or safe transport option.",
        ],
        "safety_rules": [
            "Recommend trusted transport or a known helper.",
            "Ask where and when the user needs to go.",
        ],
    },
    # Additional fraud nodes catch banking, authority, and unknown-link pressure.
    "bank_account_request": {
        "keywords": ["bank account", "bank", "otp", "password", "transfer money", "transfer", "crypto", "gift card"],
        "agent": "fraud_agent",
        "reason": "possible identity theft or money scam risk",
        "pathway": ["bank_account_request", "fraud_risk", "do_not_share_info", "contact_bank_officially"],
        "actions": [
            "Do not share bank details, OTP codes, passwords, or ID information.",
            "Stop replying and contact the real bank using an official number.",
        ],
        "safety_rules": [
            "Do not claim a scam with certainty unless obvious.",
            "Do not share OTPs, passwords, bank details, or transfer money under pressure.",
        ],
    },
    "unknown_email_link": {
        "keywords": ["email", "link", "unknown sender", "don't know them"],
        "agent": "fraud_agent",
        "reason": "unknown link or suspicious email risk",
        "pathway": ["unknown_email", "link_risk", "do_not_click", "verify_sender"],
        "actions": [
            "Do not click the link yet.",
            "Check who sent it and whether the message was expected.",
            "Look at the website address without opening it if possible.",
        ],
        "safety_rules": [
            "Unknown links should not be clicked until the sender and purpose are verified.",
            "If the sender may be known, confirm through a trusted separate channel.",
        ],
    },
    "fake_authority": {
        "keywords": ["police", "officer", "government", "tax", "court"],
        "agent": "fraud_agent",
        "reason": "possible fake authority pressure",
        "pathway": ["fake_authority", "fraud_risk", "do_not_transfer_money", "verify_officially"],
        "actions": [
            "Do not transfer money because of pressure from a caller.",
            "Verify through an official number or trusted caregiver.",
        ],
        "safety_rules": [
            "Fake authority scams often use urgency or fear.",
            "Verify through official contact information, not the caller's instructions.",
        ],
    },
    # Emotional support nodes catch loneliness and social isolation needs.
    "loneliness": {
        "keywords": ["lonely", "alone", "friend", "friends", "sad", "stressed", "stress", "worried", "talk"],
        "agent": "companion_agent",
        "reason": "possible emotional distress or social isolation",
        "pathway": ["loneliness", "emotional_context", "supportive_conversation", "small_connection_action"],
        "actions": [
            "Send a short message to a trusted friend or caregiver.",
            "Try one calming activity such as slow breathing or sitting with a warm drink.",
        ],
        "safety_rules": [
            "Provide emotional support without pretending to be a therapist.",
            "Ask whether the user wants help reaching a trusted person.",
        ],
        "still_missing": ["who the user wants to contact", "whether they want help writing a message"],
        "recommended_next_question": "Would you like help writing a short message to someone you trust?",
    },
}


# Return the graph dictionary for callers and tests that inspect the rules.
def build_care_graph() -> dict[str, dict[str, Any]]:
    return CARE_GRAPH


# Match whole single-word keywords while allowing phrase keywords by substring.
def _keyword_matches(lowered: str, keyword: str) -> bool:
    keyword = keyword.lower()
    # Whole-word matching prevents terms like "fall" matching unrelated words.
    if keyword.replace(" ", "").replace("_", "").isalnum() and " " not in keyword:
        return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", lowered) is not None
    return keyword in lowered


# Query the care graph and return explainable matches for one user message.
def query_care_graph(message: str) -> dict[str, Any]:
    # Vietnamese note: GraphRAG anh xa tin hieu cua user thanh pathway va safety rule co the giai thich.
    lowered = message.lower()
    # Collect graph outputs in separate lists so the response can show each part.
    detected_nodes: list[str] = []
    risk_pathways: list[list[str]] = []
    recommended_actions: list[str] = []
    still_missing: list[str] = []
    recommended_questions: list[str] = []
    detected_signals: list[dict[str, str]] = []

    # Scan every node and keep the nodes whose keywords are present.
    for node, data in CARE_GRAPH.items():
        # Avoid treating "friend" in suspicious email examples as loneliness.
        if node == "loneliness" and ("email" in lowered or "link" in lowered):
            continue
        matched = [keyword for keyword in data["keywords"] if _keyword_matches(lowered, keyword)]
        if not matched:
            continue

        # Merge matched node outputs into the graph reasoning result.
        detected_nodes.append(node)
        risk_pathways.append(data["pathway"])
        recommended_actions.extend(data["actions"])
        still_missing.extend(data.get("still_missing", []))
        if data.get("recommended_next_question"):
            recommended_questions.append(data["recommended_next_question"])
        detected_signals.append(
            {
                "signal": matched[0],
                "agent": data["agent"],
                "reason": data["reason"],
            }
        )

    # Deduplicate repeated rules/actions while preserving first-seen order.
    return {
        "detected_nodes": detected_nodes,
        "risk_pathways": risk_pathways,
        "reasoning_path": risk_pathways,
        "relevant_safety_rules": list(
            dict.fromkeys(
                rule
                for node in detected_nodes
                for rule in CARE_GRAPH.get(node, {}).get("safety_rules", [])
            )
        ),
        "recommended_actions": list(dict.fromkeys(recommended_actions)),
        "still_missing": list(dict.fromkeys(still_missing)),
        "recommended_next_question": recommended_questions[0] if recommended_questions else "",
        "recommended_next_action": recommended_actions[0] if recommended_actions else "",
        "detected_signals": detected_signals,
    }


# Add a short human-readable summary on top of the raw graph result.
def get_graph_reasoning(message: str) -> dict[str, Any]:
    graph_result = query_care_graph(message)
    # Empty matches are still explained so the trace is clear.
    if not graph_result["detected_nodes"]:
        graph_result["summary"] = "No specific care graph pathway was triggered."
    else:
        # Join pathways for compact display in the UI trace.
        pathways = [" -> ".join(pathway) for pathway in graph_result["risk_pathways"]]
        graph_result["summary"] = "Care graph pathways: " + "; ".join(pathways)
    return graph_result
