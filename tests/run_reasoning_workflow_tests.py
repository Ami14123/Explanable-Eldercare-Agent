# Tests for conversation topic transitions, follow-ups, and action fulfillment.
from app.conversation_state import (
    classify_transition,
    fulfill_requested_action,
    normalize_state,
    update_conversation_state,
)


# Apply one user message to state and return both state and transition.
def step(state: dict, message: str, assistant: str = "") -> tuple[dict, dict]:
    transition = classify_transition(message, state)
    updated = update_conversation_state(state, message, transition, assistant_message=assistant)
    return updated, transition


# Equality assertion with readable labels for standalone execution.
def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


# Membership assertion with readable labels for standalone execution.
def assert_in(member, container, label: str) -> None:
    if member not in container:
        raise AssertionError(f"{label}: expected {member!r} in {container!r}")


# Run conversation-state workflow cases.
def main() -> None:
    cases_passed = 0

    # Fraud should start a topic, then unrelated hunger should replace it.
    state = normalize_state({})
    state, transition = step(state, "Someone emailed me a link from someone I do not know")
    assert_equal(transition["transition"], "start_new_topic", "fraud starts topic")
    assert_equal(state["active_topic"], "fraud_safety", "fraud topic")
    state, transition = step(state, "I am hungry")
    assert_equal(transition["transition"], "start_new_topic", "fraud to food switch")
    assert_equal(state["active_topic"], "food_support", "food topic replaces fraud")
    cases_passed += 1

    # Medication text should switch the active topic to medication support.
    state, transition = step(state, "I forgot my medicine")
    assert_equal(transition["transition"], "start_new_topic", "food to medication switch")
    assert_equal(state["active_topic"], "medication_support", "medication topic")
    cases_passed += 1

    # A yes/no answer should satisfy the pending fraud question.
    state["pending_question"] = "Does it ask you to log in, send money, download something, or share private information?"
    state["pending_answer_type"] = "requested_sensitive_action"
    state, transition = step(state, "yes, it asks me to send money")
    assert_equal(transition["transition"], "answer_pending_question", "yes/no answers pending question")
    assert_equal(state["known_facts"]["requested_sensitive_action"], "yes, it asks me to send money", "pending answer stored")
    cases_passed += 1

    # Correction language should mark a correction transition.
    state, transition = step(state, "actually I mean my food, not medicine")
    assert_equal(transition["transition"], "correct_previous_assumption", "correction transition")
    cases_passed += 1

    # Message draft requests should be fulfilled immediately when clear.
    state, transition = step(state, "write a message to my daughter")
    action = fulfill_requested_action(transition["requested_action"], state, "write a message to my daughter")
    assert_equal(transition["transition"], "request_action", "message draft transition")
    assert_equal(action["should_fulfill_now"], True, "action should fulfill")
    assert action["content"].strip(), "draft content should not be empty"
    cases_passed += 1

    # Ordinary food questions should route to food support.
    state = normalize_state({})
    state, transition = step(state, "What can I make with rice and eggs?")
    assert_equal(transition["transition"], "start_new_topic", "normal food question routes to food support")
    assert_equal(state["active_topic"], "food_support", "normal question topic")
    cases_passed += 1

    # Last recommendation should be preserved for repetition guards.
    old_recommendation = "Please do not click the link until verified."
    state = normalize_state({"active_topic": "fraud_safety", "last_recommendation": old_recommendation})
    state, transition = step(state, "What should I do next?", assistant=old_recommendation)
    assert_equal(state["last_recommendation"], old_recommendation, "last recommendation tracked for repetition guard")
    cases_passed += 1

    # Rejected actions should be remembered as impossible actions.
    state = normalize_state({"active_topic": "caregiver_coordination", "pending_action": "call"})
    state, transition = step(state, "I cannot call my daughter")
    assert_equal(transition["transition"], "reject_action", "impossible action rejection")
    assert state["impossible_actions"], "impossible action should be stored"
    cases_passed += 1

    # Mixed health/fraud text should select one high-risk topic.
    state = normalize_state({})
    state, transition = step(state, "I feel dizzy and someone is asking for my bank password")
    assert_equal(transition["transition"], "start_new_topic", "mixed issue starts topic")
    assert_in(state["active_topic"], {"health_support", "fraud_safety"}, "mixed topic selected")
    cases_passed += 1

    # Unrelated cooking question should replace old high-risk topic.
    state = normalize_state({"active_topic": "fraud_safety", "topic_status": "active"})
    state, transition = step(state, "What can I cook for lunch?")
    assert_equal(transition["transition"], "start_new_topic", "unrelated topic after high risk")
    assert_equal(state["active_topic"], "food_support", "new unrelated topic replaces old high-risk topic")
    cases_passed += 1

    # Email sender follow-up should stay inside fraud topic.
    state = normalize_state({})
    state, transition = step(state, "Someone emailed me but I don't know them, should I click?")
    assert_equal(state["active_topic"], "fraud_safety", "email topic")
    assert_equal(state["known_facts"]["sender_identity"], "unknown", "unknown sender stored")
    assert_equal(state["pending_answer_type"], "sender_identity", "sender pending answer type")
    state, transition = step(state, "my daughter's friend")
    assert_equal(transition["transition"], "answer_pending_question", "sender follow-up answers pending question")
    assert_equal(state["active_topic"], "fraud_safety", "fraud topic remains active")
    assert_equal(state["known_facts"]["sender_identity"], "daughter's friend", "sender answer stored")
    cases_passed += 1

    # Hunger after fraud should start a new food topic.
    state, transition = step(state, "I am hungry")
    assert_equal(transition["transition"], "start_new_topic", "hunger replaces fraud")
    assert_equal(state["active_topic"], "food_support", "hunger topic active")
    cases_passed += 1

    # Food message draft should include the recipient.
    state, transition = step(state, "yes, a message for my daughter to buy food")
    action = fulfill_requested_action(transition["requested_action"], state, "yes, a message for my daughter to buy food")
    assert_equal(transition["transition"], "request_action", "food message draft requested")
    assert_equal(action["should_fulfill_now"], True, "food message draft fulfilled")
    assert "daughter" in action["content"].lower(), "draft should mention daughter"
    cases_passed += 1

    # Juice answer should stay in food support and save drink availability.
    state = normalize_state({"active_topic": "food_support", "known_facts": {"hungry": True}})
    state["pending_question"] = "What simple food do you have nearby?"
    state["pending_answer_type"] = "available_food"
    state, transition = step(state, "no, I have juice in the fridge")
    assert_equal(transition["transition"], "answer_pending_question", "juice correction answers food question")
    assert_equal(state["active_topic"], "food_support", "food topic remains active for correction")
    assert_equal(state["known_facts"]["drink_available"], "juice", "juice stored")
    cases_passed += 1

    # Medication timing answer should not switch topics.
    state = normalize_state({})
    state, transition = step(state, "I forgot my blood pressure medicine")
    assert_equal(state["active_topic"], "medication_support", "medication starts topic")
    state, transition = step(state, "it is 1pm now")
    assert_equal(transition["transition"], "answer_pending_question", "medication timing answers pending question")
    assert_equal(state["active_topic"], "medication_support", "medication topic remains active")
    cases_passed += 1

    # Nutrition questions should store a nutrition intent.
    state = normalize_state({"active_topic": "food_support", "known_facts": {"food_available": "banana"}})
    state, transition = step(state, "banana is good for empty stomach or not?")
    assert_equal(transition["latest_user_intent"], "nutrition_question", "banana question is nutrition intent")
    assert_equal(state["last_user_intent"], "nutrition_question", "nutrition intent stored")
    cases_passed += 1

    # Latest hunger intent should override stale fraud context.
    state = normalize_state({"active_topic": "fraud_safety", "known_facts": {"sender_identity": "unknown"}})
    state, transition = step(state, "I am hungry")
    assert_equal(transition["latest_user_intent"], "basic_food_need", "latest hunger intent overrides old fraud topic")
    assert_equal(state["active_topic"], "food_support", "hunger replaces fraud topic")
    cases_passed += 1

    print(f"All reasoning workflow tests passed ({cases_passed} cases).")


# Allow direct command-line execution.
if __name__ == "__main__":
    main()
