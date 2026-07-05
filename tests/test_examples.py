# End-to-end workflow examples that protect key routing and response behavior.
from app.graph import run_elderguard_workflow
from app.schemas import ChatRequest


# Single-turn examples check that the right broad agent answers each message.
TEST_CASES = [
    {
        "name": "3 normal daily life question",
        "message": "What can I make for lunch with rice and eggs?",
        "expected": set(),
        "not_expected": {"safety_agent", "emotional_social_agent"},
        "must_mention": ["rice"],
        "must_not_mention": ["chest pain", "otp", "dose"],
    },
    {
        "name": "4 lonely family busy",
        "message": "I feel lonely because my family is busy",
        "expected": {"emotional_social_agent"},
        "not_expected": {"safety_agent", "health_daily_care_agent"},
        "must_mention": ["lonely"],
        "must_not_mention": ["doctor", "emergency", "dose", "bank"],
    },
    {
        "name": "basic needs juice money",
        "message": "I want juice but I don't have money to buy",
        "expected": {"health_daily_care_agent"},
        "not_expected": set(),
        "must_mention": ["juice", "money"],
        "must_not_mention": ["chest pain", "emergency", "dose"],
    },
    {
        "name": "2 unknown email link",
        "message": "Someone emailed me but I don't know them, should I click the link?",
        "expected": {"safety_agent"},
        "not_expected": {"health_daily_care_agent", "emotional_social_agent"},
        "must_mention": ["email", "link", "click"],
        "must_not_mention": ["chest pain", "medicine", "dose"],
    },
    {
        "name": "5 dizziness after standing",
        "message": "I feel dizzy after standing up",
        "expected": {"health_daily_care_agent"},
        "not_expected": {"safety_agent"},
        "must_mention": ["sit", "chest pain"],
        "must_not_mention": ["bank", "otp", "dose"],
    },
    {
        "name": "6 forgot blood pressure medicine",
        "message": "I forgot my blood pressure medicine",
        "expected": {"health_daily_care_agent"},
        "not_expected": {"safety_agent", "emotional_social_agent"},
        "must_mention": ["extra dose", "pharmacist"],
        "must_not_mention": ["bank", "email", "chest pain"],
    },
]


# Run all examples directly without requiring pytest.
def main() -> None:
    # First pass verifies single-turn routing and response specificity.
    for case in TEST_CASES:
        response = run_elderguard_workflow(
            ChatRequest(message=case["message"], user_id=f"test_{case['name']}")
        )
        agents = set(response.activated_agents)
        text = response.final_message.lower()
        print(f"\n{case['name']}")
        print("Agents:", response.activated_agents)
        print("Specificity:", response.specificity_score)
        print("Questions:", response.follow_up_questions)
        print(response.final_message)

        assert case["expected"].issubset(agents), f"Missing expected agents: {case['expected'] - agents}"
        assert not (case["not_expected"] & agents), f"Unexpected agents: {case['not_expected'] & agents}"
        assert response.final_message.strip(), "final_message should not be empty"
        assert response.specificity_score >= 45, "specificity_score should be usable"
        for phrase in case["must_mention"]:
            assert phrase in text, f"Expected phrase not found: {phrase}"
        for phrase in case.get("must_not_mention", []):
            assert phrase not in text, f"Unrelated phrase found: {phrase}"

    # Follow-up examples verify that memory and topic continuity work.
    follow_up_cases = [
        {
            "name": "1 hunger plus juice follow-up",
            "turns": [
                "I am hungry",
                "no, I have juice in the fridge",
            ],
            "expected_agent": "health_daily_care_agent",
            "must_mention": ["juice"],
        },
        {
            "name": "2 suspicious email sender follow-up",
            "turns": [
                "Someone emailed me but I don't know them, should I click the link?",
                "my daughter's friend",
            ],
            "expected_agent": "safety_agent",
            "must_mention": ["daughter", "friend", "click"],
        },
        {
            "name": "follow-up medication timing",
            "turns": [
                "I forgot my blood pressure medicine",
                "it was supposed to be this morning",
            ],
            "expected_agent": "health_daily_care_agent",
            "must_mention": ["medication"],
        },
        {
            "name": "follow-up health symptom",
            "turns": [
                "I feel dizzy after standing up",
                "I am alone",
            ],
            "expected_agent": "health_daily_care_agent",
            "must_mention": ["alone"],
        },
        {
            "name": "follow-up loneliness",
            "turns": [
                "I feel lonely because my friends are busy",
                "yes help me write a message",
            ],
            "expected_agent": "action_agent",
            "must_mention": ["message"],
        },
    ]

    # Run each follow-up conversation in a separate conversation id.
    for index, case in enumerate(follow_up_cases):
        conversation_id = f"test_followup_{index}"
        response = None
        for turn in case["turns"]:
            response = run_elderguard_workflow(
                ChatRequest(message=turn, user_id="test_followup_user", conversation_id=conversation_id)
            )
        assert response is not None
        text = response.final_message.lower()
        print(f"\n{case['name']}")
        print("Agents:", response.activated_agents)
        print(response.final_message)
        assert case["expected_agent"] in response.activated_agents
        for phrase in case["must_mention"]:
            assert phrase in text, f"Expected follow-up phrase not found: {phrase}"


# Allow this file to act as a standalone smoke test script.
if __name__ == "__main__":
    main()
