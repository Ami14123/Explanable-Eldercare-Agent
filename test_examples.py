from app.graph import run_elderguard_workflow
from app.schemas import ChatRequest


TEST_CASES = [
    {
        "name": "3 normal daily life question",
        "message": "What can I make for lunch with rice and eggs?",
        "expected": set(),
        "not_expected": {"health_agent", "medication_agent", "fraud_agent", "companion_agent", "basic_needs_agent"},
        "must_mention": ["one"],
        "must_not_mention": ["chest pain", "otp", "dose"],
    },
    {
        "name": "4 lonely family busy",
        "message": "I feel lonely because my family is busy",
        "expected": {"companion_agent"},
        "not_expected": {"health_agent", "medication_agent", "fraud_agent", "basic_needs_agent"},
        "must_mention": ["lonely"],
        "must_not_mention": ["doctor", "emergency", "dose", "bank"],
    },
    {
        "name": "basic needs juice money",
        "message": "I want juice but I don't have money to buy",
        "expected": {"basic_needs_agent"},
        "not_expected": {"health_agent", "medication_agent", "fraud_agent"},
        "must_mention": ["juice", "money"],
        "must_not_mention": ["chest pain", "emergency", "dose"],
    },
    {
        "name": "2 unknown email link",
        "message": "Someone emailed me but I don't know them, should I click the link?",
        "expected": {"fraud_agent"},
        "not_expected": {"health_agent", "medication_agent", "companion_agent", "basic_needs_agent"},
        "must_mention": ["email", "link", "click"],
        "must_not_mention": ["chest pain", "medicine", "dose"],
    },
    {
        "name": "5 dizziness after standing",
        "message": "I feel dizzy after standing up",
        "expected": {"health_agent"},
        "not_expected": {"medication_agent", "fraud_agent", "basic_needs_agent"},
        "must_mention": ["sit", "chest pain"],
        "must_not_mention": ["bank", "otp", "dose"],
    },
    {
        "name": "6 forgot blood pressure medicine",
        "message": "I forgot my blood pressure medicine",
        "expected": {"medication_agent"},
        "not_expected": {"health_agent", "fraud_agent", "companion_agent", "basic_needs_agent"},
        "must_mention": ["extra dose", "pharmacist"],
        "must_not_mention": ["bank", "email", "chest pain"],
    },
]


def main() -> None:
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
        assert response.specificity_score >= 70, "specificity_score should be strong"
        for phrase in case["must_mention"]:
            assert phrase in text, f"Expected phrase not found: {phrase}"
        for phrase in case.get("must_not_mention", []):
            assert phrase not in text, f"Unrelated phrase found: {phrase}"

    follow_up_cases = [
        {
            "name": "1 hunger plus juice follow-up",
            "turns": [
                "I am hungry",
                "no, I have juice in the fridge",
            ],
            "expected_agent": "basic_needs_agent",
            "must_mention": ["juice", "hungry"],
        },
        {
            "name": "2 suspicious email sender follow-up",
            "turns": [
                "Someone emailed me but I don't know them, should I click the link?",
                "my daughter's friend",
            ],
            "expected_agent": "fraud_agent",
            "must_mention": ["daughter", "friend", "click"],
        },
        {
            "name": "follow-up medication timing",
            "turns": [
                "I forgot my blood pressure medicine",
                "it was supposed to be this morning",
            ],
            "expected_agent": "medication_agent",
            "must_mention": ["extra dose"],
        },
        {
            "name": "follow-up health symptom",
            "turns": [
                "I feel dizzy after standing up",
                "I am alone",
            ],
            "expected_agent": "health_agent",
            "must_mention": ["dizziness"],
        },
        {
            "name": "follow-up loneliness",
            "turns": [
                "I feel lonely because my friends are busy",
                "yes help me write a message",
            ],
            "expected_agent": "companion_agent",
            "must_mention": ["message"],
        },
    ]

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


if __name__ == "__main__":
    main()
