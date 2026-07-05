# HTTP-level agent regression tests for a running FastAPI backend.
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


BACKEND_URL = "http://127.0.0.1:8000"


# Expected behavior for one conversation turn.
@dataclass
class TurnExpectation:
    message: str
    expected_agents: set[str] = field(default_factory=set)
    not_expected_agents: set[str] = field(default_factory=set)
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    expected_topic: str = ""


# A scenario groups turns that share user and conversation ids.
@dataclass
class TestScenario:
    name: str
    turns: list[TurnExpectation]


# Scenario list covers topic switching, follow-ups, and specialist routing.
SCENARIOS = [
    TestScenario(
        name="Food need switches away from old email topic",
        turns=[
            TurnExpectation(
                "Someone emailed me but I don't know them, should I click the link?",
                expected_agents={"safety_agent"},
                not_expected_agents={"health_daily_care_agent"},
                must_mention=["email", "link"],
                expected_topic="suspicious_email",
            ),
            TurnExpectation(
                "my daughter's friend",
                expected_agents={"safety_agent"},
                must_mention=["daughter", "friend"],
                expected_topic="suspicious_email",
            ),
            TurnExpectation(
                "yes, it asks me to log in",
                expected_agents={"safety_agent"},
                must_mention=["log in"],
                must_not_mention=["hungry"],
                expected_topic="suspicious_email",
            ),
            TurnExpectation(
                "I am hungry",
                expected_agents={"health_daily_care_agent"},
                not_expected_agents={"safety_agent"},
                must_mention=["hungry"],
                must_not_mention=["email", "log in", "click"],
                expected_topic="basic_need_support",
            ),
        ],
    ),
    TestScenario(
        name="Email asks to lend money",
        turns=[
            TurnExpectation(
                "Someone emailed me but I don't know them, should I click the link?",
                expected_agents={"safety_agent"},
                must_mention=["email", "link"],
                expected_topic="suspicious_email",
            ),
            TurnExpectation(
                "my daughter's friend",
                expected_agents={"safety_agent"},
                must_mention=["daughter", "friend"],
                expected_topic="suspicious_email",
            ),
            TurnExpectation(
                "yes, ask me to lend her money",
                expected_agents={"safety_agent"},
                must_mention=["money", "daughter"],
                must_not_mention=["log in", "password"],
                expected_topic="suspicious_email",
            ),
        ],
    ),
    TestScenario(
        name="Hunger and juice follow-up",
        turns=[
            TurnExpectation(
                "I am hungry",
                expected_agents={"health_daily_care_agent"},
                must_mention=["hungry"],
                expected_topic="basic_need_support",
            ),
            TurnExpectation(
                "no, I have juice in the fridge",
                expected_agents={"health_daily_care_agent"},
                not_expected_agents={"safety_agent"},
                must_mention=["juice"],
                expected_topic="basic_need_support",
            ),
        ],
    ),
    TestScenario(
        name="Health only",
        turns=[
            TurnExpectation(
                "I feel dizzy after standing up",
                expected_agents={"health_daily_care_agent"},
                not_expected_agents={"safety_agent"},
                must_mention=["dizzy"],
                must_not_mention=["bank", "email", "dose"],
                expected_topic="health_symptom",
            )
        ],
    ),
    TestScenario(
        name="Medication only",
        turns=[
            TurnExpectation(
                "I forgot my blood pressure medicine this morning",
                expected_agents={"health_daily_care_agent"},
                not_expected_agents={"safety_agent"},
                must_mention=["medication"],
                must_not_mention=["email", "bank", "chest pain"],
                expected_topic="medication",
            )
        ],
    ),
    TestScenario(
        name="Companion only",
        turns=[
            TurnExpectation(
                "I feel lonely because my children are busy",
                expected_agents={"emotional_social_agent"},
                not_expected_agents={"safety_agent", "health_daily_care_agent"},
                must_mention=["lonely"],
                must_not_mention=["bank", "dose", "emergency"],
                expected_topic="emotional_support",
            )
        ],
    ),
]


# Send one chat request to the local backend.
def post_chat(message: str, user_id: str, conversation_id: str) -> dict[str, Any]:
    # Build the JSON payload exactly like a frontend client would.
    payload = {
        "message": message,
        "user_id": user_id,
        "conversation_id": conversation_id,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BACKEND_URL}/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


# Compare one response against the expected agents, topic, and phrases.
def check_turn(result: dict[str, Any], expectation: TurnExpectation) -> list[str]:
    failures: list[str] = []
    agents = set(result.get("activated_agents", []))
    final_message = str(result.get("final_message", "")).lower()
    topic = str(result.get("current_topic", ""))

    # Collect all failures so the command can report every mismatch.
    missing_agents = expectation.expected_agents - agents
    unexpected_agents = expectation.not_expected_agents & agents
    if missing_agents:
        failures.append(f"missing agents: {sorted(missing_agents)}")
    if unexpected_agents:
        failures.append(f"unexpected agents: {sorted(unexpected_agents)}")
    if expectation.expected_topic and topic != expectation.expected_topic:
        failures.append(f"expected topic {expectation.expected_topic!r}, got {topic!r}")
    if not final_message.strip():
        failures.append("final message is empty")
    for phrase in expectation.must_mention:
        if phrase.lower() not in final_message:
            failures.append(f"final message should mention {phrase!r}")
    for phrase in expectation.must_not_mention:
        if phrase.lower() in final_message:
            failures.append(f"final message should not mention {phrase!r}")
    return failures


# Run all HTTP scenarios and return a shell-friendly exit code.
def main() -> int:
    run_id = int(time.time())
    all_failures: list[str] = []

    print("ElderGuard AI automatic agent test")
    print(f"Backend: {BACKEND_URL}")
    print()

    for scenario_index, scenario in enumerate(SCENARIOS, start=1):
        # Use unique ids so previous manual runs do not pollute memory.
        user_id = f"auto_test_user_{run_id}_{scenario_index}"
        conversation_id = f"auto_test_conversation_{run_id}_{scenario_index}"
        print(f"[{scenario_index}] {scenario.name}")

        for turn_index, expectation in enumerate(scenario.turns, start=1):
            try:
                result = post_chat(expectation.message, user_id, conversation_id)
            except urllib.error.URLError as exc:
                # Missing backend is a setup issue, not a product assertion failure.
                print("  Backend is not reachable.")
                print("  Start FastAPI first with: uvicorn app.main:app --reload")
                print(f"  Details: {exc}")
                return 2

            failures = check_turn(result, expectation)
            agents = ", ".join(result.get("activated_agents", [])) or "none"
            topic = result.get("current_topic", "")
            print(f"  Turn {turn_index}: {expectation.message}")
            print(f"    agents: {agents}")
            print(f"    topic: {topic}")
            print(f"    reply: {result.get('final_message', '')[:180].replace(chr(10), ' ')}")
            if failures:
                for failure in failures:
                    all_failures.append(f"{scenario.name} / turn {turn_index}: {failure}")
                    print(f"    FAIL: {failure}")
            else:
                print("    PASS")
        print()

    if all_failures:
        # Print every failure before returning nonzero for automation.
        print("Some tests failed:")
        for failure in all_failures:
            print(f"- {failure}")
        return 1

    print("All automatic agent tests passed.")
    return 0


# Make the script runnable from the command line.
if __name__ == "__main__":
    sys.exit(main())
