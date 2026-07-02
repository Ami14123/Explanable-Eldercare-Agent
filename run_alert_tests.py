import os

os.environ["LLM_MODE"] = "mock"

from app.config import get_settings
from app.memory import init_db
from app.graph import run_elderguard_workflow
from app.schemas import ChatRequest


def check(
    message: str,
    expected_required: bool,
    expected_type: str,
    expected_level: str,
    *,
    expected_requested_contact: bool | None = None,
) -> None:
    response = run_elderguard_workflow(
        ChatRequest(
            message=message,
            user_id="alert_test_user",
            conversation_id=message[:24].replace(" ", "_"),
        )
    )
    alert = response.developer_state.get("alert_decision") or response.care_plan.get("alert_decision", {})
    actual = (
        bool(alert.get("alert_required")),
        str(alert.get("alert_type")),
        str(alert.get("alert_level")),
    )
    expected = (expected_required, expected_type, expected_level)
    if actual != expected:
        raise AssertionError(f"{message!r}: expected {expected}, got {actual}. Alert={alert}")
    if expected_requested_contact is not None and bool(alert.get("user_requested_contact")) != expected_requested_contact:
        raise AssertionError(f"{message!r}: expected user_requested_contact={expected_requested_contact}, got {alert}")
    if expected_requested_contact is True and bool(alert.get("human_confirmation_required")) is not False:
        raise AssertionError(f"{message!r}: user-requested contact should not require extra human confirmation. Alert={alert}")
    print(f"PASS: {message} -> {alert}")


def main() -> None:
    get_settings.cache_clear()
    init_db()
    check("Someone emailed me a link asking me to send money", True, "fraud", "medium")
    check("I ate expired food and now my stomach hurts", True, "health", "medium")
    check("I feel dizzy after standing up", False, "none", "low")
    check("I feel dizzy after standing and I have chest pain", True, "emergency", "high")
    check("I forgot if I took my medication dose", True, "caregiver", "medium")
    check("I feel lonely and want to hurt myself", True, "emergency", "high")
    check("I am hungry", False, "none", "low")
    check("I feel dizzy and I am about to fall, call my caregiver", True, "emergency", "high", expected_requested_contact=True)
    check("I am losing my balance and I might fall", True, "emergency", "high")
    check("Please call my caregiver", True, "caregiver", "medium", expected_requested_contact=True)
    check("I feel slightly dizzy but I am sitting safely", False, "none", "low")
    print("All alert tests passed.")


if __name__ == "__main__":
    main()
