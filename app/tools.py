from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def get_safe_route_tool(
    current_location: str,
    destination: str,
    mobility_note: str = "",
) -> dict[str, Any]:
    """
    Return a mock safe-route plan.

    Vietnamese note: Tool nay chi mo phong Safe Navigation Agent, khong goi
    Google Maps hay dich vu dinh vi that.
    """

    missing = []
    if not current_location.strip():
        missing.append("current_location")
    if not destination.strip():
        missing.append("destination")

    return {
        "tool": "get_safe_route_tool",
        "mode": "mock",
        "external_service_called": False,
        "requires_location_permission": True,
        "status": "needs_more_information" if missing else "prepared",
        "missing_fields": missing,
        "route_summary": (
            ""
            if missing
            else f"Mock route from {current_location.strip()} to {destination.strip()}."
        ),
        "safety_notes": [
            "Use well-lit and familiar paths when possible.",
            "Ask a trusted person for help if confused, dizzy, or lost.",
            "This prototype does not provide real navigation directions.",
        ],
        "mobility_note": mobility_note.strip(),
    }


def send_caregiver_alert_tool(
    caregiver_name: str,
    message: str,
    alert_level: str = "medium",
) -> dict[str, Any]:
    """Prepare a caregiver alert without sending it externally."""

    return {
        "tool": "send_caregiver_alert_tool",
        "mode": "mock",
        "external_notification_sent": False,
        "status": "prepared",
        "caregiver_name": caregiver_name.strip() or "caregiver",
        "alert_level": alert_level,
        "caregiver_message": message.strip(),
        "prototype_notice": "Alert text was prepared only. No SMS, call, or email was sent.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def create_reminder_tool(
    reminder_text: str,
    scheduled_for: str = "",
) -> dict[str, Any]:
    """Create a mock reminder object without scheduling a real notification."""

    return {
        "tool": "create_reminder_tool",
        "mode": "mock",
        "external_notification_sent": False,
        "status": "prepared" if reminder_text.strip() else "needs_more_information",
        "reminder_text": reminder_text.strip(),
        "scheduled_for": scheduled_for.strip(),
        "prototype_notice": "Reminder was prepared only. No device notification was scheduled.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
