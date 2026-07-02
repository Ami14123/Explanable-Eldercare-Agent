from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

MemoryType = Literal[
    "working_memory",
    "episodic_memory",
    "long_term_profile",
    "semantic_knowledge",
    "do_not_save",
]


def _contains(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _expires_at(ttl_minutes: int | None) -> str | None:
    if ttl_minutes is None:
        return None
    return (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()


def _item(
    memory_type: MemoryType,
    content: str,
    *,
    ttl_minutes: int | None,
    confidence: float,
    confirmed: bool,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "memory_type": memory_type,
        "content": content,
        "ttl_minutes": ttl_minutes,
        "expires_at": _expires_at(ttl_minutes),
        "confidence": confidence,
        "confirmed": confirmed,
        "reason": reason,
        "metadata": metadata or {},
    }


def classify_memory(
    message: str,
    *,
    final_message_source: str = "",
    llm_error: dict[str, Any] | None = None,
    router_decision: dict[str, Any] | None = None,
    alert_decision: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Classify user information before saving it as memory."""
    text = message.lower()
    router_decision = router_decision or {}
    alert_decision = alert_decision or {}
    if llm_error or final_message_source in {"conversation_agent_error", "workflow_error"}:
        return [
            _item(
                "do_not_save",
                "LLM/provider error turn should not become user memory.",
                ttl_minutes=0,
                confidence=1.0,
                confirmed=True,
                reason="Never save LLM errors as user memory.",
            )
        ]

    memories: list[dict[str, Any]] = []

    if _contains(text, ["password", "api key", "secret key", "token is", "my otp", "otp is"]):
        memories.append(
            _item(
                "do_not_save",
                "Sensitive credential-like content.",
                ttl_minutes=0,
                confidence=0.95,
                confirmed=True,
                reason="Avoid saving secrets, OTPs, passwords, or keys.",
            )
        )
        return memories

    available_foods = [
        food
        for food in ["bread", "rice", "soup", "eggs", "egg", "banana", "crackers", "yogurt", "juice", "water", "leftovers"]
        if food in text and _contains(text, ["have", "has", "there is", "in the fridge", "available"])
    ]
    if available_foods:
        memories.append(
            _item(
                "working_memory",
                f"Food/drink available now: {', '.join(sorted(set(available_foods)))}",
                ttl_minutes=30,
                confidence=0.85,
                confirmed=True,
                reason="Food available now is short-lived working memory only.",
                metadata={"food_available": sorted(set(available_foods))},
            )
        )

    symptoms = [term for term in ["dizzy", "dizziness", "weak", "pain", "stomach pain", "nausea", "confusion", "chest pain"] if term in text]
    if symptoms:
        memories.append(
            _item(
                "working_memory",
                f"Current symptoms mentioned: {', '.join(sorted(set(symptoms)))}",
                ttl_minutes=120,
                confidence=0.75,
                confirmed=True,
                reason="Current symptoms should not become profile memory.",
                metadata={"symptoms": sorted(set(symptoms))},
            )
        )

    incident_terms = ["scam", "email", "link", "fall", "fell", "almost fell", "missed", "forgot", "medication", "medicine", "pill", "dose"]
    if _contains(text, incident_terms):
        memories.append(
            _item(
                "episodic_memory",
                f"Incident reported: {message.strip()[:240]}",
                ttl_minutes=None,
                confidence=0.75,
                confirmed=True,
                reason="Scam, fall, and medication incidents are episodic memory.",
                metadata={
                    "active_topic": router_decision.get("active_topic", ""),
                    "alert_type": alert_decision.get("alert_type", "none"),
                    "alert_level": alert_decision.get("alert_level", "low"),
                },
            )
        )

    chronic_markers = ["i have diabetes", "i have hypertension", "i have heart disease", "i have asthma", "doctor said i have", "diagnosed with"]
    if _contains(text, chronic_markers):
        memories.append(
            _item(
                "long_term_profile",
                f"Confirmed health profile fact: {message.strip()[:240]}",
                ttl_minutes=None,
                confidence=0.9,
                confirmed=True,
                reason="User confirmed chronic condition.",
                metadata={"profile_category": "health_condition"},
            )
        )

    if _contains(text, ["my caregiver", "my daughter", "my son", "my wife", "my husband", "my neighbor", "my friend"]) and _contains(
        text, ["is", "called", "name", "phone", "contact", "can help", "takes care"]
    ):
        memories.append(
            _item(
                "long_term_profile",
                f"Caregiver/family contact fact: {message.strip()[:240]}",
                ttl_minutes=None,
                confidence=0.85,
                confirmed=True,
                reason="Caregiver or family contact belongs in long-term profile.",
                metadata={"profile_category": "support_contact"},
            )
        )

    if "what is" in text or "how do i" in text or "is it safe" in text or "good for" in text:
        memories.append(
            _item(
                "semantic_knowledge",
                f"General care knowledge question: {message.strip()[:240]}",
                ttl_minutes=None,
                confidence=0.65,
                confirmed=False,
                reason="General care knowledge should not become a user profile fact.",
            )
        )

    if not memories:
        memories.append(
            _item(
                "do_not_save",
                "No durable user memory extracted.",
                ttl_minutes=0,
                confidence=0.6,
                confirmed=False,
                reason="Message did not contain a useful memory fact.",
            )
        )
    return memories


def profile_updates_from_memories(memories: list[dict[str, Any]]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for memory in memories:
        if memory.get("memory_type") != "long_term_profile" or not memory.get("confirmed"):
            continue
        category = memory.get("metadata", {}).get("profile_category", "general")
        updates.setdefault(category, [])
        updates[category].append(
            {
                "content": memory.get("content", ""),
                "confidence": memory.get("confidence", 0),
                "confirmed": memory.get("confirmed", False),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return updates
