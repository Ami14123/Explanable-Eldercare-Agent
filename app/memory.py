# SQLite persistence layer for chat logs, memory, and profile state.
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse


# Store one full interaction log row for admin audit and technical trace.
CREATE_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS interaction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT NOT NULL,
    message TEXT NOT NULL,
    final_message TEXT NOT NULL DEFAULT '',
    activated_agents TEXT NOT NULL,
    overall_risk TEXT NOT NULL,
    main_concern TEXT NOT NULL,
    advice TEXT NOT NULL,
    caregiver_alert TEXT NOT NULL,
    next_step TEXT NOT NULL,
    raw_json TEXT NOT NULL
)
"""

# Store raw conversation turns so follow-up replies have recent context.
CREATE_MESSAGES_SQL = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    message TEXT NOT NULL
)
"""

# Store a compact user summary that can be reused across turns.
CREATE_SUMMARIES_SQL = """
CREATE TABLE IF NOT EXISTS user_summaries (
    user_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
)
"""

# Store structured conversation state per user and conversation id.
CREATE_CONVERSATION_STATES_SQL = """
CREATE TABLE IF NOT EXISTS conversation_states (
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, conversation_id)
)
"""

# Store classified care events separately from the raw interaction log.
CREATE_CARE_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS care_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_json TEXT NOT NULL
)
"""

# Store durable profile facts extracted from confirmed long-term memories.
CREATE_USER_PROFILE_SQL = """
CREATE TABLE IF NOT EXISTS user_profile (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


# Resolve and create the configured SQLite database path.
def _db_path() -> Path:
    settings = get_settings()
    path = settings.sqlite_file
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Open a SQLite connection that returns rows by column name.
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


# Create or migrate all local memory tables.
def init_db() -> None:
    with get_connection() as conn:
        # Create every table idempotently so startup is safe to repeat.
        conn.execute(CREATE_LOGS_SQL)
        conn.execute(CREATE_MESSAGES_SQL)
        conn.execute(CREATE_SUMMARIES_SQL)
        conn.execute(CREATE_CONVERSATION_STATES_SQL)
        conn.execute(CREATE_CARE_EVENTS_SQL)
        conn.execute(CREATE_USER_PROFILE_SQL)
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(interaction_logs)").fetchall()
        }
        # Keep older databases compatible with the current log schema.
        if "final_message" not in columns:
            conn.execute(
                "ALTER TABLE interaction_logs ADD COLUMN final_message TEXT NOT NULL DEFAULT ''"
            )
        conn.commit()


# Load saved graph conversation state for one user conversation.
def get_conversation_state(user_id: str, conversation_id: str = "default") -> dict[str, Any]:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT state_json
            FROM conversation_states
            WHERE user_id = ? AND conversation_id = ?
            """,
            (user_id, conversation_id or "default"),
        ).fetchone()
    # Missing state means this is the first turn for that conversation.
    return json.loads(row["state_json"]) if row else {}


# Save graph conversation state so future turns can continue the topic.
def save_conversation_state(user_id: str, conversation_id: str, state: dict[str, Any]) -> None:
    init_db()
    updated_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        # Upsert keeps one current state row per user/conversation pair.
        conn.execute(
            """
            INSERT INTO conversation_states (user_id, conversation_id, state_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, conversation_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (user_id, conversation_id or "default", json.dumps(state, ensure_ascii=False), updated_at),
        )
        conn.commit()


# Load durable profile facts for one user.
def get_user_profile(user_id: str) -> dict[str, Any]:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT profile_json FROM user_profile WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return json.loads(row["profile_json"]) if row else {}


# Save the durable profile after memory classification updates it.
def save_user_profile(user_id: str, profile: dict[str, Any]) -> None:
    init_db()
    updated_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        # Upsert preserves a single profile document per user.
        conn.execute(
            """
            INSERT INTO user_profile (user_id, profile_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at
            """,
            (user_id, json.dumps(profile, ensure_ascii=False), updated_at),
        )
        conn.commit()


# Save one structured care event and return its database id.
def save_care_event(
    user_id: str,
    conversation_id: str,
    event_type: str,
    event: dict[str, Any],
) -> int:
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        # Store the event JSON so new event types do not need schema changes.
        cursor = conn.execute(
            """
            INSERT INTO care_events (
                timestamp, user_id, conversation_id, event_type, event_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                user_id,
                conversation_id or "default",
                event_type,
                json.dumps(event, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


# Return recent care events for context and family summaries.
def get_relevant_care_events(user_id: str, limit: int = 8) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, user_id, conversation_id, event_type, event_json
            FROM care_events
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        # Decode event_json into a nested event object for callers.
        item = dict(row)
        item["event"] = json.loads(item.pop("event_json"))
        events.append(item)
    return events


# Save a single user or assistant chat message.
def save_message(user_id: str, conversation_id: str, role: str, message: str) -> int:
    # Vietnamese note: Save memory luu tung turn user/assistant vao SQLite local.
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        # The table constraint keeps role limited to user or assistant.
        cursor = conn.execute(
            """
            INSERT INTO conversation_messages (
                timestamp, user_id, conversation_id, role, message
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, user_id, conversation_id or "default", role, message),
        )
        conn.commit()
        return int(cursor.lastrowid)


# Load recent conversation turns in chronological order.
def get_recent_conversation(
    user_id: str,
    conversation_id: str = "default",
    limit: int = 10,
) -> list[dict[str, Any]]:
    # Vietnamese note: Memory loading lay cac turn gan nhat de user khong phai reset moi lan.
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, user_id, conversation_id, role, message
            FROM conversation_messages
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, conversation_id or "default", limit),
        ).fetchall()
    # SQL returns newest first, so reverse for natural conversation order.
    return [dict(row) for row in reversed(rows)]


# Load the saved text summary for one user.
def get_user_summary(user_id: str) -> str:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT summary FROM user_summaries WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return str(row["summary"]) if row else ""


# Find the latest assistant question to help detect follow-up answers.
def get_previous_assistant_question(user_id: str, conversation_id: str = "default") -> str:
    # Prefer recent message history because it is closest to the current turn.
    history = get_recent_conversation(user_id, conversation_id, limit=10)
    for item in reversed(history):
        if item.get("role") == "assistant":
            message = str(item.get("message", "")).strip()
            if "?" in message:
                return message
    # Fall back to interaction logs when individual message history is missing.
    memories = get_recent_memory(user_id, limit=6)
    for item in memories:
        final_message = str(item.get("final_message", "")).strip()
        if "?" in final_message:
            return final_message
    return ""


# Infer a rough current topic from recent conversation text.
def get_current_topic(user_id: str, conversation_id: str = "default") -> str:
    history = get_recent_conversation(user_id, conversation_id, limit=10)
    text = " ".join(str(item.get("message", "")) for item in history).lower()
    # Keyword checks keep topic detection deterministic and cheap.
    if "email" in text or "link" in text:
        return "suspicious_email"
    if "medicine" in text or "medication" in text or "dose" in text:
        return "medication"
    if "dizzy" in text or "fall" in text or "chest pain" in text:
        return "health_symptom"
    if "hungry" in text or "juice" in text or "food" in text:
        return "basic_need_support"
    if "lonely" in text or "sad" in text:
        return "emotional_support"
    return ""


# Append new information to the compact user summary.
def update_user_summary(user_id: str, new_information: str) -> None:
    init_db()
    # Empty updates should not create or modify a summary row.
    if not new_information.strip():
        return
    current = get_user_summary(user_id)
    updated_at = datetime.now(timezone.utc).isoformat()
    notes = [part.strip() for part in [current, new_information.strip()] if part.strip()]
    summary = " ".join(notes)
    if len(summary) > 2000:
        # Keep the newest information when the summary grows too long.
        summary = summary[-2000:]
    with get_connection() as conn:
        # Upsert lets the first and later summaries share one code path.
        conn.execute(
            """
            INSERT INTO user_summaries (user_id, summary, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                summary = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (user_id, summary, updated_at),
        )
        conn.commit()


# Save the complete chat interaction for audit and admin views.
def save_interaction(
    request: ChatRequest,
    response: ChatResponse,
    raw_json: dict[str, Any],
) -> int:
    # Vietnamese note: Interaction log dung cho technical trace va admin audit trong prototype.
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        # Store nested lists and objects as JSON strings in SQLite.
        cursor = conn.execute(
            """
            INSERT INTO interaction_logs (
                timestamp, user_id, message, final_message, activated_agents, overall_risk,
                main_concern, advice, caregiver_alert, next_step, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                request.user_id,
                request.message,
                response.final_message,
                json.dumps(response.activated_agents),
                response.overall_risk,
                response.main_concern,
                response.advice,
                response.caregiver_alert,
                response.next_step,
                json.dumps(raw_json, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


# Return recent interaction logs with JSON fields decoded.
def recent_logs(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM interaction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    logs: list[dict[str, Any]] = []
    for row in rows:
        # Decode JSON fields so admin routes can return normal dictionaries.
        item = dict(row)
        item["activated_agents"] = json.loads(item["activated_agents"])
        item["raw_json"] = json.loads(item["raw_json"])
        logs.append(item)
    return logs


# Return recent interactions for one user as memory context.
def get_recent_memory(user_id: str, limit: int = 8) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM interaction_logs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    memories: list[dict[str, Any]] = []
    for row in rows:
        # Match recent_logs shape so callers can reuse the same fields.
        item = dict(row)
        item["activated_agents"] = json.loads(item["activated_agents"])
        item["raw_json"] = json.loads(item["raw_json"])
        memories.append(item)
    return memories


# Summarize repeated risk patterns from recent user memory.
def summarize_memory(user_id: str) -> dict[str, Any]:
    memories = get_recent_memory(user_id, limit=12)
    # Counters track patterns that should raise attention across turns.
    counters = {
        "repeated_dizziness": 0,
        "repeated_missed_medication": 0,
        "repeated_scam_attempts": 0,
        "repeated_loneliness": 0,
    }

    # Count repeated concerns using message keywords and activated agents.
    for item in memories:
        message = item.get("message", "").lower()
        agents = set(item.get("activated_agents", []))
        if "dizzy" in message or "dizziness" in message or "fall" in message or "health_agent" in agents:
            counters["repeated_dizziness"] += 1
        if "forgot" in message or "missed" in message or "medication_agent" in agents:
            counters["repeated_missed_medication"] += 1
        if "bank" in message or "otp" in message or "transfer" in message or "fraud_agent" in agents:
            counters["repeated_scam_attempts"] += 1
        if "lonely" in message or "friend" in message or "alone" in message or "companion_agent" in agents:
            counters["repeated_loneliness"] += 1

    # Convert repeated patterns into readable risk adjustment explanations.
    risk_adjustments: list[str] = []
    if counters["repeated_dizziness"] >= 2:
        risk_adjustments.append("Repeated dizziness or fall-related messages increase health priority.")
    if counters["repeated_missed_medication"] >= 2:
        risk_adjustments.append("Repeated missed-medication messages increase medication priority.")
    if counters["repeated_scam_attempts"] >= 2:
        risk_adjustments.append("Repeated scam-related messages increase safety priority.")
    if counters["repeated_loneliness"] >= 2:
        risk_adjustments.append("Repeated loneliness messages increase companion priority.")

    # Include a short timeline so the UI can explain where the pattern came from.
    return {
        "user_id": user_id,
        "recent_count": len(memories),
        "counters": counters,
        "risk_adjustments": risk_adjustments,
        "timeline": [
            {
                "id": item.get("id"),
                "timestamp": item.get("timestamp"),
                "message": item.get("message"),
                "overall_risk": item.get("overall_risk"),
                "activated_agents": item.get("activated_agents", []),
            }
            for item in memories
        ],
    }
