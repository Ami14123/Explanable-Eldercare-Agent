import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse


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

CREATE_SUMMARIES_SQL = """
CREATE TABLE IF NOT EXISTS user_summaries (
    user_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
)
"""

CREATE_CONVERSATION_STATES_SQL = """
CREATE TABLE IF NOT EXISTS conversation_states (
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, conversation_id)
)
"""

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

CREATE_USER_PROFILE_SQL = """
CREATE TABLE IF NOT EXISTS user_profile (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _db_path() -> Path:
    settings = get_settings()
    path = settings.sqlite_file
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
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
        if "final_message" not in columns:
            conn.execute(
                "ALTER TABLE interaction_logs ADD COLUMN final_message TEXT NOT NULL DEFAULT ''"
            )
        conn.commit()


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
    return json.loads(row["state_json"]) if row else {}


def save_conversation_state(user_id: str, conversation_id: str, state: dict[str, Any]) -> None:
    init_db()
    updated_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
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


def get_user_profile(user_id: str) -> dict[str, Any]:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT profile_json FROM user_profile WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return json.loads(row["profile_json"]) if row else {}


def save_user_profile(user_id: str, profile: dict[str, Any]) -> None:
    init_db()
    updated_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
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


def save_care_event(
    user_id: str,
    conversation_id: str,
    event_type: str,
    event: dict[str, Any],
) -> int:
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
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
        item = dict(row)
        item["event"] = json.loads(item.pop("event_json"))
        events.append(item)
    return events


def save_message(user_id: str, conversation_id: str, role: str, message: str) -> int:
    # Vietnamese note: Save memory luu tung turn user/assistant vao SQLite local.
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
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
    return [dict(row) for row in reversed(rows)]


def get_user_summary(user_id: str) -> str:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT summary FROM user_summaries WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return str(row["summary"]) if row else ""


def get_previous_assistant_question(user_id: str, conversation_id: str = "default") -> str:
    history = get_recent_conversation(user_id, conversation_id, limit=10)
    for item in reversed(history):
        if item.get("role") == "assistant":
            message = str(item.get("message", "")).strip()
            if "?" in message:
                return message
    memories = get_recent_memory(user_id, limit=6)
    for item in memories:
        final_message = str(item.get("final_message", "")).strip()
        if "?" in final_message:
            return final_message
    return ""


def get_current_topic(user_id: str, conversation_id: str = "default") -> str:
    history = get_recent_conversation(user_id, conversation_id, limit=10)
    text = " ".join(str(item.get("message", "")) for item in history).lower()
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


def update_user_summary(user_id: str, new_information: str) -> None:
    init_db()
    if not new_information.strip():
        return
    current = get_user_summary(user_id)
    updated_at = datetime.now(timezone.utc).isoformat()
    notes = [part.strip() for part in [current, new_information.strip()] if part.strip()]
    summary = " ".join(notes)
    if len(summary) > 2000:
        summary = summary[-2000:]
    with get_connection() as conn:
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


def save_interaction(
    request: ChatRequest,
    response: ChatResponse,
    raw_json: dict[str, Any],
) -> int:
    # Vietnamese note: Interaction log dung cho technical trace va admin audit trong prototype.
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
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
        item = dict(row)
        item["activated_agents"] = json.loads(item["activated_agents"])
        item["raw_json"] = json.loads(item["raw_json"])
        logs.append(item)
    return logs


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
        item = dict(row)
        item["activated_agents"] = json.loads(item["activated_agents"])
        item["raw_json"] = json.loads(item["raw_json"])
        memories.append(item)
    return memories


def summarize_memory(user_id: str) -> dict[str, Any]:
    memories = get_recent_memory(user_id, limit=12)
    counters = {
        "repeated_dizziness": 0,
        "repeated_missed_medication": 0,
        "repeated_scam_attempts": 0,
        "repeated_loneliness": 0,
    }

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

    risk_adjustments: list[str] = []
    if counters["repeated_dizziness"] >= 2:
        risk_adjustments.append("Repeated dizziness or fall-related messages increase health priority.")
    if counters["repeated_missed_medication"] >= 2:
        risk_adjustments.append("Repeated missed-medication messages increase medication priority.")
    if counters["repeated_scam_attempts"] >= 2:
        risk_adjustments.append("Repeated scam-related messages increase safety priority.")
    if counters["repeated_loneliness"] >= 2:
        risk_adjustments.append("Repeated loneliness messages increase companion priority.")

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
