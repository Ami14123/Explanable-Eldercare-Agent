# Security tests for admin auth, public chat access, and sanitized errors.
from __future__ import annotations

import os

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


# Build a TestClient with deterministic mock settings.
def _client_with_admin_token(token: str = "test-admin-token") -> TestClient:
    os.environ["ADMIN_API_TOKEN"] = token
    os.environ["LLM_MODE"] = "mock"
    get_settings.cache_clear()
    return TestClient(app)


# Logs should reject unauthenticated admin access without leaking tracebacks.
def test_logs_requires_admin_token() -> None:
    client = _client_with_admin_token()

    response = client.get("/logs")

    assert response.status_code == 401
    assert "traceback" not in response.text.lower()


# Admin data operations should require the same token protection as logs.
def test_admin_data_operations_require_admin_token() -> None:
    client = _client_with_admin_token()

    download_response = client.post("/download-kaggle-datasets")
    rebuild_response = client.post("/rebuild-vector-store")

    assert download_response.status_code == 401
    assert rebuild_response.status_code == 401
    assert "traceback" not in download_response.text.lower()
    assert "traceback" not in rebuild_response.text.lower()


# Wrong admin tokens should be rejected as forbidden.
def test_logs_rejects_wrong_admin_token() -> None:
    client = _client_with_admin_token()

    response = client.get(
        "/logs",
        headers={"X-Admin-Token": "wrong-token"},
    )

    assert response.status_code == 403
    assert "traceback" not in response.text.lower()


# Correct admin token should allow logs to return a list.
def test_logs_accepts_correct_admin_token() -> None:
    token = "test-admin-token"
    client = _client_with_admin_token(token)

    response = client.get(
        "/logs",
        headers={"X-Admin-Token": token},
    )

    assert response.status_code == 200
    assert isinstance(response.json(), list)


# Public chat must remain available without admin credentials.
def test_chat_does_not_require_admin_token() -> None:
    client = _client_with_admin_token()

    response = client.post(
        "/chat",
        json={
            "message": "I am hungry",
            "user_id": "security_test_user",
            "conversation_id": "phase1",
        },
    )

    assert response.status_code == 200
    assert response.json()["final_message"]


# Public status should avoid exposing local paths or tracebacks.
def test_status_does_not_expose_local_paths() -> None:
    client = _client_with_admin_token()

    response = client.get("/status")

    assert response.status_code == 200
    data = response.json()
    assert "sqlite_path" not in data
    assert "vector_store_path" not in data
    assert "traceback" not in response.text.lower()


# Unexpected chat failures should return sanitized developer_state only.
def test_chat_error_response_is_sanitized(monkeypatch) -> None:
    from app.api import routes_chat

    # Force the route into its generic error handler.
    def fail_workflow(_request):
        raise RuntimeError("synthetic failure for security test")

    monkeypatch.setattr(routes_chat, "run_elderguard_workflow", fail_workflow)
    client = _client_with_admin_token()

    response = client.post(
        "/chat",
        json={
            "message": "Hello",
            "user_id": "security_test_user",
            "conversation_id": "error_case",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["developer_state"] == {"error": "chat_failed"}
    assert "traceback" not in response.text.lower()
    assert "synthetic failure" not in response.text
