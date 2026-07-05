# FastAPI application entry point for the ElderGuard backend.
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_admin import router as admin_router
from app.api.routes_chat import router as chat_router
from app.memory import init_db


# Create one app object so ASGI servers can import it directly.
app = FastAPI(
    title="ElderGuard AI",
    description="Explainable multi-agent eldercare backend for the Streamlit prototype.",
    version="1.0.0",
)


# Prepare the SQLite memory tables before serving requests.
@app.on_event("startup")
def startup() -> None:
    init_db()


# Expose a tiny health endpoint for manual checks.
@app.get("/")
def root() -> dict[str, str]:
    return {"name": "ElderGuard AI", "status": "running"}


# Register chat routes before admin routes so the public API is available.
app.include_router(chat_router)
# Vietnamese note: FastAPI dang ky route chinh truoc khi request di vao LangGraph.
app.include_router(admin_router)
