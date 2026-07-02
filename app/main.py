import traceback
from pathlib import Path

from fastapi import FastAPI, Query

from app.config import get_settings
from app.graph import run_elderguard_workflow
from app.kaggle_data import download_datasets, get_csv_files
from app.memory import init_db, recent_logs
from app.rag import VECTOR_STORE_DIR, build_vector_store
from app.schemas import ChatRequest, ChatResponse, KaggleDownloadResponse


app = FastAPI(
    title="ElderGuard AI",
    description="Autonomous multi-agent eldercare backend for Dify.",
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "ElderGuard AI", "status": "running"}


@app.get("/status")
def status() -> dict[str, str | int | bool]:
    settings = get_settings()
    csv_files = get_csv_files()
    return {
        "name": "ElderGuard AI",
        "backend": "running",
        "llm_mode": settings.llm_mode,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.active_llm_model,
        "database": "SQLite",
        "sqlite_path": str(settings.sqlite_file),
        "kaggle_configured": bool(
            settings.kaggle_api_token or (settings.kaggle_username and settings.kaggle_key)
        ),
        "kaggle_cached_csv_files": len(csv_files),
        "vector_store_exists": VECTOR_STORE_DIR.exists(),
        "vector_store_path": str(VECTOR_STORE_DIR),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        response = run_elderguard_workflow(request)
        # Ensure response is valid
        if not response.final_message:
            response.final_message = "I'm here to help. Could you tell me more about what you need?"
        return response
    except Exception as e:
        import traceback
        Path("data").mkdir(exist_ok=True)
        Path("data/last_chat_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        # Return a graceful error response instead of crashing
        return ChatResponse(
            final_message="I encountered an issue. Please try again.",
            active_mode="Error",
            final_message_source="error_handler",
            current_topic="",
            is_follow_up=False,
            known_information=[],
            follow_up_questions=[],
            specificity_score=0,
            missing_information=[],
            hypotheses=[],
            need_more_information=False,
            reasoning_summary="Chat endpoint error",
            xai_simple="An error occurred",
            caregiver_summary="",
            activated_agents=[],
            overall_risk="medium",
            main_concern="System error",
            advice="Try again",
            caregiver_alert="",
            next_step="Retry",
            memory_id=0,
            human_confirmation_required=False,
            confirmation_reason="",
            routing_explanation="",
            graph_reasoning={},
            detected_signals=[],
            risk_scores={},
            memory_summary={},
            situation={},
            care_plan={},
            developer_state={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.get("/logs")
def logs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return recent_logs(limit=limit)


@app.post("/download-kaggle-datasets", response_model=KaggleDownloadResponse)
def download_kaggle_datasets() -> KaggleDownloadResponse:
    return KaggleDownloadResponse(**download_datasets())


@app.post("/rebuild-vector-store")
def rebuild_vector_store() -> dict[str, object]:
    return build_vector_store()
