# Administrative FastAPI routes for status, logs, data, and RAG rebuilds.
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import require_admin_token
from app.config import get_settings
from app.kaggle_data import download_datasets, get_csv_files
from app.memory import recent_logs
from app.rag import VECTOR_STORE_DIR, build_vector_store
from app.schemas import KaggleDownloadResponse


# Keep admin routes together under one router object.
router = APIRouter()


# Report runtime status without requiring admin credentials.
@router.get("/status")
def status() -> dict[str, str | int | bool]:
    """Return public runtime status without exposing local file paths."""

    # Load settings and cached CSVs to summarize active local capabilities.
    settings = get_settings()
    csv_files = get_csv_files()
    # Return only safe metadata, never secrets or filesystem paths.
    return {
        "name": "ElderGuard AI",
        "backend": "running",
        "llm_mode": settings.llm_mode,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.active_llm_model,
        "database": "SQLite",
        "kaggle_configured": bool(
            settings.kaggle_api_token
            or (settings.kaggle_username and settings.kaggle_key)
        ),
        "kaggle_cached_csv_files": len(csv_files),
        "vector_store_exists": VECTOR_STORE_DIR.exists(),
    }


# Return recent stored chat logs for authorized admin users.
@router.get("/logs")
def logs(
    _: None = Depends(require_admin_token),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    """Return recent logs for authorized administrators only."""

    return recent_logs(limit=limit)


# Download optional Kaggle demo data when an admin explicitly requests it.
@router.post("/download-kaggle-datasets", response_model=KaggleDownloadResponse)
def download_kaggle_datasets(
    _: None = Depends(require_admin_token),
) -> KaggleDownloadResponse:
    """Download optional demo datasets for authorized administrators only."""

    return KaggleDownloadResponse(**download_datasets())


# Rebuild local embeddings so the RAG retriever sees current knowledge files.
@router.post("/rebuild-vector-store")
def rebuild_vector_store(
    _: None = Depends(require_admin_token),
) -> dict[str, object]:
    """Rebuild the local vector store for authorized administrators only."""

    return build_vector_store()
