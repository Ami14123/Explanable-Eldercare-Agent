# Central settings loader for mock/live modes and local paths.
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


# Load a local .env file when python-dotenv is installed.
def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        # Keep startup working even when dotenv is not installed.
        return
    load_dotenv()


# Store environment-backed settings in one typed object.
@dataclass
class Settings:
    # Control whether the app uses mock responses or a live LLM.
    llm_mode: str = "mock"
    # OpenRouter settings are needed only in live mode.
    openrouter_api_key: str = ""
    openrouter_model: str = ""
    # Admin token protects local admin endpoints when configured.
    admin_api_token: str = ""
    # Kaggle settings support optional dataset download helpers.
    kaggle_api_token: str = ""
    kaggle_username: str = ""
    kaggle_key: str = ""
    # Local storage paths keep memory and raw data predictable.
    sqlite_path: str = "data/elderguard.sqlite3"
    kaggle_data_dir: str = "data/raw"

    # Make mock checks readable at call sites.
    @property
    def mock_mode(self) -> bool:
        return self.llm_mode == "mock"

    # Report the provider name for UI and API diagnostics.
    @property
    def llm_provider(self) -> str:
        return "none" if self.mock_mode else "openrouter"

    # Report the active model without exposing a fake value in mock mode.
    @property
    def active_llm_model(self) -> str:
        return "none" if self.mock_mode else self.openrouter_model

    # Convert the database path only when callers need a Path object.
    @property
    def sqlite_file(self) -> Path:
        return Path(self.sqlite_path)

    # Convert the raw data directory only when callers need filesystem access.
    @property
    def raw_data_dir(self) -> Path:
        return Path(self.kaggle_data_dir)

    # Validate settings early so bad live configuration fails clearly.
    def validate(self) -> None:
        if self.llm_mode not in {"mock", "live"}:
            raise RuntimeError("LLM_MODE must be either 'mock' or 'live'.")
        # Live mode needs both an API key and model before any network call.
        if self.llm_mode == "live":
            if not self.openrouter_api_key.strip() or self.openrouter_api_key.strip() == "your_key_here":
                raise RuntimeError("LLM_MODE=live requires OPENROUTER_API_KEY.")
            if not self.openrouter_model.strip():
                raise RuntimeError("LLM_MODE=live requires OPENROUTER_MODEL.")


# Cache settings so every request uses one consistent configuration object.
@lru_cache
def get_settings() -> Settings:
    # Read .env values before checking os.environ.
    _load_dotenv_if_available()
    # Build the settings object from environment variables with safe defaults.
    settings = Settings(
        llm_mode=os.getenv("LLM_MODE", "mock").strip().lower(),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", ""),
        admin_api_token=os.getenv("ADMIN_API_TOKEN", ""),
        kaggle_api_token=os.getenv("KAGGLE_API_TOKEN", ""),
        kaggle_username=os.getenv("KAGGLE_USERNAME", ""),
        kaggle_key=os.getenv("KAGGLE_KEY", ""),
        sqlite_path=os.getenv("SQLITE_PATH", "data/elderguard.sqlite3"),
        kaggle_data_dir=os.getenv("KAGGLE_DATA_DIR", "data/raw"),
    )
    # Fail fast if the selected mode is incomplete or unsupported.
    settings.validate()
    return settings
