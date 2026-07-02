from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


@dataclass
class Settings:
    llm_mode: str = "mock"
    openrouter_api_key: str = ""
    openrouter_model: str = ""
    kaggle_api_token: str = ""
    kaggle_username: str = ""
    kaggle_key: str = ""
    sqlite_path: str = "data/elderguard.sqlite3"
    kaggle_data_dir: str = "data/raw"

    @property
    def mock_mode(self) -> bool:
        return self.llm_mode == "mock"

    @property
    def llm_provider(self) -> str:
        return "none" if self.mock_mode else "openrouter"

    @property
    def active_llm_model(self) -> str:
        return "none" if self.mock_mode else self.openrouter_model

    @property
    def sqlite_file(self) -> Path:
        return Path(self.sqlite_path)

    @property
    def raw_data_dir(self) -> Path:
        return Path(self.kaggle_data_dir)

    def validate(self) -> None:
        if self.llm_mode not in {"mock", "live"}:
            raise RuntimeError("LLM_MODE must be either 'mock' or 'live'.")
        if self.llm_mode == "live":
            if not self.openrouter_api_key.strip() or self.openrouter_api_key.strip() == "your_key_here":
                raise RuntimeError("LLM_MODE=live requires OPENROUTER_API_KEY.")
            if not self.openrouter_model.strip():
                raise RuntimeError("LLM_MODE=live requires OPENROUTER_MODEL.")


@lru_cache
def get_settings() -> Settings:
    _load_dotenv_if_available()
    settings = Settings(
        llm_mode=os.getenv("LLM_MODE", "mock").strip().lower(),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", ""),
        kaggle_api_token=os.getenv("KAGGLE_API_TOKEN", ""),
        kaggle_username=os.getenv("KAGGLE_USERNAME", ""),
        kaggle_key=os.getenv("KAGGLE_KEY", ""),
        sqlite_path=os.getenv("SQLITE_PATH", "data/elderguard.sqlite3"),
        kaggle_data_dir=os.getenv("KAGGLE_DATA_DIR", "data/raw"),
    )
    settings.validate()
    return settings
