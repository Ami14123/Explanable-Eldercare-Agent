import os
import shutil
from pathlib import Path

import pandas as pd

from app.config import get_settings


DEFAULT_DATASETS = [
    "uciml/sms-spam-collection-dataset",
    "ziya07/elderly-fall-detection-iot-dataset",
    "programmer3/iot-based-chronic-medication-adherence-dataset",
]


def _dataset_dir(dataset: str) -> Path:
    settings = get_settings()
    safe_name = dataset.replace("/", "__")
    path = settings.raw_data_dir / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _has_cached_files(path: Path) -> bool:
    return path.exists() and any(file.is_file() for file in path.rglob("*"))


def _configure_kaggle_credentials() -> tuple[bool, str]:
    settings = get_settings()
    username = settings.kaggle_username.strip()
    key = settings.kaggle_key.strip()

    if not username or not key:
        return False, "KAGGLE_USERNAME and KAGGLE_KEY are not set."

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    return True, ""


def _download_with_kagglehub(datasets: list[str]) -> dict[str, list[str]]:
    settings = get_settings()
    token = settings.kaggle_api_token.strip()
    downloaded: list[str] = []
    cached: list[str] = []
    errors: list[str] = []

    if not token:
        return {
            "downloaded": downloaded,
            "cached": cached,
            "errors": ["KAGGLE_API_TOKEN is not set."],
            "csv_files": [],
        }

    os.environ["KAGGLE_API_TOKEN"] = token
    token_dir = Path("data/kaggle")
    token_dir.mkdir(parents=True, exist_ok=True)
    (token_dir / "access_token").write_text(token, encoding="utf-8")
    os.environ["KAGGLE_CONFIG_DIR"] = str(token_dir.resolve())

    try:
        import kagglehub
    except Exception as exc:
        return {
            "downloaded": downloaded,
            "cached": cached,
            "errors": [f"kagglehub is not installed yet. Run pip install -r requirements.txt. Details: {exc}"],
            "csv_files": [],
        }

    for dataset in datasets:
        destination = _dataset_dir(dataset)
        if _has_cached_files(destination):
            cached.append(dataset)
            continue

        try:
            source_path = Path(kagglehub.dataset_download(dataset))
            for item in source_path.rglob("*"):
                if item.is_file():
                    relative = item.relative_to(source_path)
                    output_path = destination / relative
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, output_path)
            downloaded.append(dataset)
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")

    return {
        "downloaded": downloaded,
        "cached": cached,
        "errors": errors,
        "csv_files": [str(path) for path in get_csv_files()],
    }


def download_datasets(datasets: list[str] | None = None) -> dict[str, list[str]]:
    selected = datasets or DEFAULT_DATASETS
    downloaded: list[str] = []
    cached: list[str] = []
    errors: list[str] = []

    if get_settings().kaggle_api_token.strip():
        return _download_with_kagglehub(selected)

    credentials_ok, credential_error = _configure_kaggle_credentials()
    if not credentials_ok:
        return {
            "downloaded": downloaded,
            "cached": cached,
            "errors": [credential_error],
            "csv_files": [],
        }

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        return {
            "downloaded": downloaded,
            "cached": cached,
            "errors": [f"Kaggle authentication failed: {exc}"],
            "csv_files": [],
        }

    for dataset in selected:
        destination = _dataset_dir(dataset)
        if _has_cached_files(destination):
            cached.append(dataset)
            continue

        try:
            api.dataset_download_files(
                dataset,
                path=str(destination),
                unzip=True,
                quiet=False,
            )
            downloaded.append(dataset)
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")

    csv_files = [str(path) for path in get_csv_files()]
    return {
        "downloaded": downloaded,
        "cached": cached,
        "errors": errors,
        "csv_files": csv_files,
    }


def get_csv_files() -> list[Path]:
    settings = get_settings()
    if not settings.raw_data_dir.exists():
        return []
    return sorted(settings.raw_data_dir.rglob("*.csv"))


def load_csv_files() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in get_csv_files():
        try:
            frames[str(path)] = pd.read_csv(path)
        except Exception:
            continue
    return frames
