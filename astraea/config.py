"""Read and validate local model configuration.

Values saved from the web settings dialog live in data/settings.json and take
precedence over .env, so a fresh copy of the client works without editing any
project file. Only whitelisted keys can be written.
"""

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
USER_SETTINGS_FILE = PROJECT_DIR / "data" / "settings.json"

SETTING_KEYS = (
    "ASTRAEA_API_KEY",
    "ASTRAEA_BASE_URL",
    "ASTRAEA_MODEL",
    "ASTRAEA_COMPACTOR_MODEL",
    "ASTRAEA_CONTEXT_WINDOW",
    "ASTRAEA_WORKSPACE",
    "ASTRAEA_EMBEDDING_API_KEY",
    "ASTRAEA_EMBEDDING_BASE_URL",
    "ASTRAEA_EMBEDDING_MODEL",
    "ASTRAEA_EMBEDDING_DIMENSIONS",
)
SECRET_KEYS = frozenset({"ASTRAEA_API_KEY", "ASTRAEA_EMBEDDING_API_KEY"})
DEFAULTS = {
    "ASTRAEA_API_KEY": "",
    "ASTRAEA_BASE_URL": "https://api.deepseek.com",
    "ASTRAEA_MODEL": "deepseek-chat",
    "ASTRAEA_COMPACTOR_MODEL": "",
    "ASTRAEA_CONTEXT_WINDOW": "64000",
    "ASTRAEA_WORKSPACE": str(PROJECT_DIR / "workspace"),
    "ASTRAEA_EMBEDDING_API_KEY": "",
    "ASTRAEA_EMBEDDING_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "ASTRAEA_EMBEDDING_MODEL": "text-embedding-v4",
    "ASTRAEA_EMBEDDING_DIMENSIONS": "1024",
}
_FILE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    context_window: int
    compactor_model: str
    workspace: Path


def _read_file() -> dict:
    with _FILE_LOCK:
        try:
            raw = json.loads(USER_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    return {key: str(value) for key, value in raw.items()
            if key in SETTING_KEYS and isinstance(value, (str, int, float)) and str(value).strip()}


def user_settings() -> dict:
    """Values explicitly saved by the user in the web UI."""
    return _read_file()


def save_user_settings(values: dict) -> dict:
    """Merge overrides. An empty string keeps the current value; None removes it."""
    with _FILE_LOCK:
        try:
            current = json.loads(USER_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            current = {}
        if not isinstance(current, dict):
            current = {}
        cleaned = {key: str(value) for key, value in current.items()
                   if key in SETTING_KEYS and isinstance(value, (str, int, float)) and str(value).strip()}
        for key, value in (values or {}).items():
            if key not in SETTING_KEYS:
                continue
            if value is None:
                cleaned.pop(key, None)
            elif str(value).strip():
                cleaned[key] = str(value).strip()
        USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = USER_SETTINGS_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(USER_SETTINGS_FILE)
        return dict(cleaned)


def get_config(key: str, default: str = "") -> str:
    """UI overrides win over .env / process environment."""
    override = _read_file().get(key)
    if override:
        return override
    load_dotenv(PROJECT_DIR / ".env")
    return os.getenv(key, default)


def effective_settings() -> dict:
    return {key: get_config(key, DEFAULTS.get(key, "")) for key in SETTING_KEYS}


def mask_secret(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    return "****" if len(text) <= 8 else f"{text[:3]}****{text[-4:]}"


def load_settings() -> Settings:
    api_key = get_config("ASTRAEA_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key":
        raise ValueError("请先在网页「设置」里填写对话模型的 API Key（或修改 .env 的 ASTRAEA_API_KEY）。")

    context_window = int(get_config("ASTRAEA_CONTEXT_WINDOW", "64000"))
    if context_window < 1000:
        raise ValueError("ASTRAEA_CONTEXT_WINDOW must be at least 1000.")

    workspace = Path(get_config("ASTRAEA_WORKSPACE", str(PROJECT_DIR / "workspace"))).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    model = get_config("ASTRAEA_MODEL", "deepseek-chat")

    return Settings(
        api_key=api_key,
        base_url=get_config("ASTRAEA_BASE_URL", "https://api.deepseek.com"),
        model=model,
        context_window=context_window,
        compactor_model=get_config("ASTRAEA_COMPACTOR_MODEL", "") or model,
        workspace=workspace,
    )
