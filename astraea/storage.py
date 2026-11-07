"""Local storage for complete sessions and archived context."""

import json
from pathlib import Path

from langchain_core.messages import message_to_dict, messages_from_dict

from astraea.config import PROJECT_DIR


DATA_DIR = PROJECT_DIR / "data"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_session(session_id: str, messages: list, summary: str, tier: int, active: list | None = None) -> None:
    _write_json(
        DATA_DIR / "sessions" / f"{session_id}.json",
        {
            "session_id": session_id,
            "tier": tier,
            "summary": summary,
            "messages": [message_to_dict(message) for message in messages],
            "active": [message_to_dict(message) for message in (messages if active is None else active)],
        },
    )


def load_session(session_id: str) -> dict:
    if len(session_id) != 32 or any(char not in "0123456789abcdef" for char in session_id):
        raise ValueError("Invalid session ID")
    path = DATA_DIR / "sessions" / f"{session_id}.json"
    if not path.is_file():
        raise FileNotFoundError(session_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("session_id") != session_id:
        raise ValueError("Session ID mismatch")
    data["active"] = messages_from_dict(data.get("active", data["messages"]))
    data["messages"] = messages_from_dict(data["messages"])
    return data


def list_sessions() -> list[dict]:
    entries = []
    for path in (DATA_DIR / "sessions").glob("*.json"):
        try:
            data = load_session(path.stem)
            title = next((str(item.content).strip() for item in data["messages"]
                          if item.type == "human" and str(item.content).strip()), "")
            if not title:
                continue
            entries.append({"session_id": path.stem, "title": title[:80], "updated_at": path.stat().st_mtime})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return sorted(entries, key=lambda item: item["updated_at"], reverse=True)


def archive_fragment(session_id: str, kind: str, text: str) -> None:
    """Keep removed text outside the active context for later retrieval."""
    path = DATA_DIR / "archives" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"kind": kind, "text": text}, ensure_ascii=False) + "\n")


def list_archive_fragments(session_id: str) -> list[dict]:
    """Read one session's archived pieces in their original order."""
    load_session(session_id)  # Validate the ID and ensure the session exists.
    path = DATA_DIR / "archives" / f"{session_id}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieve_fragments(session_id: str, query: str, limit: int = 2) -> list[str]:
    """Small local relevance search; no model or file-reading agent involved."""
    path = DATA_DIR / "archives" / f"{session_id}.jsonl"
    if not path.exists():
        return []
    terms = {query[i:i + 2] for i in range(len(query) - 1) if not query[i:i + 2].isspace()}
    ranked = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        score = sum(term in item["text"] for term in terms)
        if score:
            ranked.append((score, item["text"]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [text[:1200] for _, text in ranked[:limit]]
