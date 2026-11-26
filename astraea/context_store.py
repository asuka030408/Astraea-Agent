"""Session-scoped external records for tool output and the current task state."""

import hashlib
import json
import re
from datetime import datetime, timezone

import astraea.storage as storage


def _terms(text: str) -> set[str]:
    text = text.casefold()
    words = {word for word in re.findall(r"[a-z0-9_./-]{3,}", text)
             if word not in {"the", "and", "for", "with", "file", "tool"}}
    han = re.findall(r"[\u4e00-\u9fff]+", text)
    return words | {part[i:i + 2] for part in han for i in range(len(part) - 1)}


def relevance(query: str, text: str) -> int:
    terms = _terms(query)
    if not terms:
        return 0
    haystack = text.casefold()
    return sum(1 for term in terms if term in haystack)


def short_args(args: dict) -> dict:
    return {str(key): str(value)[:120] for key, value in list(args.items())[:8]}


class ContextStore:
    def __init__(self, session_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("Invalid session ID")
        self.session_id = session_id

    @property
    def directory(self):
        return storage.DATA_DIR / "context" / self.session_id

    def put_result(self, call: dict, result) -> str:
        """Idempotently save one complete tool result, including file reads."""
        call_id = str(call.get("id") or getattr(result, "tool_call_id", ""))
        if not call_id:
            raise ValueError("Tool result has no call ID")
        item_id = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:24]
        path = self.directory / "items" / f"{item_id}.json"
        if not path.is_file():
            args = call.get("args") or {}
            text = str(result.content)
            storage._write_json(path, {
                "id": item_id, "call_id": call_id, "tool": str(call.get("name") or getattr(result, "name", "tool")),
                "args": args, "content": text, "status": getattr(result, "status", "success"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        return item_id

    def get(self, item_id: str) -> dict:
        if item_id == "task_state":
            path = self.directory / "state.json"
        elif re.fullmatch(r"[0-9a-f]{24}", item_id):
            path = self.directory / "items" / f"{item_id}.json"
        else:
            raise ValueError("Invalid context ID")
        if not path.is_file():
            raise FileNotFoundError(item_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def search(self, query: str, limit: int = 5) -> list[dict]:
        ranked = []
        for path in (self.directory / "items").glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            score = relevance(query, f"{item['tool']} {item['args']} {item['content']}")
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (pair[0], pair[1].get("created_at", "")), reverse=True)
        return [{"id": item["id"], "tool": item["tool"], "args": short_args(item["args"]),
                 "preview": item["content"][:240]} for _, item in ranked[:max(1, min(limit, 10))]]

    def list_items(self, limit: int = 200) -> list[dict]:
        paths = sorted((self.directory / "items").glob("*.json"),
                       key=lambda path: path.stat().st_mtime, reverse=True)
        entries = []
        for path in paths[:limit]:
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            entries.append({"kind": "tool_result", "id": item["id"],
                            "text": f"{item['tool']} {short_args(item['args'])}\n{item['content'][:4000]}"})
        return entries

    def save_state(self, *, goal: str, last_response: str, files: list[str]) -> None:
        storage._write_json(self.directory / "state.json", {
            "current_goal": goal[:4000], "last_response": last_response[:2000],
            "recent_files": files[-20:], "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def state(self) -> dict:
        try:
            return self.get("task_state")
        except (OSError, ValueError):
            return {}
