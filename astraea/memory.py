"""Post-turn memory extraction with Markdown sources and a rebuildable vector index."""

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from openai import OpenAI

from astraea.config import PROJECT_DIR, get_config


MEMORY_DIR = PROJECT_DIR / "data" / "memory"
_LOCK = Lock()
_SOURCE = re.compile(r"^\s*-\s*(.*?)\s*<!-- source:([0-9a-f]{32}|manual) -->\s*$")


def read_profile_markdown(directory: Path = MEMORY_DIR) -> str:
    """Return the profile for prompt injection; an empty template adds no message."""
    path = directory / "profile.md"
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8").strip()
    return content if any(line.lstrip().startswith("- ") for line in content.splitlines()) else ""


@dataclass(frozen=True)
class MemoryItem:
    text: str
    source: str


def _read_items(path: Path) -> list[MemoryItem]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _SOURCE.match(line)
        if match:
            items.append(MemoryItem(match.group(1), match.group(2)))
        elif line.lstrip().startswith("- "):
            text = _clean(line.lstrip()[2:])
            if text:
                items.append(MemoryItem(text, "manual"))
    return items


def _write_items(path: Path, heading: str, items: list[MemoryItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# {heading}\n\n" + "".join(f"- {item.text} <!-- source:{item.source} -->\n" for item in items)
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _clean(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("<!--", "").replace("-->", "").split())[:400]


def _unique_add(items: list[MemoryItem], additions: list, session_id: str) -> list[MemoryItem]:
    seen = {item.text.casefold() for item in items}
    for value in additions:
        text = _clean(value)
        if text and text.casefold() not in seen:
            items.append(MemoryItem(text, session_id))
            seen.add(text.casefold())
    return items


def _extract_json(content: object) -> dict:
    text = str(content)
    start = text.find("{")
    if start < 0:
        raise ValueError("Memory extractor returned no JSON object")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Memory extractor returned invalid JSON")
    return value


def _recent_transcript(messages: list) -> str:
    recent = [item for item in messages if item.type in ("human", "ai") and item.content]
    return "\n".join(f"{item.type}: {str(item.content)[:1500]}" for item in recent[-8:])


class MemoryManager:
    def __init__(self, extraction_model, directory: Path = MEMORY_DIR, embedder=None):
        self.extraction_model = extraction_model
        self.directory = directory
        self.embedding_key = get_config("ASTRAEA_EMBEDDING_API_KEY", "").strip()
        self.embedding_url = get_config("ASTRAEA_EMBEDDING_BASE_URL",
                                        "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.embedding_model = get_config("ASTRAEA_EMBEDDING_MODEL", "text-embedding-v4")
        self.dimensions = int(get_config("ASTRAEA_EMBEDDING_DIMENSIONS", "1024"))
        self.embedder = embedder
        self.initialize()

    def initialize(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for filename, heading in (("profile.md", "用户习惯与长期特征"),
                                  ("events.md", "重要事项（向量索引原文）")):
            path = self.directory / filename
            if not path.exists():
                _write_items(path, heading, [])
        with closing(sqlite3.connect(self.directory / "vectors.sqlite3")) as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL, model TEXT NOT NULL, dimensions INTEGER NOT NULL, embedding TEXT NOT NULL)")
            db.commit()

    def process_turn(self, session_id: str, messages: list) -> dict:
        """Extract after a completed turn. Markdown persists even if indexing fails later."""
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("Invalid session ID")
        with _LOCK:
            profile_path = self.directory / "profile.md"
            events_path = self.directory / "events.md"
            profile = _read_items(profile_path)
            events = _read_items(events_path)
            prompt = (
                "You maintain a user's local long-term memory. Extract only facts explicitly stated by the user "
                "in the most recent user turn, using the recent conversation only to resolve references. "
                "Never store API keys, passwords, tokens, private credentials, guesses, or ordinary chat filler. "
                "profile_add: stable user habits, preferences, hobbies, and ongoing projects or goals. "
                "profile_remove: existing profile statements directly contradicted by the newest user turn; copy exact text. "
                "events_add: specific important tasks, decisions, milestones, or events useful later but not stable traits. "
                "Avoid duplicates of existing memory. Return ONLY a JSON object with three arrays of short Chinese "
                "sentences: profile_add, profile_remove, events_add. Empty arrays are valid.\n\n"
                f"Existing profile:\n{chr(10).join(item.text for item in profile)[-8000:]}\n\n"
                f"Existing events:\n{chr(10).join(item.text for item in events)[-8000:]}\n\n"
                f"Recent conversation:\n{_recent_transcript(messages)}"
            )
            result = _extract_json(self.extraction_model.invoke(prompt).content)
            additions = result.get("profile_add", [])
            removals = result.get("profile_remove", [])
            event_additions = result.get("events_add", [])
            if not all(isinstance(value, list) for value in (additions, removals, event_additions)):
                raise ValueError("Memory extractor returned invalid arrays")
            removed = {_clean(value).casefold() for value in removals}
            profile = [item for item in profile if item.text.casefold() not in removed]
            before_profile, before_events = len(profile), len(events)
            _unique_add(profile, additions, session_id)
            _unique_add(events, event_additions, session_id)
            _write_items(profile_path, "用户习惯与长期特征", profile)
            _write_items(events_path, "重要事项（向量索引原文）", events)
            self.sync_index(events)
            return {"profile_added": len(profile) - before_profile,
                    "profile_removed": len(removed), "events_added": len(events) - before_events}

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is not None:
            return self.embedder(texts)
        if not self.embedding_key:
            raise ValueError("ASTRAEA_EMBEDDING_API_KEY is not configured")
        client = OpenAI(api_key=self.embedding_key, base_url=self.embedding_url, timeout=30, max_retries=1)
        response = client.embeddings.create(model=self.embedding_model, input=texts, dimensions=self.dimensions)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def sync_index(self, events: list[MemoryItem] | None = None) -> None:
        """Rebuild missing or stale vectors from events.md; it remains the source of truth."""
        events = events if events is not None else _read_items(self.directory / "events.md")
        self.initialize()
        with closing(sqlite3.connect(self.directory / "vectors.sqlite3")) as db, db:
            desired = {hashlib.sha256(item.text.encode("utf-8")).hexdigest(): item for item in events}
            existing = {row[0]: (row[1], row[2]) for row in db.execute("SELECT id, model, dimensions FROM vectors")}
            stale = [item for key, item in desired.items()
                     if key not in existing or existing[key] != (self.embedding_model, self.dimensions)]
            for offset in range(0, len(stale), 10):
                batch = stale[offset:offset + 10]
                vectors = self._embed([item.text for item in batch])
                if len(vectors) != len(batch) or any(len(vector) != self.dimensions for vector in vectors):
                    raise ValueError("Embedding model returned unexpected dimensions")
                db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?, ?)",
                               [(hashlib.sha256(item.text.encode("utf-8")).hexdigest(), item.text, item.source,
                                 self.embedding_model, self.dimensions, json.dumps(vector))
                                for item, vector in zip(batch, vectors)])
            db.executemany("DELETE FROM vectors WHERE id = ?", [(key,) for key in existing if key not in desired])

    def search_events(self, query: str, limit: int = 5) -> list[dict]:
        """Exact cosine search over the small local vector store."""
        with _LOCK:
            self.sync_index()
            with closing(sqlite3.connect(self.directory / "vectors.sqlite3")) as db:
                rows = db.execute("SELECT text, source, embedding FROM vectors WHERE model = ? AND dimensions = ?",
                                  (self.embedding_model, self.dimensions)).fetchall()
        if not rows or not query.strip() or limit <= 0:
            return []
        vector = self._embed([query])[0]
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return []
        ranked = []
        for text, source, encoded in rows:
            candidate = json.loads(encoded)
            denominator = norm * math.sqrt(sum(value * value for value in candidate))
            score = sum(left * right for left, right in zip(vector, candidate)) / denominator if denominator else 0
            ranked.append({"text": text, "source_session": source, "score": score})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:max(0, limit)]
