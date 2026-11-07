"""Semantic recall of important facts from earlier conversations."""

import json

from langchain_core.tools import tool

from astraea.memory import MemoryManager


@tool
def search_memory(query: str, top_k: int = 5) -> str:
    """Search important past events, tasks, and decisions by meaning when the user asks about earlier information. The user profile is already supplied separately; do not call this for ordinary questions."""
    if not query.strip():
        return "[]"
    results = MemoryManager(None).search_events(query, limit=min(max(top_k, 1), 10))
    return json.dumps(results, ensure_ascii=False)
