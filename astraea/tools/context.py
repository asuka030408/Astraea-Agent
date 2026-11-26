"""On-demand access to context stored outside model requests."""

import json

from langchain.tools import tool

from astraea.context_store import ContextStore, relevance, short_args
from astraea.storage import load_session


def context_tools(session_id: str) -> list:
    store = ContextStore(session_id)

    @tool
    def search_context(query: str, limit: int = 5) -> str:
        """Find earlier tool results or file contents from this conversation by keyword."""
        if not query.strip():
            return "ERROR: Query is empty."
        return json.dumps(store.search(query, limit), ensure_ascii=False)

    @tool
    def load_context(context_id: str, start_char: int = 0, max_chars: int = 8000,
                     field: str = "content") -> str:
        """Read a stored result's content or full call arguments by ID; task_state reads task state."""
        try:
            item = store.get(context_id)
            if context_id == "task_state":
                return json.dumps(item, ensure_ascii=False)
            if field not in {"content", "args"}:
                return "ERROR: field must be content or args."
            start = max(0, start_char)
            count = max(1, min(max_chars, 12000))
            text = item["content"] if field == "content" else json.dumps(item["args"], ensure_ascii=False)
            return json.dumps({"id": context_id, "tool": item["tool"], "args": short_args(item["args"]),
                               "field": field, "content": text[start:start + count], "start_char": start,
                               "total_chars": len(text)}, ensure_ascii=False)
        except (OSError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool
    def search_history(query: str, limit: int = 5) -> str:
        """Find older user and assistant dialogue in this conversation's full local transcript."""
        if not query.strip():
            return "ERROR: Query is empty."
        try:
            messages = load_session(session_id)["messages"]
        except (OSError, ValueError) as exc:
            return f"ERROR: {exc}"
        ranked = []
        for index, message in enumerate(messages):
            if message.type not in {"human", "ai"} or getattr(message, "tool_calls", None):
                continue
            score = relevance(query, str(message.content))
            if score:
                ranked.append((score, index, message))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return json.dumps([{"index": index, "role": message.type, "preview": str(message.content)[:240]}
                           for _, index, message in ranked[:max(1, min(limit, 10))]], ensure_ascii=False)

    @tool
    def load_history(index: int, start_char: int = 0, max_chars: int = 8000) -> str:
        """Read a saved dialogue message by its search_history index, in bounded chunks."""
        try:
            messages = load_session(session_id)["messages"]
            if index < 0 or index >= len(messages):
                return "ERROR: History index is outside this conversation."
            message = messages[index]
            text = str(message.content)
            start = max(0, start_char)
            return json.dumps({"index": index, "role": message.type,
                               "content": text[start:start + max(1, min(max_chars, 12000))],
                               "start_char": start, "total_chars": len(text)}, ensure_ascii=False)
        except (OSError, ValueError) as exc:
            return f"ERROR: {exc}"

    return [search_context, load_context, search_history, load_history]
