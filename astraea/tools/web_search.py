"""Keyless web search; page reading remains the Fetch MCP tool's job."""

import json

from ddgs import DDGS
from langchain_core.tools import tool


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the public web for current information and return titles, URLs, and short snippets. Use fetch_fetch to read a result before relying on its details."""
    if not query.strip():
        return "[]"
    results = DDGS(timeout=15).text(query.strip(), max_results=min(max(max_results, 1), 10))
    return json.dumps([{"title": item.get("title", ""), "url": item.get("href", ""),
                        "snippet": item.get("body", "")} for item in results], ensure_ascii=False)
