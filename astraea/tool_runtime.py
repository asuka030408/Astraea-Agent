"""Per-session tool routing, schema reduction, safe caching and read-only plans."""

import asyncio
import copy
import json
import re
import time
from collections import OrderedDict
from typing import Any
from typing_extensions import TypedDict
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from astraea.context_store import relevance, short_args
from astraea.tools import TOOLS
from astraea.tools.files import safe_path, workspace


READONLY = {name: item for item in TOOLS
            if (name := item.name) in {"read_file", "read_file_lines", "list_files", "file_info", "search_files"}}
CACHEABLE = {"read_file", "read_file_lines", "list_files", "file_info"}
ALWAYS = {"search_tools", "load_tools"}
MCP_PREFIXES = ("fetch_", "filesystem_", "git_", "time_")
FILE_PATTERN = re.compile(r"(?<![A-Za-z0-9_./\\-])([A-Za-z0-9_./\\-]+\.(?:txt|md|py|json|ya?ml|toml|csv|html|js|css))(?![A-Za-z0-9_./\\-])", re.I)


class ReadStep(TypedDict):
    tool: str
    args: dict[str, Any]


def compact_schema(item) -> dict:
    """Shorten descriptions without changing names, argument types or required fields."""
    schema = copy.deepcopy(convert_to_openai_tool(item))
    function = schema.get("function", {})
    description = " ".join(str(function.get("description", "")).split())
    function["description"] = description[:110]

    def shorten(node):
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("examples", None)
            if isinstance(node.get("description"), str):
                node["description"] = " ".join(node["description"].split())[:90]
            for value in node.values():
                shorten(value)
        elif isinstance(node, list):
            for value in node:
                shorten(value)

    shorten(function.get("parameters", {}))
    return schema


class ToolRuntime:
    def __init__(self, context):
        self.context = context
        self.catalog = {}
        self.pinned = set()
        self.cache = OrderedDict()
        self.locks = {}
        self.mcp_lock = None
        self.loop = None
        self.current_query = ""
        self.workspace_root = workspace()
        self.prefetched_keys = set()
        self.metrics = {"cache_hits": 0, "cache_misses": 0, "prefetches": 0,
                        "prefetch_hits": 0, "compressed_results": 0,
                        "model_tool_definitions": [], "model_schema_tokens": []}

    def _loop_locks(self):
        loop = asyncio.get_running_loop()
        if self.loop is not loop:
            self.loop = loop
            self.locks = {}
            self.mcp_lock = asyncio.Lock()

    def tools(self) -> list:
        runtime = self

        @tool
        def search_tools(query: str, limit: int = 8) -> str:
            """Find available tool names and short descriptions when the needed tool is not visible."""
            if not query.strip():
                return "ERROR: Query is empty."
            ranked = []
            for name, item in runtime.catalog.items():
                description = str(getattr(item, "description", ""))
                score = relevance(query, name.replace("_", " ") + " " + description)
                if score:
                    ranked.append((score, name, description))
            ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
            return json.dumps([{"name": name, "description": description[:130]}
                               for _, name, description in ranked[:max(1, min(limit, 12))]], ensure_ascii=False)

        @tool
        def load_tools(names: list[str]) -> str:
            """Expose named tools on the next model call after search_tools finds them."""
            selected = list(dict.fromkeys(names))[:12]
            loaded = [name for name in selected if name in runtime.catalog]
            runtime.pinned.update(loaded)
            return json.dumps({"loaded": loaded, "unknown": [name for name in selected if name not in runtime.catalog]},
                              ensure_ascii=False)

        @tool
        async def run_readonly_plan(steps: list[ReadStep], parallel: bool = True) -> str:
            """Run independent read_file, read_file_lines, list_files, file_info or search_files steps; return ordered previews and full-result IDs."""
            if not steps or len(steps) > 8:
                return "ERROR: Provide one to eight steps."
            parsed = []
            for step in steps:
                if not isinstance(step, dict) or step.get("tool") not in READONLY or not isinstance(step.get("args"), dict):
                    return "ERROR: Each step needs a read-only tool name and an args object."
                parsed.append((step["tool"], step["args"]))

            semaphore = asyncio.Semaphore(4 if parallel else 1)

            async def one(index, name, args):
                async with semaphore:
                    try:
                        content = await runtime.cached_text(name, args,
                            lambda: READONLY[name].ainvoke(args))
                    except Exception as exc:
                        content = f"ERROR: {type(exc).__name__}: {exc}"
                    call = {"id": f"plan-{uuid4().hex}", "name": name, "args": args}
                    item_id = runtime.context.store.put_result(call,
                        ToolMessage(content=content, tool_call_id=call["id"], name=name))
                    return {"step": index, "tool": name, "args": short_args(args),
                            "content": content[:300], "total_chars": len(content), "context_id": item_id}

            results = await asyncio.gather(*(one(i, name, args) for i, (name, args) in enumerate(parsed)))
            return json.dumps(results, ensure_ascii=False)

        return [search_tools, load_tools, run_readonly_plan]

    def bind_catalog(self, tools: list) -> None:
        self.catalog = {item.name: item for item in tools}
        self.workspace_root = workspace()

    def select_schemas(self, messages: list, tools: list) -> list[dict]:
        """Expose a small relevant set; discovery tools can activate any registered tool."""
        available = {item.name: item for item in tools}
        current = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        self.current_query = current[:1000]
        query = current.casefold()
        names = set(ALWAYS) | self.pinned
        paths = FILE_PATTERN.findall(current)
        file_intent = bool(paths or any(word in query for word in
                           ("文件", "目录", "代码", "文档", "file", "folder", "directory", "read", "write", "path")))
        web_intent = any(word in query for word in ("网上", "搜索网页", "网站", "网页", "最新", "web", "online", "url", "http"))
        memory_intent = any(word in query for word in ("之前", "历史", "记忆", "归档", "上次", "以前", "memory", "remember", "history", "previous"))
        time_intent = any(word in query for word in ("时间", "日期", "几点", "时区", "time", "date", "timezone"))
        git_intent = any(word in query for word in ("git", "提交记录", "分支", "仓库", "diff", "commit", "branch"))
        test_intent = any(word in query for word in ("测试", "运行", "pytest", "test", "run command"))
        if not any((file_intent, web_intent, memory_intent, time_intent, git_intent, test_intent)):
            names.update({"read_file", "search_web", "search_memory"})
        if file_intent:
            names.add("read_file")
            if any(word in query for word in ("列", "目录", "多少", "几个", "数量", "list", "directory", "folder", "count")):
                names.add("list_files")
            if any(word in query for word in ("搜索", "查找", "find", "search")):
                names.add("search_files")
            if any(word in query for word in ("第几行", "行号", "lines", "line ")):
                names.add("read_file_lines")
            if any(word in query for word in ("大小", "元数据", "信息", "size", "metadata")):
                names.add("file_info")
            if len(paths) >= 2 or any(word in query for word in ("多个文件", "批量", "several files", "multiple files")):
                names.add("run_readonly_plan")
            if any(word in query for word in ("写", "创建", "write", "create")):
                names.add("write_file")
            if any(word in query for word in ("修改", "替换", "edit", "replace")):
                names.update({"replace_in_file", "write_file"})
            if any(word in query for word in ("追加", "append")):
                names.add("append_file")
            if any(word in query for word in ("复制", "copy")):
                names.add("copy_file")
            if any(word in query for word in ("移动", "重命名", "move", "rename")):
                names.add("move_file")
            if any(word in query for word in ("删除", "移除", "delete", "remove")):
                names.update({"delete_file", "file_info"})
        if web_intent:
            names.update({"search_web", "fetch_fetch"})
        if memory_intent:
            names.update({"search_memory", "search_context", "load_context", "search_history", "load_history"})
        if time_intent:
            names.update(name for name in available if name.startswith("time_"))
        if git_intent:
            names.update({"run_command", "git_git_status", "git_git_diff", "git_git_log", "git_git_branch"})
            if any(word in query for word in ("提交", "commit", "合并", "merge", "切换分支", "checkout")):
                names.update({"git_git_add", "git_git_commit", "git_git_merge", "git_git_checkout"})
        if test_intent:
            names.add("run_command")

        # Continue exposing only tools used after the latest user message.
        current_index = next((i for i in range(len(messages) - 1, -1, -1)
                              if isinstance(messages[i], HumanMessage)), len(messages))
        latest_calls = next((m.tool_calls for m in reversed(messages[current_index + 1:])
                             if isinstance(m, AIMessage) and m.tool_calls), [])
        names.update(call.get("name") for call in latest_calls)
        selected = [compact_schema(item) for item in tools if item.name in names]
        self.metrics["model_tool_definitions"].append(len(selected))
        from astraea.context import _measure_tool_schemas
        self.metrics["model_schema_tokens"].append(_measure_tool_schemas(selected))
        for key in ("model_tool_definitions", "model_schema_tokens"):
            if len(self.metrics[key]) > 100:
                self.metrics[key].pop(0)
        return selected

    def _cache_key(self, name: str, args: dict) -> str | None:
        if name not in CACHEABLE:
            return None
        try:
            root = self.workspace_root
            relative = args.get("directory", ".") if name == "list_files" else args["path"]
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or (name != "list_files" and target == root):
                return None
            stat = target.stat()
        except (OSError, KeyError, TypeError, ValueError):
            return None
        payload = {"name": name, "args": args, "workspace": str(root),
                   "target": str(target), "mtime_ns": stat.st_mtime_ns,
                   "ctime_ns": stat.st_ctime_ns, "size": stat.st_size}
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    async def cached_text(self, name: str, args: dict, runner, *, for_prefetch: bool = False) -> str:
        self._loop_locks()
        key = self._cache_key(name, args)
        if key is None:
            result = await runner()
            return str(result.content) if isinstance(result, ToolMessage) else str(result)
        cached = self.cache.get(key)
        if cached and time.monotonic() - cached[0] < 120:
            self.metrics["cache_hits"] += 1
            if not for_prefetch and key in self.prefetched_keys:
                self.metrics["prefetch_hits"] += 1
                self.prefetched_keys.discard(key)
            self.cache.move_to_end(key)
            return cached[1]
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < 120:
                self.metrics["cache_hits"] += 1
                if not for_prefetch and key in self.prefetched_keys:
                    self.metrics["prefetch_hits"] += 1
                    self.prefetched_keys.discard(key)
                self.cache.move_to_end(key)
                return cached[1]
            self.metrics["cache_misses"] += 1
            result = await runner()
            content = str(result.content) if isinstance(result, ToolMessage) else str(result)
            if not content.startswith("ERROR:") and not (isinstance(result, ToolMessage) and result.status == "error"):
                self.cache[key] = (time.monotonic(), content)
                self.cache.move_to_end(key)
                while len(self.cache) > 128:
                    self.cache.popitem(last=False)
            return content

    def prefetch(self, user_text: str) -> list[asyncio.Task]:
        """Speculatively read at most two explicit workspace files, without writes."""
        if not any(word in user_text.casefold() for word in ("读", "看", "打开", "检查", "read", "open", "inspect")):
            return []
        paths = list(dict.fromkeys(FILE_PATTERN.findall(user_text)))[:2]
        tasks = []
        for path in paths:
            try:
                if not safe_path(path).is_file():
                    continue
            except (OSError, ValueError):
                continue
            self.metrics["prefetches"] += 1
            self.prefetched_keys.add(self._cache_key("read_file", {"path": path}))
            tasks.append(asyncio.create_task(self.cached_text("read_file", {"path": path},
                                                lambda path=path: READONLY["read_file"].ainvoke({"path": path}),
                                                for_prefetch=True)))
        return tasks

    def compress_result(self, result: ToolMessage, item_id: str | None) -> ToolMessage:
        text = str(result.content)
        if not item_id or len(text) <= 2500:
            return result
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            preview = []
            for entry in parsed[:8]:
                if isinstance(entry, dict):
                    preview.append({key: str(entry[key])[:100] for key in
                                    ("step", "tool", "context_id", "title", "url", "content") if key in entry})
                else:
                    preview.append(str(entry)[:100])
            reduced = json.dumps({"truncated": True, "total_items": len(parsed),
                                  "preview": preview, "full_context_id": item_id}, ensure_ascii=False)
            self.metrics["compressed_results"] += 1
            return result.model_copy(update={"content": reduced})
        lines = text.splitlines()
        matches = [line[:180] for line in lines if relevance(self.current_query, line) > 0][:5]
        middle = "\n".join(matches)[:600]
        reduced = (text[:1100] + "\n" + ("[Relevant lines]\n" + middle + "\n" if middle else "") +
                   "[Middle omitted]\n" + text[-450:] +
                   f"\n[Full result in external context {item_id}; use load_context]")
        self.metrics["compressed_results"] += 1
        return result.model_copy(update={"content": reduced})


class ToolOptimizationMiddleware(AgentMiddleware):
    def __init__(self, runtime: ToolRuntime):
        self.runtime = runtime

    async def awrap_model_call(self, request, handler):
        schemas = self.runtime.select_schemas(request.messages, request.tools)
        return await handler(request.override(tools=schemas))

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call.get("name", "")
        args = request.tool_call.get("args") or {}
        if name in CACHEABLE:
            content = await self.runtime.cached_text(name, args, lambda: handler(request))
            result = ToolMessage(content=content, tool_call_id=request.tool_call["id"], name=name)
        elif name.startswith(MCP_PREFIXES):
            self.runtime._loop_locks()
            async with self.runtime.mcp_lock:
                result = await handler(request)
        else:
            result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result
        try:
            item_id = self.runtime.context.store.put_result(request.tool_call, result)
        except (OSError, ValueError, TypeError) as exc:
            print(f"[Tools] Could not store complete result: {exc}")
            item_id = None
        return result
