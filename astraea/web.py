"""Small local HTTP interface using the same agent and context manager as the CLI."""

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import AsyncCallbackHandler
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

from astraea import config as config_module
from astraea.agent import build_model
from astraea.config import (DEFAULTS, SECRET_KEYS, SETTING_KEYS, effective_settings,
                            load_settings, mask_secret, save_user_settings, user_settings)
from astraea.context import ContextManager, PROFILE_PREFIX, estimate_tokens, usage_tokens
from astraea.memory import MemoryManager, read_profile_markdown
from astraea.mcp_tools import (bind_event_loop, close_mcp_session, invoke_with_mcp,
                               main_event_loop, reset_mcp_session)
from astraea.storage import list_archive_fragments, list_sessions


@asynccontextmanager
async def lifespan(_app):
    bind_event_loop(asyncio.get_running_loop())
    try:
        yield
    finally:
        await close_mcp_session()


app = FastAPI(title="Astraea Agent", lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class Session:
    context: ContextManager
    lock: Lock = field(default_factory=Lock)


sessions: dict[str, Session] = {}
_runtime = None


def current_usage_tokens(context: ContextManager) -> int:
    profile = read_profile_markdown()
    profile_tokens = (estimate_tokens([SystemMessage(content=PROFILE_PREFIX + profile)]) if profile else 0)
    return usage_tokens(context.active) + profile_tokens


def runtime():
    global _runtime
    if _runtime is None:
        settings = load_settings()
        model = build_model(settings)
        _runtime = (settings, model, build_model(settings, settings.compactor_model))
    return _runtime


def reset_runtime() -> None:
    """Drop cached model/settings so the next turn uses saved UI values."""
    global _runtime
    _runtime = None


def _run_on_main_loop(coro):
    """Keep sync endpoints on the loop that owns the persistent MCP session."""
    loop = main_event_loop()
    if loop is None or loop.is_closed():
        return asyncio.run(coro)
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=50_000)


class SettingsUpdate(BaseModel):
    values: dict[str, str | None] = Field(default_factory=dict)


def _settings_view() -> dict:
    values = effective_settings()
    return {
        "values": {key: mask_secret(value) if key in SECRET_KEYS else value
                   for key, value in values.items()},
        "configured": bool(values["ASTRAEA_API_KEY"].strip()
                           and values["ASTRAEA_API_KEY"].strip() != "your-api-key"),
        "embedding_configured": bool(values["ASTRAEA_EMBEDDING_API_KEY"].strip()),
        "saved_keys": sorted(user_settings()),
        "defaults": DEFAULTS,
        "settings_file": str(config_module.USER_SETTINGS_FILE),
    }


@app.get("/api/settings")
def read_settings():
    """Effective values with secrets masked; secrets are never sent back in full."""
    return _settings_view()


@app.post("/api/settings")
async def update_settings(payload: SettingsUpdate):
    values = {key: value for key, value in payload.values.items() if key in SETTING_KEYS}
    for key in ("ASTRAEA_CONTEXT_WINDOW", "ASTRAEA_EMBEDDING_DIMENSIONS"):
        candidate = values.get(key)
        if candidate not in (None, ""):
            try:
                number = int(candidate)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"{key} 必须是整数") from exc
            if number < (1000 if key == "ASTRAEA_CONTEXT_WINDOW" else 1):
                raise HTTPException(status_code=400, detail=f"{key} 数值不合法")
    for key in ("ASTRAEA_BASE_URL", "ASTRAEA_EMBEDDING_BASE_URL"):
        candidate = values.get(key)
        if candidate and not str(candidate).startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail=f"{key} 必须是 http(s) 地址")

    before = effective_settings().get("ASTRAEA_WORKSPACE")
    save_user_settings(values)
    reset_runtime()
    if before != effective_settings().get("ASTRAEA_WORKSPACE"):
        await reset_mcp_session()
    return _settings_view()


class StreamCallbacks(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.tool_names = {}

    async def on_llm_new_token(self, token: str, **kwargs):
        if token:
            self.queue.put_nowait({"type": "token", "text": token})

    async def on_tool_start(self, serialized: dict, input_str: str, *, run_id, **kwargs):
        name = serialized.get("name", "tool")
        self.tool_names[run_id] = name
        self.queue.put_nowait({"type": "tool_call", "name": name, "detail": str(input_str)[:4000]})

    async def on_tool_end(self, output, *, run_id, **kwargs):
        name = self.tool_names.pop(run_id, "tool")
        self.queue.put_nowait({"type": "tool_result", "name": name, "detail": str(output)[:4000]})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/app.css")
def css():
    return FileResponse(STATIC_DIR / "app.css", media_type="text/css")


@app.get("/app.js")
def javascript():
    return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")


@app.post("/api/session")
def new_session():
    try:
        settings, _, compactor_model = runtime()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session_id = uuid4().hex
    context = ContextManager(compactor_model, settings.context_window)
    context.session_id = session_id
    context._save()
    sessions[session_id] = Session(context)
    return {"session_id": session_id, "workspace": str(settings.workspace)}


@app.get("/api/sessions")
def session_list():
    return {"sessions": list_sessions()}


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    try:
        settings, _, compactor_model = runtime()
        context = get_session(session_id, compactor_model, settings.context_window)
        with context.lock:
            history = [{"role": item.type, "content": str(item.content)} for item in context.context.history
                       if item.type == "human" or (item.type == "ai" and item.content)]
            return {"session_id": session_id, "messages": history, "context_tier": context.context.tier,
                    "context_percent": round(100 * current_usage_tokens(context.context) / settings.context_window, 1),
                    "workspace": str(settings.workspace)}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


def get_session(session_id: str, compactor_model, context_window: int) -> Session:
    session = sessions.get(session_id)
    if session is None:
        session = Session(ContextManager.resume(compactor_model, context_window, session_id))
        sessions[session_id] = session
    return session


@app.get("/api/sessions/{session_id}/context")
def active_context(session_id: str):
    try:
        settings, _, compactor_model = runtime()
        session = get_session(session_id, compactor_model, settings.context_window)
        with session.lock:
            context = session.context
            profile = read_profile_markdown()
            display_messages = ([{"role": "system", "content": PROFILE_PREFIX + profile, "tool_calls": []}]
                                if profile else [])
            messages = display_messages + [{"role": item.type, "content": str(item.content),
                         "tool_calls": getattr(item, "tool_calls", []) or []}
                        for item in context.active]
            return {"session_id": session_id, "tier": context.tier,
                    "tokens_estimate": current_usage_tokens(context),
                    "context_window": settings.context_window,
                    "summary": context.summary, "messages": messages,
                    "archives": list_archive_fragments(session_id)}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@app.post("/api/chat")
def chat(request: ChatRequest):
    session = sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new chat.")
    settings, model, _ = runtime()
    with session.lock:
        try:
            messages = session.context.prepare(request.message.strip(), read_profile_markdown())
            prior_count = len(messages)
            result = _run_on_main_loop(invoke_with_mcp(model, settings, messages))
            new_messages = result["messages"][prior_count:]
            session.context.record(result["messages"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Agent request failed: {exc}") from exc

        try:
            MemoryManager(build_model(settings)).process_turn(session.context.session_id, session.context.history)
            memory_saved = True
        except Exception as exc:
            print(f"[Memory] Could not process turn: {type(exc).__name__}: {exc}")
            memory_saved = False

        events = []
        for message in new_messages:
            for call in getattr(message, "tool_calls", []) or []:
                events.append({"kind": "tool_call", "name": call["name"], "detail": call.get("args", {})})
            if message.type == "tool":
                events.append({"kind": "tool_result", "name": message.name or "tool", "detail": str(message.content)[:4000]})

        return {
            "answer": str(result["messages"][-1].content),
            "events": events,
            "context_tier": session.context.tier,
            "context_percent": round(100 * current_usage_tokens(session.context) / settings.context_window, 1),
            "memory_saved": memory_saved,
        }


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    session = sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new chat.")
    settings, model, _ = runtime()

    async def stream():
        if not session.lock.acquire(blocking=False):
            yield json.dumps({"type": "error", "message": "This chat is already processing a request."}) + "\n"
            return
        task = None
        try:
            try:
                messages = session.context.prepare(request.message.strip(), read_profile_markdown())
            except Exception as exc:
                yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"
                return
            queue = asyncio.Queue()
            callbacks = StreamCallbacks(queue)

            async def run_turn():
                try:
                    result = await invoke_with_mcp(model, settings, messages, callbacks=[callbacks])
                    session.context.record(result["messages"])
                    queue.put_nowait({"type": "answer", "text": str(result["messages"][-1].content)})
                    try:
                        memory = MemoryManager(build_model(settings))
                        await asyncio.to_thread(memory.process_turn, session.context.session_id, session.context.history)
                        memory_saved = True
                    except Exception as exc:
                        print(f"[Memory] Could not process turn: {type(exc).__name__}: {exc}")
                        memory_saved = False
                    queue.put_nowait({"type": "done", "context_tier": session.context.tier,
                                      "context_percent": round(100 * current_usage_tokens(session.context) / settings.context_window, 1),
                                      "memory_saved": memory_saved})
                except Exception as exc:
                    queue.put_nowait({"type": "error", "message": f"Agent request failed: {exc}"})
                finally:
                    queue.put_nowait(None)

            task = asyncio.create_task(run_turn())
            yield json.dumps({"type": "start"}) + "\n"
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            session.lock.release()

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
