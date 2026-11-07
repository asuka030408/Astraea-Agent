"""Connect Astraea to the vendored MCP reference servers.

The web process keeps one MCP session alive for its whole lifetime. CLI and
desktop clients create a new event loop per turn, so they still use a
short-lived connection: a stdio transport cannot be shared across loops.
"""

import asyncio
import shutil
import subprocess
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from langchain.mcp import MCPAdapter

from astraea.config import PROJECT_DIR, Settings
from astraea.context import set_discovered_tools


VENDOR_DIR = PROJECT_DIR / "vendor" / "mcp-servers"

# Set by the web app's lifespan hook. Only this loop owns the persistent session.
_persistent_loop = None
_turn_lock = None

_DISCONNECT_HINTS = (
    "connection closed", "connection lost", "broken pipe", "closed pipe",
    "transport closed", "stream closed", "stdio", "eof", "task group",
    "session terminated", "generator didn't stop",
)


def _venv_command(name: str) -> str:
    folder = Path(sys.executable).parent
    candidate = folder / (name + (".exe" if sys.platform == "win32" else ""))
    if not candidate.is_file():
        raise FileNotFoundError(f"MCP server is not installed: {candidate}")
    return str(candidate)


def mcp_config(settings: Settings) -> dict:
    """Use project-local server code; Git is enabled only for a Git workspace."""
    node = shutil.which("node")
    filesystem = VENDOR_DIR / "src" / "filesystem" / "dist" / "index.js"
    if not node or not filesystem.is_file():
        raise FileNotFoundError("Build the vendored Filesystem MCP server before starting Astraea")
    servers = {
        "fetch": {"command": _venv_command("mcp-server-fetch"),
                  "args": ["--no-readability-js"], "env": {"PYTHONIOENCODING": "utf-8"}},
        "filesystem": {"command": node, "args": [str(filesystem), str(settings.workspace)]},
        "time": {"command": _venv_command("mcp-server-time"), "args": []},
    }
    repo = subprocess.run(["git", "-C", str(settings.workspace), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, timeout=5, check=False)
    if repo.returncode == 0:
        servers["git"] = {"command": _venv_command("mcp-server-git"),
                          "args": ["--repository", repo.stdout.strip()]}
    return {"mcpServers": servers}


def _check_servers(config: dict, tools: list) -> None:
    """Fail loudly when a configured server did not come up.

    MCPAdapter silently drops a server that fails to start, but the system
    prompt still tells the model to use its tools.
    """
    names = {tool.name for tool in tools}
    missing = [server for server in config["mcpServers"]
               if not any(name == server or name.startswith(server + "_") for name in names)]
    if missing:
        raise RuntimeError(
            "MCP server(s) reported no tools: " + ", ".join(missing) +
            f" (got {len(names)} tools). Check the server executables and the workspace path."
        )


def _looks_like_disconnect(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in _DISCONNECT_HINTS)


class _MCPSession:
    """One long-lived MCPAdapter session, guarded against concurrent starts."""

    def __init__(self):
        self._stack = None
        self._tools = None
        self._lock = None
        self._loop = None

    async def tools(self, settings: Settings) -> list:
        loop = asyncio.get_running_loop()
        if self._stack is not None and self._loop is not loop:
            await self.close()
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._stack is not None:
                return self._tools
            config = mcp_config(settings)
            stack = AsyncExitStack()
            try:
                adapter = await stack.enter_async_context(MCPAdapter(config))
                tools = await adapter.list_tools()
                _check_servers(config, tools)
            except Exception:
                await stack.aclose()
                raise
            self._stack = stack
            self._tools = tools
            self._loop = loop
            print(f"[MCP] persistent session ready: {len(tools)} tools from {', '.join(config['mcpServers'])}")
            return tools

    async def close(self) -> None:
        stack = self._stack
        self._stack, self._tools, self._loop, self._lock = None, None, None, None
        if stack is not None:
            await stack.aclose()


_session = _MCPSession()


def bind_event_loop(loop) -> None:
    """Declare which event loop may own the persistent session."""
    global _persistent_loop, _turn_lock
    _persistent_loop = loop
    _turn_lock = None


def _get_turn_lock():
    """Serialize agent turns sharing one MCP session (the adapter warns about
    concurrent requests on a reused client session)."""
    global _turn_lock
    if _turn_lock is None:
        _turn_lock = asyncio.Lock()
    return _turn_lock


def main_event_loop():
    return _persistent_loop


async def close_mcp_session() -> None:
    await _session.close()


async def reset_mcp_session() -> None:
    """Drop the session (e.g. after the workspace changes); it reconnects lazily."""
    await _session.close()


async def invoke_with_mcp(model, settings: Settings, messages: list, callbacks=None):
    """Run one agent turn, reusing the persistent MCP session when possible."""
    from astraea.agent import build_agent

    async def run(tools):
        set_discovered_tools(tools)
        agent = build_agent(model, extra_tools=tools)
        return await agent.ainvoke({"messages": messages}, config={"callbacks": callbacks or []})

    if asyncio.get_running_loop() is _persistent_loop:
        async with _get_turn_lock():
            tools = await _session.tools(settings)
            try:
                return await run(tools)
            except Exception as exc:
                if not _looks_like_disconnect(exc):
                    raise
                print(f"[MCP] connection lost ({type(exc).__name__}); reconnecting once")
                await _session.close()
                tools = await _session.tools(settings)
                return await run(tools)

    # CLI / desktop: one event loop per turn, so keep the old behavior.
    config = mcp_config(settings)
    async with MCPAdapter(config) as adapter:
        tools = await adapter.list_tools()
        _check_servers(config, tools)
        return await run(tools)
