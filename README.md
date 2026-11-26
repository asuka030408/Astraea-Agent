# Astraea Agent

A small Python coding agent with command-line, web, and native desktop interfaces. LangChain's `create_agent` runs the model/tool loop.

See [CHANGELOG.md](CHANGELOG.md) for the latest context and tool runtime update.

Project layout:

- `main.py`: minimal program entry point.
- `astraea/cli.py`: terminal input/output; `main()` is the CLI, not the agent loop.
- `astraea/agent.py`: `build_agent()` assembles the model, tools, and instructions.
- `astraea/config.py`: reads `.env` and the user settings overlay, and validates settings.
- `astraea/tools/`: tool implementations and their registration list.
- `astraea/web.py` and `astraea/static/`: local API and thin browser interface.
- `astraea/desktop_qt.py`: native Qt desktop client with an integrated settings dialog.
- `astraea/context.py`: selects a bounded model-visible context before every model call.
- `astraea/context_store.py`: stores complete tool results and task state outside model requests.
- `astraea/tool_runtime.py`: selects tool schemas, compresses model-visible results, caches safe reads, and runs read-only plans.
- `astraea/memory.py`: post-turn fact extraction, Markdown memory, and the vector store.
- `astraea/storage.py`: saves full sessions and archived context under `data/`.
- `astraea/mcp_tools.py`: owns the persistent MCP session.

## Features at a glance

| Area | What it does |
| --- | --- |
| Interfaces | A streaming web UI (FastAPI), a native Qt desktop client, and a terminal CLI. All three share the same LangChain `create_agent` loop. |
| Tools | Fourteen built-in tools cover bounded file operations, tests/read-only Git commands, web search and memory search. Four session-bound context tools and three runtime tools bring the registered total to twenty-one before MCP. Model requests receive a relevant subset, with `search_tools` and `load_tools` available for discovery. MCP adds Fetch, Filesystem, Time and Git (for Git workspaces). `delete_file` moves a file to recoverable workspace trash. |
| Context management | A middleware selects context before **every model request**, including intermediate tool rounds. It preserves the full transcript in `data/sessions/`, saves tool results and task state under `data/context/`, and sends only relevant tool groups within separate dialogue, tool and retrieval budgets. Older dialogue gets a reusable checkpoint only when its budget is exceeded. |
| Memory | After each finished turn a separate model request extracts durable, explicitly stated facts into two Markdown files: `profile.md` (stable preferences, injected as a system message on every request) and `events.md` (specific events and decisions, indexed into a local SQLite vector store). The model recalls them on demand through `search_memory` using exact cosine search. Markdown is the source of truth, so vectors can be rebuilt and an embedding outage never loses memory. |
| Configuration | The web **设置** dialog takes chat and embedding API keys, model names, context window, and workspace. Values go to git-ignored `data/settings.json`, override `.env`, are always masked in the API response, and rebuild the model immediately. A new user can set up the client without editing any source file. |
| MCP lifecycle | The web process starts the MCP servers once (~6 s) on the first turn, caches their tool list, and keeps one session for its whole lifetime. Turns sharing it are serialized; a transport failure reconnects once. CLI/desktop still use a short-lived connection per turn. |
| Persistence | Every session's complete transcript, last model request view, final response and budget report are saved locally; the web UI can resume earlier conversations. |

All session data, memory, secrets, and the third-party `vendor/` tree are git-ignored — a clone contains source code only.

## Setup (PowerShell)

From the directory containing this repository:

```powershell
cd path\to\Astraea-Agent
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# The MCP reference servers are third-party code and are NOT included in this repository.
git clone --depth 1 --filter=blob:none --sparse https://github.com/modelcontextprotocol/servers.git vendor/mcp-servers
git -C vendor/mcp-servers sparse-checkout set src/fetch src/filesystem src/git src/time
npm install --prefix vendor/mcp-servers/src/filesystem --cache vendor/mcp-servers/.npm-cache
Copy-Item .env.example .env
```

A fresh copy of the client does **not** need an edited `.env`: open the web interface, click **设置** in the header, and fill in the chat model API Key plus the Bailian embedding Key (optional, for vector memory). Values are saved to `data/settings.json` (git-ignored) and take precedence over `.env`. Keys are always masked in the UI and never returned in full by the API. Alternatively, edit `.env` and replace `your-api-key`; the example uses DeepSeek's OpenAI-compatible endpoint, and `ASTRAEA_BASE_URL` / `ASTRAEA_MODEL` select another compatible provider.

## Run

CLI:

```powershell
.\.venv\Scripts\python.exe main.py
```

Web interface:

```powershell
.\.venv\Scripts\python.exe web_main.py
```

Open `http://127.0.0.1:8000` in your browser. The web interface streams the answer as it is generated, shows tool calls/results and context usage, and has a left sidebar of saved conversations. Use **停止** to cancel the current turn; canceled turns are not saved. Use **新对话** to start another chat, or select an earlier chat to view and continue it. Only one request runs at a time per chat.

Use **设置** in the chat header to enter the chat model API Key, API address, model, summary model, context window, workspace, and the Bailian embedding credentials. The dialog opens automatically when no API Key is configured, so a new user can set up the client without touching project files. Saving writes `data/settings.json` (which overrides `.env`) and immediately rebuilds the cached model; start a new chat to pick up a changed context window or workspace. Leave a Key field blank to keep the stored value. The server binds to `127.0.0.1` only: do not expose it to a LAN without adding authentication.

Use **上下文管理** in the chat header to inspect the selected conversation. The window shows the last model request view and final response first, then legacy archives and externally stored tool results. It refreshes while open. External context is scoped to that conversation and is separate from cross-conversation Memory.

## Memory

After each completed turn in the CLI, web, or desktop client, a separate model request extracts explicit, durable information stated by the user. Stable habits, preferences, hobbies, and ongoing goals go to `data/memory/profile.md`. Specific important events and decisions go to `data/memory/events.md`. Each entry records the source session ID. The extractor skips credentials and uncertain inferences; duplicate text is not added again. A directly contradicted profile entry may be removed.

`profile.md` is injected into every main chat request as a system message. It is not copied into the saved conversation history, so edits to the profile take effect on the next turn. `events.md` is the readable source for the local SQLite vector store at `data/memory/vectors.sqlite3`. Missing or stale vectors are rebuilt from that Markdown file on later turns. The `search_memory` tool performs exact cosine search and returns top-K matches when the model chooses to recall a past event or decision. Its results are not added to every request automatically. The embedding configuration lives in the web **设置** dialog (or `.env` under `ASTRAEA_EMBEDDING_*`); the default model is Bailian `text-embedding-v4` with 1024 dimensions. An embedding failure leaves the Markdown intact and reports a Memory failure without discarding the chat answer.

Native desktop client:

```powershell
.\.venv\Scripts\python.exe desktop_main.py
```

On Windows, you can also double-click `start_desktop.cmd` after installing the dependencies.

The desktop client opens to the chat workbench. Use **设置** in the sidebar to enter the API Key, API address, model, and workspace. The Key is masked and used only in memory for that run. The desktop client does not need an `.env` API Key. **摘要模型** is optional and defaults to the main model. Press Enter to send; Shift+Enter adds a line break.

The file/command tools act on the `workspace/` folder inside the project directory by default. Set `ASTRAEA_WORKSPACE` in the web **设置** dialog or `.env` to an existing project directory if you want Astraea to work elsewhere. Native file tools include reading, writing, searching, listing, inspecting metadata, line-range reading, appending, unique text replacement, copying, moving and recoverable deletion. `run_command` permits only `pytest`, `python -m pytest`, or read-only `git status/diff/log`, with a 30-second timeout; it is a bounded demo tool, not a process sandbox.

Local MCP reference servers from `vendor/mcp-servers` add Fetch (read a known URL), Filesystem (inside the configured workspace), and Time. Git tools are enabled when the workspace is a Git repository and are confined to that repository. The reference Memory server is excluded; Astraea keeps its own Memory implementation. The native `search_web` tool provides keyless web search results; Fetch can then read selected URLs. The web process keeps one MCP session alive for its whole lifetime: servers start once (about 6 seconds) on the first chat turn and the discovered tool list is cached, so later turns have no startup delay. The session is closed on shutdown, reconnects once after a transport failure, and fails loudly if a configured server reports no tools. Agent turns sharing that session are serialized so concurrent chats cannot mix requests on one client session. The CLI and desktop clients create a new event loop per turn, so they still use a short-lived connection; a stdio transport cannot be shared between loops.

In the CLI, use `/clear` to start a new conversation or `/exit` to quit.

## Tool call optimization

Before each model call, `ToolOptimizationMiddleware` selects relevant tool definitions and shortens their descriptions while preserving names, parameter types and required fields. `search_tools` searches the full registered catalog; `load_tools` exposes requested tools on the next model call. Tool execution still uses the complete registry. For explicit read requests, up to two existing workspace files may be prefetched while the model plans. A two-minute, fingerprint-checked cache is limited to read-only file tools; changes to file size or modification/creation time invalidate entries. Writes, deletions, time calls and MCP mutations are never cached.

Independent native file reads can run concurrently through LangGraph's parallel tool calls or `run_readonly_plan` (up to eight steps, four at a time). The plan returns ordered previews and per-step IDs for loading full results. Calls sharing the persistent MCP adapter are serialized to avoid mixing concurrent requests on one session. Large results are preserved in external context and reduced only in the model-visible view; the full session transcript stays intact. The web context API exposes tool counters and the current tool-schema budget.

Run `tests\benchmark_tool_optimization.py` for reproducible, controlled-latency microbenchmarks. `tests\live_tool_optimization_eval.py` runs paired baseline/optimized tasks against the configured chat model and writes results under `docs/`. It uses the model API and may incur charges. See `docs/tool-optimization-report.md` for measurements and limits.

## Adaptive context management

Set `ASTRAEA_CONTEXT_WINDOW` in **设置** or `.env` to the chosen model's context size (default 64,000). The estimate includes system text, message content, tool-call arguments and the actual tool definitions available on each model call. It uses `cl100k_base`; provider accounting may differ, so 2% is left as a safety margin and the output gets a separate 15% reserve (at most 4,096 tokens). The current user request, system prompt, tool definitions and a capped long-term profile are fixed costs; if they alone do not fit, the turn fails with a clear error.

The remaining input budget is split into **46% prior dialogue/checkpoint, 34% ordinary tool results, 15% retrieved context/Memory results, and 5% task state and manifest**. Unused portions need not be filled. Tool-call/result groups are ranked against the current user request and retained as complete pairs; unrelated groups are removed only from the model request. A large selected result is clipped with an external context ID. A request can therefore use two earlier results without resending the other eighteen. The complete graph state and session transcript are unchanged.

Every completed tool result, including file reads, is saved under `data/context/<session>/items/`. `search_context` finds earlier records within the current session and `load_context` reads one by ID. `search_history` and `load_history` can retrieve older dialogue from the full saved transcript. A lightweight `task_state` record stores the last goal, response and recent file paths. These external records are **conversation context**, not long-term Memory. Older `data/archives/` records remain readable for sessions created before this policy.

Long dialogue is compressed separately from tool output: when dialogue exceeds its budget, a separate model updates a concise checkpoint of goals, constraints, decisions and open questions. The checkpoint is reused across model calls and updated only after enough new dialogue accumulates. If checkpoint generation fails, recent dialogue and external records remain available. `/clear` starts a new CLI session; the web UI can resume saved sessions.

## Verify context management

Run the deterministic checks without API requests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For an optional real-model smoke check (uses the configured API):

```powershell
.\.venv\Scripts\python.exe tests\live_smoke.py
```

It checks whether an early fact remains recoverable after the adaptive selector trims a long conversation. This checks basic continuity, not perfect recall.

To check the web API and a real tool call with a temporary workspace, run `tests\live_web_smoke.py` the same way. That test also uses the configured model API.

## What to extend next

1. Add approval handling for file writes and commands (there is no safety review layer yet).
2. Add more MCP servers, including hosted external services.
3. Add CLI/desktop session resume and a UI for inspecting the Memory files.
