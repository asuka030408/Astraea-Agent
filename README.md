# Astraea Agent

A small Python coding agent with command-line, web, and native desktop interfaces. LangChain's `create_agent` runs the model/tool loop.

Project layout:

- `main.py`: minimal program entry point.
- `astraea/cli.py`: terminal input/output; `main()` is the CLI, not the agent loop.
- `astraea/agent.py`: `build_agent()` assembles the model, tools, and instructions.
- `astraea/config.py`: reads `.env` and the user settings overlay, and validates settings.
- `astraea/tools/`: tool implementations and their registration list.
- `astraea/web.py` and `astraea/static/`: local API and thin browser interface.
- `astraea/desktop_qt.py`: native Qt desktop client with an integrated settings dialog.
- `astraea/context.py`: measures context use and applies the T1–T5 policy.
- `astraea/memory.py`: post-turn fact extraction, Markdown memory, and the vector store.
- `astraea/storage.py`: saves full sessions and archived context under `data/`.
- `astraea/mcp_tools.py`: owns the persistent MCP session.

## Features at a glance

| Area | What it does |
| --- | --- |
| Interfaces | A streaming web UI (FastAPI), a native Qt desktop client, and a terminal CLI. All three share the same LangChain `create_agent` loop. |
| Tools | Six built-in tools: `read_file`, `write_file`, `search_files`, `run_command` (bounded to pytest and read-only git), keyless `search_web`, and `search_memory`. Local MCP reference servers add Fetch, Filesystem, Time, and Git (when the workspace is a repository). |
| Context management | Before every request it counts tokens over message text **plus tool-call arguments plus tool schemas**, then runs a five-tier policy T1–T5: shorten old large tool results, archive older failed attempts, write a nine-section checkpoint with a separate model, and retrieve archived text when needed. The full original transcript is never thrown away: it stays in `data/sessions/`, and moved text is indexed under `data/archives/`. If a request still cannot fit, it fails loudly instead of silently dropping information. |
| Memory | After each finished turn a separate model request extracts durable, explicitly stated facts into two Markdown files: `profile.md` (stable preferences, injected as a system message on every request) and `events.md` (specific events and decisions, indexed into a local SQLite vector store). The model recalls them on demand through `search_memory` using exact cosine search. Markdown is the source of truth, so vectors can be rebuilt and an embedding outage never loses memory. |
| Configuration | The web **设置** dialog takes chat and embedding API keys, model names, context window, and workspace. Values go to git-ignored `data/settings.json`, override `.env`, are always masked in the API response, and rebuild the model immediately. A new user can set up the client without editing any source file. |
| MCP lifecycle | The web process starts the MCP servers once (~6 s) on the first turn, caches their tool list, and keeps one session for its whole lifetime. Turns sharing it are serialized; a transport failure reconnects once. CLI/desktop still use a short-lived connection per turn. |
| Persistence | Every session's complete transcript, active context, and compaction tier are saved locally; the web UI can resume earlier conversations. |

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

Use **上下文管理** in the chat header to inspect the selected conversation. The window shows its current active `messages` first, then text moved to its external archive. It refreshes while open. Archived text is scoped to that conversation and is not a cross-conversation memory feature.

## Memory

After each completed turn in the CLI, web, or desktop client, a separate model request extracts explicit, durable information stated by the user. Stable habits, preferences, hobbies, and ongoing goals go to `data/memory/profile.md`. Specific important events and decisions go to `data/memory/events.md`. Each entry records the source session ID. The extractor skips credentials and uncertain inferences; duplicate text is not added again. A directly contradicted profile entry may be removed.

`profile.md` is injected into every main chat request as a system message. It is not copied into the saved conversation history, so edits to the profile take effect on the next turn. `events.md` is the readable source for the local SQLite vector store at `data/memory/vectors.sqlite3`. Missing or stale vectors are rebuilt from that Markdown file on later turns. The `search_memory` tool performs exact cosine search and returns top-K matches when the model chooses to recall a past event or decision. Its results are not added to every request automatically. The embedding configuration lives in the web **设置** dialog (or `.env` under `ASTRAEA_EMBEDDING_*`); the default model is Bailian `text-embedding-v4` with 1024 dimensions. An embedding failure leaves the Markdown intact and reports a Memory failure without discarding the chat answer.

Native desktop client:

```powershell
.\.venv\Scripts\python.exe desktop_main.py
```

On Windows, you can also double-click `start_desktop.cmd` after installing the dependencies.

The desktop client opens to the chat workbench. Use **设置** in the sidebar to enter the API Key, API address, model, and workspace. The Key is masked and used only in memory for that run. The desktop client does not need an `.env` API Key. **摘要模型** is optional and defaults to the main model. Press Enter to send; Shift+Enter adds a line break.

The four file/command tools act on the `workspace/` folder inside the project directory by default. Set `ASTRAEA_WORKSPACE` in the web **设置** dialog or `.env` to an existing project directory if you want Astraea to work elsewhere. These tools are `read_file`, `write_file`, `search_files`, and `run_command`. The command tool runs only `pytest`, `python -m pytest`, or read-only `git status/diff/log`, with a 30-second timeout; it is a bounded demo tool, not a process sandbox.

Local MCP reference servers from `vendor/mcp-servers` add Fetch (read a known URL), Filesystem (inside the configured workspace), and Time. Git tools are enabled when the workspace is a Git repository and are confined to that repository. The reference Memory server is excluded; Astraea keeps its own Memory implementation. The native `search_web` tool provides keyless web search results; Fetch can then read selected URLs. The web process keeps one MCP session alive for its whole lifetime: servers start once (about 6 seconds) on the first chat turn and the discovered tool list is cached, so later turns have no startup delay. The session is closed on shutdown, reconnects once after a transport failure, and fails loudly if a configured server reports no tools. Agent turns sharing that session are serialized so concurrent chats cannot mix requests on one client session. The CLI and desktop clients create a new event loop per turn, so they still use a short-lived connection; a stdio transport cannot be shared between loops.

In the CLI, use `/clear` to start a new conversation or `/exit` to quit.

## Context compaction

Set `ASTRAEA_CONTEXT_WINDOW` in the web **设置** dialog or `.env` to the context size of your chosen model. The default is 64,000 tokens. The estimate includes message text, tool-call arguments (for example the body of a `write_file` call), and the tool definitions attached to every request. Built-in tool schemas are measured exactly; MCP tool schemas are measured after their first discovery and a conservative allowance is used before that. Provider tokenizers differ, so this remains a budget setting rather than a guarantee.

Before each new model request, the harness checks the active history plus the new question. Astraea's five tiers are:

- **T1 (30%)**: Python shortens old, large tool results. No model call. The compressible boundary is the later of the second-to-last user turn and the second-to-last tool round-trip, so a single long turn with many tool calls can still be compacted without splitting a tool call from its result.
- **T2 (60%)**: Python archives up to two older failed tool attempts.
- **T3 (75%)**: Python archives any remaining older failed tool attempts.
- **T4 (90%)**: a separate compactor model writes a nine-section checkpoint. Set `ASTRAEA_COMPACTOR_MODEL` to use another model; otherwise it uses the main model.
- **T5 (95% or an explicit question about earlier details)**: retrieve relevant archived text, within the available budget. If the request still cannot fit, stop with an error instead of silently deleting information.

These percentages are Astraea's chosen defaults. T1–T3 only have work to do when the relevant old tool events exist. T4's summary enters the next request's `messages`. Complete original messages and the active context remain in `data/sessions/`; moved text is also indexed in `data/archives/`. Retrieval uses simple local text matching. There is no subagent or repository-file reader in this version.

Only T4 summaries use an additional model request and incur model usage. `/clear` starts a new CLI session; previous sessions remain saved on disk. The web interface can resume saved conversations; the CLI has no resume command.

## Verify compaction

Run the deterministic checks without API requests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Then run the real-model T4 smoke check (two API requests):

```powershell
.\.venv\Scripts\python.exe tests\live_smoke.py
```

It places a check word in an early message, forces T4, checks that the summary enters the next request, and asks the model to recall the word. This checks basic continuity; it does not prove that every possible detail survives compression.

To check the web API and a real tool call with a temporary workspace, run `tests\live_web_smoke.py` the same way. That test also uses the configured model API.

## What to extend next

1. Add approval handling for file writes and commands (there is no safety review layer yet).
2. Add more MCP servers, including hosted external services.
3. Add CLI/desktop session resume and a UI for inspecting the Memory files.
