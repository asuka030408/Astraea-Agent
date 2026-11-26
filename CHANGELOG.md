# Changelog

## 2026-11-26 — Adaptive context and tool runtime

### Context management

- Select a bounded context before every model call, including calls within a long tool round. Keep the complete session transcript in local storage.
- Store tool results and lightweight task state outside model requests. Add session-scoped search and load tools for earlier results and dialogue.
- Replace the fixed T1–T5 thresholds with separate budgets for dialogue, tool results, retrieved content, and task state. Create a reusable dialogue checkpoint only when its budget is exceeded.
- Show the latest model request view, external records, and budget data in the Web context inspector.

### Tools

- Add eight workspace file tools for listing, metadata, line reads, append, exact replacement, copy, move, and recoverable deletion.
- Add four conversation-context tools and three runtime tools, bringing the registered first-party total to 21 before MCP tools.
- Select relevant tool definitions for each model request. Add `search_tools` and `load_tools` for explicit discovery and activation, and shorten descriptions without changing required parameters.
- Add bounded parallel execution for independent read-only file steps through `run_readonly_plan`. Keep calls on the shared MCP session serialized.
- Cache only read-only file operations with file metadata invalidation. Prefetch at most two explicitly named files while the model plans.
- Compress large tool results only in the model-visible view; preserve full results in session-scoped external storage.

### Verification

- 40 automated tests pass. A local MCP time-tool smoke check returned a nonempty result through the optimized Agent path.
- In 14 paired `deepseek-chat` file-task runs, both versions completed 14/14 tasks correctly. Mean turn time was 1,474.8 ms before and 1,159.7 ms after (21.37% faster in this sample); measured accuracy changed by 0 percentage points. These short-task measurements are sensitive to model and network variance.
- See `docs/context-optimization-report.md` and `docs/tool-optimization-report.md` for methods, raw data, and limitations.

### Current limits

- Tool routing uses explicit rules; it does not load MCP server processes on demand.
- Prediction is limited to explicit workspace file reads. The cache is process-local and is never used for writes or deletions.
- File writes and other modifying tools do not yet have an approval layer.
