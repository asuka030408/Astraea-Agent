"""Per-model-call context selection; the complete transcript stays on disk."""

import json
from threading import Lock
from uuid import uuid4

import tiktoken
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from astraea.context_store import ContextStore, relevance
from astraea.storage import load_session, save_session


ENCODING = tiktoken.get_encoding("cl100k_base")
PROFILE_PREFIX = "Long-term user profile (current user message takes precedence):\n"
SUMMARY_PREFIX = "Task checkpoint from earlier dialogue:\n"
STATE_PREFIX = "Stored task state (may be stale):\n"
MANIFEST = ("Older tool results and file contents are stored outside this request. "
            "Use search_context/load_context for tool data and search_history/load_history for old dialogue.")
MCP_SCHEMA_RESERVE = 3800
_TOOLS_LOCK = Lock()
_discovered_tools = None
_schema_tokens = None


def _message_payload(message) -> str:
    parts = [str(message.content)]
    parts.extend(json.dumps(call, ensure_ascii=False, default=str)
                 for call in (getattr(message, "tool_calls", None) or []))
    for attr in ("name", "tool_call_id"):
        if getattr(message, attr, None):
            parts.append(str(getattr(message, attr)))
    return "\n".join(parts)


def estimate_tokens(messages: list) -> int:
    return sum(len(ENCODING.encode(_message_payload(m))) + 12 for m in messages)


def _measure_tool_schemas(tools) -> int:
    from langchain_core.utils.function_calling import convert_to_openai_tool
    total = 0
    for item in tools:
        try:
            total += len(ENCODING.encode(json.dumps(convert_to_openai_tool(item), ensure_ascii=False, default=str))) + 8
        except Exception:
            total += 150
    return total


def set_discovered_tools(tools) -> None:
    global _discovered_tools, _schema_tokens
    with _TOOLS_LOCK:
        _discovered_tools, _schema_tokens = list(tools or []), None


def tool_schema_tokens() -> int:
    global _schema_tokens
    with _TOOLS_LOCK:
        if _schema_tokens is None:
            from astraea.tools import TOOLS
            from astraea.tools.context import context_tools
            from astraea.tool_runtime import ToolRuntime
            builtin = _measure_tool_schemas([*TOOLS, *context_tools("0" * 32),
                                             *ToolRuntime(None).tools()])
            extra = MCP_SCHEMA_RESERVE if _discovered_tools is None else _measure_tool_schemas(_discovered_tools)
            _schema_tokens = builtin + extra
        return _schema_tokens


def usage_tokens(messages: list) -> int:
    return estimate_tokens(messages) + tool_schema_tokens()


def _units(messages: list) -> list[tuple[str, int, list]]:
    """Keep an AI tool call and all matching results together."""
    units = []
    i = 0
    while i < len(messages):
        message = messages[i]
        calls = getattr(message, "tool_calls", None) if isinstance(message, AIMessage) else None
        if calls:
            results = messages[i + 1:i + 1 + len(calls)]
            if (len(results) == len(calls) and all(isinstance(r, ToolMessage) for r in results)
                    and {r.tool_call_id for r in results} == {c.get("id") for c in calls}):
                units.append(("tool", i, [message, *results]))
                i += 1 + len(calls)
                continue
        units.append(("dialogue", i, [message]))
        i += 1
    return units


def _trim(text: str, limit: int) -> str:
    encoded = ENCODING.encode(text)
    return ENCODING.decode(encoded[:max(0, limit)]) if len(encoded) > limit else text


class ContextManager:
    def __init__(self, compactor_model, context_window: int):
        self.compactor_model = compactor_model
        self.context_window = context_window
        self.session_id = uuid4().hex
        self.history = []
        self.active = []
        self.summary = ""
        self.summary_source_count = 0
        self.tier = 0  # Legacy UI field: 1 means a checkpoint exists.
        self.last_budget = {}
        self._profile_message = None
        self._last_visible = []

    @property
    def store(self) -> ContextStore:
        return ContextStore(self.session_id)

    @classmethod
    def resume(cls, compactor_model, context_window: int, session_id: str):
        data = load_session(session_id)
        obj = cls(compactor_model, context_window)
        obj.session_id = session_id
        obj.history = data["messages"]
        obj.active = data["active"]
        obj.summary = data.get("summary", "")
        obj.summary_source_count = data.get("summary_source_count", 0)
        obj.tier = data.get("tier", int(bool(obj.summary)))
        obj.last_budget = data.get("last_budget", {})
        return obj

    def prepare(self, user_input: str, profile_markdown: str = "") -> list:
        if not user_input.strip():
            raise ValueError("The user message is empty.")
        self._profile_message = (SystemMessage(content=PROFILE_PREFIX + profile_markdown)
                                 if profile_markdown.strip() else None)
        return [*([self._profile_message] if self._profile_message else []),
                *self.history, HumanMessage(content=user_input)]

    def record(self, messages: list) -> None:
        first_new = len(self.history) + int(self._profile_message is not None)
        fresh = messages[first_new:]
        self.history.extend(m for m in fresh if not self._is_harness(m))
        self.active = list(self._last_visible or messages)
        if fresh and isinstance(fresh[-1], AIMessage) and (not self.active or self.active[-1] is not fresh[-1]):
            self.active.append(fresh[-1])
        self._save_task_state(fresh)
        self._save()

    async def select(self, messages: list, system_message, tools: list) -> list:
        """Create a bounded view without modifying the agent's full message state."""
        profile = [m for m in messages if self._is_profile(m)]
        if profile:
            profile_cap = min(2048, max(128, int(self.context_window * .10)))
            if estimate_tokens(profile) > profile_cap:
                text = str(profile[0].content)
                half = max(20, (profile_cap - 50) // 2)
                profile = [SystemMessage(content=_trim(text, half) +
                                         "\n[Middle of profile omitted from this request]\n" +
                                         ENCODING.decode(ENCODING.encode(text)[-half:]))]
        ordinary = [m for m in messages if not self._is_profile(m)]
        current = next((m for m in reversed(ordinary) if isinstance(m, HumanMessage)), None)
        if current is None:
            raise ValueError("No user request in model state.")
        units = _units(ordinary)
        current_index = next(i for i in range(len(units) - 1, -1, -1)
                             if any(m is current for m in units[i][2]))
        query = str(current.content)
        for _, _, group in reversed(units[current_index + 1:]):
            opener = group[0]
            if isinstance(opener, AIMessage) and str(opener.content).strip():
                query += " " + str(opener.content)[:500]
                break

        schema = _measure_tool_schemas(tools)
        system_cost = estimate_tokens([system_message]) if system_message else 0
        profile_cost = estimate_tokens(profile)
        current_cost = estimate_tokens([current])
        output_reserve = min(4096, max(256, int(self.context_window * .15)))
        input_limit = self.context_window - output_reserve - max(64, int(self.context_window * .02))
        fixed = schema + system_cost + profile_cost + current_cost
        if fixed >= input_limit:
            raise ValueError(f"Context window too small for prompt, tools, profile and question "
                             f"({fixed} tokens; {input_limit} available). Increase ASTRAEA_CONTEXT_WINDOW.")
        variable = input_limit - fixed
        history_budget = int(variable * .46)
        tool_budget = int(variable * .34)
        retrieval_budget = int(variable * .15)
        state_budget = variable - history_budget - tool_budget - retrieval_budget

        tool_units = [u for u in units if u[0] == "tool"]
        stored_ids = {}
        for _, index, group in tool_units:
            ids = []
            for call in group[0].tool_calls:
                result = next((m for m in group[1:] if m.tool_call_id == call.get("id")), None)
                if result is not None:
                    ids.append(self.store.put_result(call, result))
            stored_ids[index] = ids

        dialogue = [u for u in units if u[0] == "dialogue" and not any(m is current for m in u[2])]
        await self._checkpoint(dialogue, history_budget)
        checkpoint = (SystemMessage(content=SUMMARY_PREFIX + _trim(self.summary, max(100, history_budget // 4)))
                      if self.summary else None)
        dialogue_limit = max(0, history_budget - (estimate_tokens([checkpoint]) if checkpoint else 0))
        chosen = {units[current_index][1]}
        dialogue_used = 0
        recent = dialogue[-4:]
        ranked = [*reversed(recent), *sorted(dialogue[:-4],
                    key=lambda u: (relevance(query, _message_payload(u[2][0])), u[1]), reverse=True)]
        for _, index, group in ranked:
            cost = estimate_tokens(group)
            if cost <= dialogue_limit - dialogue_used:
                chosen.add(index)
                dialogue_used += cost

        def score(unit):
            return relevance(query, " ".join(_message_payload(m)[:2500] for m in unit[2]))

        ordinary_tools, retrievals = [], []
        for unit in tool_units:
            names = {call.get("name") for call in unit[2][0].tool_calls}
            (retrievals if names & {"search_memory", "search_context", "load_context",
                                   "search_history", "load_history"} else ordinary_tools).append(unit)
        replacement = {}
        group_counts = 0
        used_tools = used_retrieval = 0
        for pool, allowance in ((ordinary_tools, tool_budget), (retrievals, retrieval_budget)):
            positives = [u for u in pool if score(u) > 0]
            candidates = (sorted(positives, key=lambda u: (score(u), u[1]), reverse=True)
                          if positives else list(reversed(pool[-1:])))
            used = 0
            for _, index, group in candidates[:4]:
                remaining = allowance - used
                if remaining < 100:
                    break
                fitted = self._fit_group(group, stored_ids.get(index, []), remaining)
                cost = estimate_tokens(fitted)
                if cost <= remaining:
                    chosen.add(index)
                    replacement[index] = fitted
                    used += cost
                    group_counts += 1
            if pool is ordinary_tools:
                used_tools = used
            else:
                used_retrieval = used

        omitted = len(tool_units) - group_counts
        manifest = SystemMessage(content=f"{omitted} historical tool groups are external. {MANIFEST}") if omitted else None
        state = self.store.state()
        state_text = (f"Goal: {state.get('current_goal', '')[:450]}\n"
                      f"Last response: {state.get('last_response', '')[:300]}\n"
                      f"Recent files: {', '.join(state.get('recent_files', [])[-8:])}") if state else ""
        state_message = (SystemMessage(content=STATE_PREFIX + _trim(state_text, max(0, state_budget - 80)))
                         if state_text and state_budget > 80 else None)
        visible = [*profile, *[m for m in (state_message, checkpoint, manifest) if m is not None]]
        for _, index, group in units:
            if index in chosen:
                visible.extend(replacement.get(index, group))
        total = schema + system_cost + estimate_tokens(visible)
        if total > input_limit:
            for optional in (manifest, state_message, checkpoint):
                if optional in visible:
                    visible.remove(optional)
                    total = schema + system_cost + estimate_tokens(visible)
                    if total <= input_limit:
                        break
        if total > input_limit:
            raise ValueError("Model request exceeds its token budget; shorten the question or increase ASTRAEA_CONTEXT_WINDOW.")
        self._last_visible = self.active = visible
        self.last_budget = {"system": system_cost, "tools": schema, "profile": profile_cost,
                            "current": current_cost, "dialogue": dialogue_used + (estimate_tokens([checkpoint]) if checkpoint in visible else 0),
                            "tool_results": used_tools, "retrieval": used_retrieval,
                            "state": sum(estimate_tokens([m]) for m in (state_message, manifest) if m in visible),
                            "dialogue_limit": history_budget, "tool_result_limit": tool_budget,
                            "retrieval_limit": retrieval_budget, "state_limit": state_budget,
                            "output_reserved": output_reserve, "input_total": total,
                            "input_limit": input_limit, "omitted_tool_groups": omitted}
        return visible

    def _fit_group(self, group: list, ids: list[str], budget: int) -> list:
        runtime = getattr(self, "_tool_runtime", None)
        if runtime is not None:
            group = [group[0], *(runtime.compress_result(result, ids[i] if i < len(ids) else None)
                                  for i, result in enumerate(group[1:]))]
        if estimate_tokens(group) <= budget:
            return group
        opener = group[0]
        calls = []
        for call in opener.tool_calls:
            shortened = dict(call)
            args = dict(call.get("args") or {})
            for key, value in args.items():
                if len(str(value)) > 200:
                    args[key] = str(value)[:120] + "… [full arguments stored externally]"
            shortened["args"] = args
            calls.append(shortened)
        opener = opener.model_copy(update={"tool_calls": calls, "content": str(opener.content)[:300]})
        per_result = max(20, (budget - estimate_tokens([opener]) - 40 * (len(group) - 1)) // max(1, len(group) - 1))
        results = []
        for i, result in enumerate(group[1:]):
            content = _trim(str(result.content), per_result)
            if content != str(result.content):
                content += f"\n[More in external context {ids[i] if i < len(ids) else 'search_context'}]"
            results.append(result.model_copy(update={"content": content}))
        return [opener, *results]

    async def _checkpoint(self, dialogue: list, budget: int) -> None:
        older = dialogue[:-6]
        if (not older or estimate_tokens([m for _, _, group in dialogue for m in group]) <= budget
                or len(older) - self.summary_source_count < (1 if not self.summary else 4)):
            return
        fresh = older[self.summary_source_count:]
        source = "\n".join(f"{m.type}: {_trim(str(m.content), 700)}" for _, _, group in fresh for m in group)
        prompt = ("Update a compact task checkpoint. Preserve goals, constraints, decisions, completed work, "
                  "unresolved questions and exact file names. Avoid tool output and invented facts. "
                  "Use at most 250 words.\nPrevious checkpoint:\n" + self.summary[:2000] +
                  "\nNew older dialogue:\n" + source[-10000:])
        try:
            result = await self.compactor_model.ainvoke(prompt)
            text = str(result.content).strip()
        except Exception as exc:
            print(f"[Context] Checkpoint unavailable: {type(exc).__name__}: {exc}")
            return
        if text:
            self.summary = text[:2500]
            self.summary_source_count = len(older)
            self.tier = 1
            self._save()

    def _save_task_state(self, fresh: list) -> None:
        goal = next((str(m.content) for m in reversed(self.history) if isinstance(m, HumanMessage)), "")
        answer = next((str(m.content) for m in reversed(self.history) if isinstance(m, AIMessage) and not m.tool_calls), "")
        files = list(self.store.state().get("recent_files", []))
        for message in fresh:
            for call in getattr(message, "tool_calls", None) or []:
                args = call.get("args") or {}
                files.extend(args[key][:300] for key in ("path", "source", "destination")
                             if isinstance(args.get(key), str))
        self.store.save_state(goal=goal, last_response=answer, files=list(dict.fromkeys(files))[-20:])

    def _save(self) -> None:
        save_session(self.session_id, self.history, self.summary, self.tier, self.active,
                     summary_source_count=self.summary_source_count, last_budget=self.last_budget)

    @staticmethod
    def _is_profile(message) -> bool:
        return isinstance(message, SystemMessage) and str(message.content).startswith(PROFILE_PREFIX)

    @classmethod
    def _is_harness(cls, message) -> bool:
        return isinstance(message, SystemMessage) and (cls._is_profile(message) or
            str(message.content).startswith((SUMMARY_PREFIX, STATE_PREFIX)))


class ContextMiddleware(AgentMiddleware):
    """Trim the model request, not the graph state or saved conversation."""

    def __init__(self, context: ContextManager):
        self.context = context

    async def awrap_model_call(self, request, handler):
        visible = await self.context.select(request.messages, request.system_message, request.tools)
        return await handler(request.override(messages=visible))

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if isinstance(result, ToolMessage):
            try:
                self.context.store.put_result(request.tool_call, result)
            except (OSError, TypeError, ValueError) as exc:
                print(f"[Context] Could not store tool result: {exc}")
        return result
