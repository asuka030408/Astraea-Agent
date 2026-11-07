"""Event-aware context management: script first, model summary near capacity."""

import json
from threading import Lock
from uuid import uuid4

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from astraea.storage import archive_fragment, load_session, retrieve_fragments, save_session


ENCODING = tiktoken.get_encoding("cl100k_base")
THRESHOLDS = {1: 0.30, 2: 0.60, 3: 0.75, 4: 0.90, 5: 0.95}
SUMMARY_PREFIX = "Previous conversation summary:\n"
RETRIEVAL_PREFIX = "Retrieved archived context:\n"
PROFILE_PREFIX = "Long-term user profile (may be outdated; the current user message takes precedence):\n"
NINE_SECTIONS = (
    "1. Current goal\n2. User constraints\n3. Decisions and reasons\n"
    "4. Completed work\n5. Current state\n6. Failed attempts and lessons\n"
    "7. Relevant files or artifacts\n8. Open questions or risks\n9. Next action"
)

# Tool definitions travel with every request. MCP tools are discovered per
# turn, so reserve a conservative allowance until their exact schemas are known.
# The 20 vendored MCP tools measure about 3750 tokens with this encoder.
MCP_SCHEMA_RESERVE = 3800
_TOOLS_LOCK = Lock()
_discovered_tools = None
_schema_tokens = None


def _message_payload(message) -> str:
    """Everything from one message that the provider actually receives."""
    parts = [str(message.content)]
    for call in getattr(message, "tool_calls", None) or []:
        parts.append(json.dumps(call, ensure_ascii=False, default=str))
    for attribute in ("name", "tool_call_id"):
        value = getattr(message, attribute, None)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def estimate_tokens(messages: list) -> int:
    """Approximate model tokens; provider counts may differ."""
    return sum(len(ENCODING.encode(_message_payload(message))) + 12 for message in messages)


def _measure_tool_schemas(tools) -> int:
    from langchain_core.utils.function_calling import convert_to_openai_tool
    total = 0
    for item in tools:
        try:
            schema = convert_to_openai_tool(item)
            total += len(ENCODING.encode(json.dumps(schema, ensure_ascii=False, default=str))) + 8
        except Exception:
            total += 150
    return total


def set_discovered_tools(tools) -> None:
    """Record the MCP tools found this turn so the budget uses exact schemas."""
    global _discovered_tools, _schema_tokens
    with _TOOLS_LOCK:
        _discovered_tools = list(tools or [])
        _schema_tokens = None


def tool_schema_tokens() -> int:
    """Token cost of the tool definitions attached to every request."""
    global _schema_tokens
    with _TOOLS_LOCK:
        if _schema_tokens is None:
            from astraea.tools import TOOLS
            builtin = _measure_tool_schemas(TOOLS)
            extra = (MCP_SCHEMA_RESERVE if _discovered_tools is None
                     else _measure_tool_schemas(_discovered_tools))
            _schema_tokens = builtin + extra
        return _schema_tokens


def usage_tokens(messages: list) -> int:
    """Full request size: messages plus the tool definitions that go with them."""
    return estimate_tokens(messages) + tool_schema_tokens()


def _tool_group_starts(messages: list) -> list[int]:
    """Index of each AIMessage that opens a tool call/result round-trip."""
    return [index for index, message in enumerate(messages)
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None)]


def _compaction_boundary(messages: list, keep_turns: int = 2, keep_tool_groups: int = 2) -> int:
    """Allow compaction inside one long turn, not just across turns.

    A single question with many tool calls has no earlier turns, so a turn-only
    boundary is always zero and every tier has nothing to compact. Tool
    round-trips provide a second boundary. Both candidates land on a
    HumanMessage or on the AIMessage opening a tool group, so a tool call is
    never separated from its results.
    """
    turns = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
    turn_boundary = turns[-keep_turns] if len(turns) > keep_turns else 0
    groups = _tool_group_starts(messages)
    tool_boundary = groups[-keep_tool_groups] if len(groups) > keep_tool_groups else 0
    return max(turn_boundary, tool_boundary)


def _message_text(message) -> str:
    return f"{message.type}: {message.content}"


class ContextManager:
    def __init__(self, compactor_model, context_window: int):
        self.compactor_model = compactor_model
        self.context_window = context_window
        self.session_id = uuid4().hex
        self.history = []
        self.active = []
        self.summary = ""
        self.tier = 0
        self._previous_active = []
        self._profile_message = None

    @classmethod
    def resume(cls, compactor_model, context_window: int, session_id: str):
        data = load_session(session_id)
        context = cls(compactor_model, context_window)
        context.session_id = session_id
        context.history = data["messages"]
        context.active = data["active"]
        context._previous_active = list(context.active)
        context.summary = data.get("summary", "")
        context.tier = data.get("tier", 0)
        return context

    def record(self, messages: list) -> None:
        """Save a completed turn without compacting it yet."""
        messages = [message for message in messages if not self._is_profile(message)]
        added = messages[len(self._previous_active):]
        self.history.extend(message for message in added if not self._is_harness_message(message))
        self.active = list(messages)
        self._previous_active = list(messages)
        self._profile_message = None
        self._save()

    def prepare(self, user_input: str, profile_markdown: str = "") -> list:
        """Apply pressure thresholds before the model receives the new question."""
        self._profile_message = (SystemMessage(content=PROFILE_PREFIX + profile_markdown)
                                 if profile_markdown.strip() else None)
        self._apply_thresholds(user_input)
        request = self._request(user_input)
        if self._usage(user_input) >= THRESHOLDS[5] or self._asks_about_history(user_input):
            self._retrieve(user_input)
            request = self._request(user_input)
        if usage_tokens(request) >= self.context_window * 0.95:
            raise ValueError(
                "The active context is too large; shorten the question or increase "
                "ASTRAEA_CONTEXT_WINDOW. Tool definitions are included in this budget."
            )
        return request

    def _request(self, user_input: str) -> list:
        return [*([self._profile_message] if self._profile_message else []),
                *self.active, HumanMessage(content=user_input)]

    def _usage(self, user_input: str) -> float:
        return usage_tokens(self._request(user_input)) / self.context_window

    def _apply_thresholds(self, user_input: str) -> None:
        summarized_now = False
        while self.tier < 4 and self._usage(user_input) >= THRESHOLDS[self.tier + 1]:
            next_tier = self.tier + 1
            before = self._usage(user_input)
            if next_tier == 1:
                self._micro_compact()
            elif next_tier == 2:
                self._archive_failed_attempts(max_attempts=2)
            elif next_tier == 3:
                self._archive_failed_attempts(max_attempts=None)
            else:
                self._model_summary()
                summarized_now = True
            self.tier = next_tier
            self._save()
            print(f"[Context] T{next_tier}: {before:.0%} before, {self._usage(user_input):.0%} after")
        if self.tier == 4 and not summarized_now and self._usage(user_input) >= THRESHOLDS[4]:
            self._model_summary()
            self._save()

    def _micro_compact(self) -> None:
        """T1: shorten only old tool output; no model request."""
        # T1 only shortens text, so it can reach one group further than the
        # structural tiers without breaking any tool call/result pair.
        boundary = _compaction_boundary(self.active, keep_tool_groups=1)
        changed = list(self.active)
        for index, message in enumerate(changed[:boundary]):
            if isinstance(message, ToolMessage) and len(str(message.content)) > 600:
                archive_fragment(self.session_id, "tool_output", _message_text(message))
                changed[index] = message.model_copy(update={"content": str(message.content)[:400] + "\n[Older tool output archived]"})
        self._set_active(changed)

    def _archive_failed_attempts(self, max_attempts: int | None) -> None:
        """T2/T3: move older failed tool-call/result pairs out of active context."""
        boundary = _compaction_boundary(self.active)
        changed = []
        removed = 0
        index = 0
        while index < len(self.active):
            message = self.active[index]
            calls = getattr(message, "tool_calls", None) if isinstance(message, AIMessage) else None
            if index < boundary and calls and (max_attempts is None or removed < max_attempts):
                count = len(calls)
                results = self.active[index + 1:index + 1 + count]
                ids = {call.get("id") for call in calls}
                if (len(results) == count and all(isinstance(item, ToolMessage) for item in results)
                        and {item.tool_call_id for item in results} == ids
                        and all(getattr(item, "status", "success") == "error" for item in results)):
                    archive_fragment(self.session_id, "failed_attempt", "\n".join(_message_text(item) for item in [message, *results]))
                    changed.append(AIMessage(content="[Earlier failed tool attempt archived; details can be retrieved if needed.]"))
                    removed += 1
                    index += count + 1
                    continue
            changed.append(message)
            index += 1
        self._set_active(changed)

    def _model_summary(self) -> None:
        """T4: use model B to write a nine-section checkpoint."""
        boundary = _compaction_boundary(self.active)
        if boundary == 0:
            return
        older = self.active[:boundary]
        prompt = (
            "Write a concise checkpoint in exactly these nine sections:\n" + NINE_SECTIONS +
            "\nPreserve exact user constraints and uncertainty. Do not invent facts. "
            "Mention archived failed attempts without claiming they were resolved.\n\n" +
            "\n".join(_message_text(message)[:4000] for message in older)
        )
        result = str(self.compactor_model.invoke(prompt).content).strip()
        if not result:
            return
        archive_fragment(self.session_id, "summarized_history", "\n".join(_message_text(message) for message in older))
        self.summary = result
        self._set_active([SystemMessage(content=SUMMARY_PREFIX + result), *self.active[boundary:]])

    def _retrieve(self, user_input: str) -> None:
        fragments = retrieve_fragments(self.session_id, user_input)
        if not fragments:
            return
        base = [message for message in self.active if not self._is_retrieval(message)]
        retrieved = SystemMessage(content=RETRIEVAL_PREFIX + "\n\n".join(fragments))
        candidate = [retrieved, *base, HumanMessage(content=user_input)]
        if usage_tokens(candidate) < self.context_window * 0.95:
            self._set_active(candidate[:-1])
            print("[Context] T5: restored relevant archived text")

    def _set_active(self, messages: list) -> None:
        self.active = messages
        self._previous_active = list(messages)

    def _save(self) -> None:
        save_session(self.session_id, self.history, self.summary, self.tier, self.active)

    @staticmethod
    def _is_retrieval(message) -> bool:
        return isinstance(message, SystemMessage) and str(message.content).startswith(RETRIEVAL_PREFIX)

    @staticmethod
    def _is_profile(message) -> bool:
        return isinstance(message, SystemMessage) and str(message.content).startswith(PROFILE_PREFIX)

    @classmethod
    def _is_harness_message(cls, message) -> bool:
        return isinstance(message, SystemMessage) and (
            str(message.content).startswith(SUMMARY_PREFIX) or cls._is_retrieval(message) or cls._is_profile(message)
        )

    @staticmethod
    def _asks_about_history(text: str) -> bool:
        return any(word in text.lower() for word in ("之前", "最开始", "刚才", "上次", "以前", "remember", "earlier", "previous"))
