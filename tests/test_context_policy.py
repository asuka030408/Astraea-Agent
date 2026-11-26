"""Adaptive context selection and external storage checks."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from pydantic import Field

from astraea.agent import build_agent
from astraea.context import ContextManager, ContextMiddleware, _measure_tool_schemas, _units, estimate_tokens
from astraea.tools.context import context_tools


class FakeModel:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, _prompt):
        self.calls += 1
        return AIMessage(content="Goal: finish the task. Constraint: keep the original files.")


class RecordingChatModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)
    bound: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound.append(tools)
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


def tool_round(number: int, content: str) -> list:
    call_id = f"call-{number}"
    return [AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": f"file{number}.txt"}, "id": call_id}]),
            ToolMessage(content=content, tool_call_id=call_id, name="read_file")]


class ContextTests(unittest.TestCase):
    def test_real_agent_loop_keeps_full_tool_round_trip(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory, \
                 patch("astraea.storage.DATA_DIR", Path(directory) / "data"), \
                 patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}):
                Path(directory, "a.txt").write_text("hello", encoding="utf-8")
                model = RecordingChatModel(responses=[
                    AIMessage(content="", tool_calls=[{"name": "list_files", "args": {"directory": "."}, "id": "c1"}]),
                    AIMessage(content="done")])
                context = ContextManager(model, 20000)
                result = await build_agent(model, context=context).ainvoke(
                    {"messages": context.prepare("List files", "User prefers concise answers")})
                context.record(result["messages"])
                self.assertEqual(len(model.seen), 2)
                self.assertTrue(all(len(bound) < 21 for bound in model.bound))
                self.assertEqual(context.last_budget["tools"], _measure_tool_schemas(model.bound[-1]))
                self.assertEqual([m.type for m in context.history], ["human", "ai", "tool", "ai"])
                self.assertTrue(any("User prefers concise" in str(m.content) for m in model.seen[0]))
                self.assertTrue(any(isinstance(m, ToolMessage) for m in model.seen[1]))
                self.assertEqual(len(list((Path(directory) / "data" / "context" / context.session_id / "items").glob("*.json"))), 1)

        asyncio.run(run())

    def test_twenty_calls_send_only_two_relevant_results(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            context = ContextManager(FakeModel(), 18000)
            transcript = [HumanMessage(content="Read the files")]
            for number in range(20):
                content = ("alpha facts " if number == 0 else "beta facts " if number == 1 else "unrelated data ") * 100
                transcript.extend(tool_round(number, content))
            context.record(transcript)
            request = context.prepare("Compare the alpha and beta facts")
            visible = asyncio.run(context.select(request, SystemMessage(content="Assistant"), []))
            kept = [m.tool_call_id for m in visible if isinstance(m, ToolMessage)]
            self.assertEqual(kept, ["call-0", "call-1"])
            self.assertEqual(context.last_budget["omitted_tool_groups"], 18)
            self.assertEqual(len(list((Path(directory) / "context" / context.session_id / "items").glob("*.json"))), 20)
            self.assertEqual(len(context.history), len(transcript))
            context.record([*request, AIMessage(content="Compared.")])
            self.assertEqual(len(context.history), len(transcript) + 2)

    def test_context_tools_are_session_scoped_and_load_full_result(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            first = ContextManager(FakeModel(), 18000)
            second = ContextManager(FakeModel(), 18000)
            item_id = first.store.put_result({"id": "c1", "name": "read_file", "args": {"path": "note.txt"}},
                                             ToolMessage(content="secret comet " * 1000, tool_call_id="c1"))
            search, load = context_tools(first.session_id)[:2]
            result = json.loads(search.invoke({"query": "comet"}))
            self.assertEqual(result[0]["id"], item_id)
            self.assertEqual(json.loads(load.invoke({"context_id": item_id, "max_chars": 100}))["total_chars"], 13000)
            self.assertTrue(context_tools(second.session_id)[1].invoke({"context_id": item_id}).startswith("ERROR:"))
            first.record([HumanMessage(content="The old decision was to use comet blue"),
                          AIMessage(content="Decision recorded")])
            second.record([HumanMessage(content="Another conversation")])
            history_search, history_load = context_tools(first.session_id)[2:]
            match = json.loads(history_search.invoke({"query": "comet blue"}))[0]
            self.assertIn("comet blue", json.loads(history_load.invoke({"index": match["index"]}))["content"])
            self.assertEqual(json.loads(context_tools(second.session_id)[2].invoke({"query": "comet blue"})), [])

    def test_budget_reserves_output_and_separates_retrieval(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            from astraea.tools import TOOLS
            context = ContextManager(FakeModel(), 22000)
            transcript = [HumanMessage(content="Recall the issue"),
                          AIMessage(content="", tool_calls=[{"name": "search_memory", "args": {"query": "issue"}, "id": "m1"}]),
                          ToolMessage(content="issue detail " * 2000, tool_call_id="m1")]
            context.record(transcript)
            visible = asyncio.run(context.select(context.prepare("Recall the issue", "User likes concise answers"),
                                                 SystemMessage(content="Assistant"), TOOLS))
            budget = context.last_budget
            self.assertGreater(budget["output_reserved"], 0)
            self.assertLessEqual(budget["input_total"], budget["input_limit"])
            self.assertGreater(budget["retrieval"], 0)
            self.assertTrue(any("User likes concise" in str(m.content) for m in visible))

    def test_large_profile_cannot_consume_the_whole_request(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            context = ContextManager(FakeModel(), 16000)
            request = context.prepare("Continue the task", "User preference " * 10000)
            visible = asyncio.run(context.select(request, SystemMessage(content="Assistant"), []))
            profile = [m for m in visible if isinstance(m, SystemMessage) and "Long-term user profile" in str(m.content)]
            self.assertEqual(len(profile), 1)
            self.assertLess(estimate_tokens(profile), 2200)
            self.assertLess(context.last_budget["input_total"], context.last_budget["input_limit"])
            context.record([*request, AIMessage(content="Done")])
            self.assertEqual([m.type for m in context.history], ["human", "ai"])

    def test_checkpoint_only_when_dialogue_budget_needs_it(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            model = FakeModel()
            context = ContextManager(model, 12000)
            transcript = []
            for number in range(10):
                transcript.extend([HumanMessage(content=f"Constraint {number}: " + "keep facts " * 500),
                                   AIMessage(content="Acknowledged " * 200)])
            context.record(transcript)
            request = context.prepare("Continue")
            first = asyncio.run(context.select(request, SystemMessage(content="Assistant"), []))
            self.assertEqual(model.calls, 1)
            self.assertTrue(any(str(m.content).startswith("Task checkpoint") for m in first))
            asyncio.run(context.select(request, SystemMessage(content="Assistant"), []))
            self.assertEqual(model.calls, 1)
            resumed = ContextManager.resume(model, 12000, context.session_id)
            self.assertEqual(resumed.summary, context.summary)

    def test_middleware_does_not_mutate_full_agent_state(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            context = ContextManager(FakeModel(), 18000)
            messages = [HumanMessage(content="Use alpha")]
            for number in range(10):
                messages.extend(tool_round(number, "alpha" if number == 0 else "noise"))

            class Request:
                def __init__(self, messages):
                    self.messages = messages
                    self.system_message = SystemMessage(content="Assistant")
                    self.tools = []

                def override(self, **kwargs):
                    return Request(kwargs["messages"])

            seen = []

            async def handler(request):
                seen.append(request.messages)
                return AIMessage(content="done")

            asyncio.run(ContextMiddleware(context).awrap_model_call(Request(messages), handler))
            self.assertEqual(len(messages), 21)
            self.assertEqual([m.tool_call_id for m in seen[0] if isinstance(m, ToolMessage)], ["call-0"])
            self.assertLess(estimate_tokens(seen[0]), estimate_tokens(messages))
            self.assertEqual([m.tool_call_id for m in messages if isinstance(m, ToolMessage)],
                             [f"call-{i}" for i in range(10)])

    def test_tool_units_keep_parallel_calls_with_their_results(self):
        messages = [AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "a"},
                                                       {"name": "b", "args": {}, "id": "b"}]),
                    ToolMessage(content="one", tool_call_id="a"), ToolMessage(content="two", tool_call_id="b")]
        self.assertEqual(len(_units(messages)), 1)
        self.assertEqual(_units(messages)[0][0], "tool")


if __name__ == "__main__":
    unittest.main()
