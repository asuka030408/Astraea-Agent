"""Focused checks for Astraea's event-aware compaction policy."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from astraea.context import ContextManager, estimate_tokens, tool_schema_tokens, usage_tokens, _compaction_boundary
from astraea.storage import archive_fragment


class FakeModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        return AIMessage(content="1. Current goal: continue.\n2. User constraints: keep the check word.")


class ContextTests(unittest.TestCase):
    def test_t1_uses_no_model_and_preserves_full_history(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            model = FakeModel()
            history = [
                HumanMessage(content="early task"),
                AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "call-1"}]),
                ToolMessage(content="old output " * 250, tool_call_id="call-1"),
                AIMessage(content="saw file"),
                HumanMessage(content="second turn"), AIMessage(content="yes"),
                HumanMessage(content="third turn"), AIMessage(content="yes"),
            ]
            budget = int(usage_tokens([*history, HumanMessage(content="next question")]) / 0.35)
            context = ContextManager(model, budget)
            context.record(history)
            request = context.prepare("next question")
            self.assertEqual(context.tier, 1)
            self.assertEqual(model.calls, 0)
            self.assertIn("archived", request[2].content)
            self.assertEqual(len(context.history), len(history))
            saved = json.loads((Path(directory) / "sessions" / f"{context.session_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["messages"]), len(history))

    def test_t2_archives_complete_failed_tool_pair(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            context = ContextManager(FakeModel(), 1000)
            context.active = [
                HumanMessage(content="first"),
                AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "failed"}]),
                ToolMessage(content="permission denied", tool_call_id="failed", status="error"),
                AIMessage(content="retry later"),
                HumanMessage(content="second"), AIMessage(content="ok"),
                HumanMessage(content="third"), AIMessage(content="ok"),
            ]
            context._previous_active = list(context.active)
            context._archive_failed_attempts(max_attempts=2)
            self.assertFalse(any(isinstance(m, ToolMessage) for m in context.active))
            self.assertTrue((Path(directory) / "archives" / f"{context.session_id}.jsonl").exists())

    def test_t4_summary_enters_next_request(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            model = FakeModel()
            history = [
                HumanMessage(content="check word: purple comet " * 100), AIMessage(content="noted " * 100),
                HumanMessage(content="second"), AIMessage(content="ok"),
                HumanMessage(content="third"), AIMessage(content="ok"),
            ]
            question = "continue"
            context = ContextManager(model, int(usage_tokens([*history, HumanMessage(content=question)]) / 0.91))
            context.record(history)
            request = context.prepare(question)
            self.assertEqual(context.tier, 4)
            self.assertEqual(model.calls, 1)
            self.assertIsInstance(request[0], SystemMessage)
            self.assertIn("Previous conversation summary", request[0].content)

    def test_t5_retrieves_archived_context(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            context = ContextManager(FakeModel(), tool_schema_tokens() + 3000)
            archive_fragment(context.session_id, "tool_output", "最开始的校验词是紫色彗星")
            request = context.prepare("最开始的校验词是什么？")
            self.assertIn("紫色彗星", request[0].content)

    def test_estimate_tokens_counts_tool_call_arguments(self):
        empty = AIMessage(content="")
        big = AIMessage(content="", tool_calls=[{
            "name": "write_file", "args": {"path": "a.py", "content": "y" * 8000}, "id": "c1"}])
        self.assertLess(estimate_tokens([empty]), 30)
        self.assertGreater(estimate_tokens([big]), 50 * estimate_tokens([empty]))

    def test_usage_budget_includes_tool_schemas(self):
        schema = tool_schema_tokens()
        self.assertGreater(schema, 500)
        self.assertEqual(usage_tokens([]), schema)
        messages = [HumanMessage(content="你好")]
        self.assertEqual(usage_tokens(messages), estimate_tokens(messages) + schema)

    def test_tool_groups_give_a_boundary_inside_one_long_turn(self):
        messages = [HumanMessage(content="读一堆文件")]
        for number in range(5):
            messages.append(AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": f"c{number}"}]))
            messages.append(ToolMessage(content="x" * 100, tool_call_id=f"c{number}"))
        boundary = _compaction_boundary(messages)
        self.assertEqual(boundary, 7)
        self.assertTrue(getattr(messages[boundary], "tool_calls", None))
        self.assertIsInstance(messages[boundary + 1], ToolMessage)
        self.assertEqual(_compaction_boundary(messages, keep_tool_groups=1), 9)

    def test_long_single_turn_compacts_and_stays_usable(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            model = FakeModel()
            history = [HumanMessage(content="把这些文件都读一遍并总结")]
            for number in range(12):
                history.append(AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": f"f{number}"}, "id": f"call-{number}"}]))
                history.append(ToolMessage(content="x" * 6000, tool_call_id=f"call-{number}"))
            history.append(AIMessage(content="总结完毕"))
            window = int(usage_tokens([*history, HumanMessage(content="继续")]) / 1.02)
            context = ContextManager(model, window)
            context.record(history)
            request = context.prepare("继续")
            self.assertLess(usage_tokens(request), window * 0.95)
            self.assertEqual(len(context.history), len(history))
            self.assertTrue((Path(directory) / "archives" / f"{context.session_id}.jsonl").is_file())
            context.record([*request, AIMessage(content="继续完成")])
            self.assertIsInstance(context.prepare("再试一次"), list)

    def test_profile_is_injected_but_not_saved_as_chat_history(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            context = ContextManager(FakeModel(), 64000)
            request = context.prepare("你好", "# 用户习惯\n\n- 喜欢简洁回答")
            self.assertIsInstance(request[0], SystemMessage)
            self.assertIn("喜欢简洁回答", request[0].content)
            context.record([*request, AIMessage(content="你好")])
            self.assertEqual([item.type for item in context.history], ["human", "ai"])
            self.assertEqual([item.type for item in context.active], ["human", "ai"])
            next_request = context.prepare("继续", "# 用户习惯\n\n- 喜欢详细回答")
            self.assertIn("喜欢详细回答", next_request[0].content)
            self.assertNotIn("喜欢简洁回答", str(next_request))


if __name__ == "__main__":
    unittest.main()
