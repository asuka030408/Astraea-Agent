"""Web entry points without model requests."""

import asyncio
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from astraea.web import app, sessions
from astraea.web import ChatRequest, Session, chat_stream
from astraea.context import ContextManager
from astraea.storage import archive_fragment, save_session
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class WebTests(unittest.TestCase):
    def test_stream_returns_tokens_and_saves_answer(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            sessions.clear()
            context = ContextManager(object(), 64000)
            sessions[context.session_id] = Session(context)
            settings = type("Settings", (), {"context_window": 64000, "workspace": "test-workspace"})()

            async def fake_invoke(_model, _settings, messages, callbacks=None):
                await callbacks[0].on_llm_new_token("你好")
                return {"messages": [*messages, AIMessage(content="你好")]}

            with patch("astraea.web.runtime", return_value=(settings, object(), object())), \
                 patch("astraea.web.invoke_with_mcp", side_effect=fake_invoke), \
                 patch("astraea.web.build_model"), \
                 patch("astraea.web.MemoryManager"):
                client = TestClient(app)
                response = client.post("/api/chat/stream", json={"session_id": context.session_id, "message": "你好"})
                self.assertEqual(response.status_code, 200)
                events = [json.loads(line) for line in response.text.splitlines()]
                self.assertIn("token", [event["type"] for event in events])
                self.assertEqual(events[-1]["type"], "done")
                self.assertEqual([item.type for item in context.history], ["human", "ai"])
            sessions.clear()

    def test_stream_cancel_releases_session_lock(self):
        async def check():
            with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
                sessions.clear()
                context = ContextManager(object(), 64000)
                session = Session(context)
                sessions[context.session_id] = session
                settings = type("Settings", (), {"context_window": 64000, "workspace": "test-workspace"})()

                async def never_finishes(*_args, **_kwargs):
                    await asyncio.Event().wait()

                with patch("astraea.web.runtime", return_value=(settings, object(), object())), \
                     patch("astraea.web.invoke_with_mcp", side_effect=never_finishes):
                    response = await chat_stream(ChatRequest(session_id=context.session_id, message="等待"))
                    first = await response.body_iterator.__anext__()
                    self.assertIn('"type": "start"', first)
                    await response.body_iterator.aclose()
                    self.assertFalse(session.lock.locked())
                    self.assertEqual(context.history, [])
                sessions.clear()

        asyncio.run(check())

    def test_homepage_and_session(self):
        client = TestClient(app)
        self.assertEqual(client.get("/").status_code, 200)
        self.assertIn("Astraea", client.get("/").text)
        with patch("astraea.web.runtime") as mocked:
            settings = type("Settings", (), {"context_window": 64000, "workspace": "test-workspace"})()
            mocked.return_value = (settings, object(), object())
            response = client.post("/api/session")
            self.assertEqual(response.status_code, 200)
            self.assertIn("session_id", response.json())

    def test_history_survives_server_restart(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            sessions.clear()
            save_session("a" * 32, [HumanMessage(content="检查项目"), AIMessage(content="已检查")], "", 0)
            save_session("b" * 32, [], "", 0)
            settings = type("Settings", (), {"context_window": 64000, "workspace": "test-workspace"})()
            with patch("astraea.web.runtime", return_value=(settings, object(), object())):
                client = TestClient(app)
                listing = client.get("/api/sessions")
                self.assertEqual(listing.status_code, 200)
                self.assertEqual(listing.json()["sessions"][0]["title"], "检查项目")
                self.assertEqual(len(listing.json()["sessions"]), 1)
                detail = client.get("/api/sessions/" + "a" * 32)
                self.assertEqual(detail.status_code, 200)
                self.assertEqual([item["content"] for item in detail.json()["messages"]], ["检查项目", "已检查"])
                self.assertEqual([item.content for item in sessions["a" * 32].context.active], ["检查项目", "已检查"])
            sessions.clear()

    def test_context_and_archives_are_scoped_to_one_session(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            sessions.clear()
            first, second = "c" * 32, "d" * 32
            save_session(first, [HumanMessage(content="旧问题"), AIMessage(content="旧回答")], "摘要", 4,
                         [SystemMessage(content="Previous conversation summary:\n摘要"), HumanMessage(content="新问题")])
            save_session(second, [HumanMessage(content="另一个对话")], "", 0)
            archive_fragment(first, "summarized_history", "第一场对话的归档")
            archive_fragment(second, "tool_output", "第二场对话的归档")
            settings = type("Settings", (), {"context_window": 64000, "workspace": "test-workspace"})()
            with patch("astraea.web.runtime", return_value=(settings, object(), object())):
                client = TestClient(app)
                context = client.get(f"/api/sessions/{first}/context").json()
                self.assertEqual(context["tier"], 4)
                self.assertEqual(context["summary"], "摘要")
                self.assertEqual([item["content"] for item in context["messages"]],
                                 ["Previous conversation summary:\n摘要", "新问题"])
                archives = context["archives"]
                self.assertEqual([item["text"] for item in archives], ["第一场对话的归档"])
                self.assertEqual(client.get("/api/sessions/" + "x" * 32 + "/context").status_code, 404)
            sessions.clear()


if __name__ == "__main__":
    unittest.main()
