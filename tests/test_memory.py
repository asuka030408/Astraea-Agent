"""Memory extraction and indexing without network requests."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from astraea.memory import MemoryManager
from astraea.tools.memory import search_memory
from unittest.mock import patch


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, prompt):
        return AIMessage(content=json.dumps(next(self.responses), ensure_ascii=False))


class MemoryTests(unittest.TestCase):
    def test_search_tool_returns_vector_matches(self):
        with patch("astraea.tools.memory.MemoryManager") as manager:
            manager.return_value.search_events.return_value = [{"text": "项目决定", "source_session": "a" * 32, "score": 0.9}]
            result = json.loads(search_memory.invoke({"query": "过去的决定", "top_k": 3}))
            self.assertEqual(result[0]["text"], "项目决定")
            manager.return_value.search_events.assert_called_once_with("过去的决定", limit=3)

    def test_manual_markdown_bullet_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = MemoryManager(FakeModel([{"profile_add": [], "profile_remove": [], "events_add": []}]),
                                    Path(directory), lambda texts: [[1.0] + [0.0] * 1023 for _ in texts])
            (Path(directory) / "profile.md").write_text("# 用户习惯\n\n- 喜欢简洁的回答\n", encoding="utf-8")
            manager.process_turn("e" * 32, [HumanMessage(content="你好")])
            self.assertIn("喜欢简洁的回答", (Path(directory) / "profile.md").read_text(encoding="utf-8"))

    def test_profile_and_events_use_two_markdown_files_and_one_vector_index(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel([
                {"profile_add": ["用户正在开发 Astraea Agent"], "profile_remove": [],
                 "events_add": ["用户决定每轮对话后整理 Memory"]},
                {"profile_add": ["用户正在开发 Astraea Agent"], "profile_remove": [],
                 "events_add": ["用户决定每轮对话后整理 Memory"]},
            ])
            calls = []

            def embed(texts):
                calls.extend(texts)
                return [[1.0] + [0.0] * 1023 for _ in texts]

            manager = MemoryManager(model, Path(directory), embed)
            messages = [HumanMessage(content="我在开发 Astraea Agent，每轮对话后整理 Memory"),
                        AIMessage(content="好的")]
            for _ in range(2):
                manager.process_turn("a" * 32, messages)
            profile = (Path(directory) / "profile.md").read_text(encoding="utf-8")
            events = (Path(directory) / "events.md").read_text(encoding="utf-8")
            self.assertEqual(profile.count("用户正在开发 Astraea Agent"), 1)
            self.assertEqual(events.count("用户决定每轮对话后整理 Memory"), 1)
            self.assertEqual(calls, ["用户决定每轮对话后整理 Memory"])
            with closing(sqlite3.connect(Path(directory) / "vectors.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM vectors").fetchone()[0], 1)
            self.assertEqual(manager.search_events("Memory 整理")[0]["text"], "用户决定每轮对话后整理 Memory")

    def test_markdown_survives_embedding_failure_and_reindexes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel([{"profile_add": [], "profile_remove": [], "events_add": ["重要事项"]}])
            manager = MemoryManager(model, Path(directory), lambda _texts: (_ for _ in ()).throw(RuntimeError("offline")))
            with self.assertRaises(RuntimeError):
                manager.process_turn("b" * 32, [HumanMessage(content="重要事项")])
            self.assertIn("重要事项", (Path(directory) / "events.md").read_text(encoding="utf-8"))
            manager.embedder = lambda texts: [[0.0] * 1024 for _ in texts]
            manager.sync_index()
            with closing(sqlite3.connect(Path(directory) / "vectors.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM vectors").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
