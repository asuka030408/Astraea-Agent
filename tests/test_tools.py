"""Tool behavior and workspace boundaries."""

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astraea.tools import (TOOLS, read_file, run_command, search_files, write_file,
                           list_files, file_info, read_file_lines, append_file,
                           replace_in_file, copy_file, move_file, delete_file)
from astraea.tools.web_search import search_web


class ToolTests(unittest.TestCase):
    def test_native_tool_catalog_exceeds_ten_and_file_operations_are_bounded(self):
        self.assertGreaterEqual(len(TOOLS), 11)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}):
            write_file.invoke({"path": "notes/a.txt", "content": "one\ntwo\n"})
            self.assertIn("notes/a.txt", list_files.invoke({"directory": "notes"}))
            self.assertEqual(read_file_lines.invoke({"path": "notes/a.txt", "start_line": 2}), "2: two")
            self.assertEqual(json.loads(file_info.invoke({"path": "notes/a.txt"}))["type"], "file")
            self.assertIn("Appended", append_file.invoke({"path": "notes/a.txt", "content": "three"}))
            self.assertIn("Replaced", replace_in_file.invoke({"path": "notes/a.txt", "old_text": "three", "new_text": "four"}))
            self.assertIn("Copied", copy_file.invoke({"source": "notes/a.txt", "destination": "notes/b.txt"}))
            self.assertIn("Moved", move_file.invoke({"source": "notes/b.txt", "destination": "notes/c.txt"}))
            result = delete_file.invoke({"path": "notes/c.txt"})
            self.assertIn("recoverable", result)
            self.assertFalse((Path(directory) / "notes" / "c.txt").exists())
            self.assertEqual(len(list((Path(directory) / ".astraea-trash").iterdir())), 1)
            self.assertTrue(copy_file.invoke({"source": "notes/a.txt", "destination": "../outside.txt"}).startswith("ERROR:"))

    def test_web_search_returns_source_urls(self):
        with patch("astraea.tools.web_search.DDGS") as search:
            search.return_value.text.return_value = [{"title": "Python Docs", "href": "https://docs.python.org/", "body": "Documentation"}]
            result = json.loads(search_web.invoke({"query": "Python docs", "max_results": 3}))
            self.assertEqual(result[0]["url"], "https://docs.python.org/")
            search.return_value.text.assert_called_once_with("Python docs", max_results=3)

    def test_write_read_search_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}):
            self.assertIn("Wrote", write_file.invoke({"path": "notes/example.txt", "content": "hello Astraea"}))
            self.assertEqual(read_file.invoke({"path": "notes/example.txt"}), "hello Astraea")
            self.assertIn("notes/example.txt:1", search_files.invoke({"query": "Astraea"}))
            self.assertTrue(read_file.invoke({"path": "../outside.txt"}).startswith("ERROR:"))
            self.assertTrue(write_file.invoke({"path": "../outside.txt", "content": "x"}).startswith("ERROR:"))
            outside = Path(directory).parent / "outside.txt"
            self.assertFalse(outside.exists())

    def test_command_rejects_shell_and_arbitrary_programs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}):
            self.assertTrue(run_command.invoke({"command": "powershell Remove-Item *"}).startswith("ERROR:"))
            self.assertTrue(run_command.invoke({"command": "git -C .. status"}).startswith("ERROR:"))


if __name__ == "__main__":
    unittest.main()
