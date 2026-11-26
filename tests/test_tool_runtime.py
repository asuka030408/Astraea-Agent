"""Tool routing, cache, plan and result fidelity checks."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from astraea.context import ContextManager, _measure_tool_schemas
from astraea.tool_runtime import READONLY, ToolOptimizationMiddleware, ToolRuntime, compact_schema
from astraea.tools import TOOLS


class ToolRuntimeTests(unittest.TestCase):
    def test_dynamic_catalog_and_explicit_loading_keep_execution_schema(self):
        with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
            context = ContextManager(None, 64000)
            runtime = ToolRuntime(context)
            all_tools = [*TOOLS, *runtime.tools()]
            runtime.bind_catalog(all_tools)
            visible = runtime.select_schemas([HumanMessage(content="请读取 note.txt")], all_tools)
            names = {item["function"]["name"] for item in visible}
            self.assertIn("read_file", names)
            self.assertIn("search_tools", names)
            self.assertNotIn("delete_file", names)
            runtime.tools()[1].invoke({"names": ["delete_file"]})
            loaded = runtime.select_schemas([HumanMessage(content="请读取 note.txt")], all_tools)
            self.assertIn("delete_file", {item["function"]["name"] for item in loaded})
            original = next(item for item in all_tools if item.name == "read_file")
            compact = compact_schema(original)
            self.assertEqual(compact["function"]["parameters"]["required"], ["path"])
            self.assertEqual(compact["function"]["parameters"]["properties"]["path"]["type"], "string")
            self.assertLess(_measure_tool_schemas(visible), _measure_tool_schemas(all_tools))

    def test_route_keeps_only_requested_mcp_family(self):
        @tool
        def time_get_current_time() -> str:
            """Get the current time."""
            return "12:00"

        @tool
        def git_git_status() -> str:
            """Show repository status."""
            return "clean"

        @tool
        def git_git_commit() -> str:
            """Commit changes."""
            return "committed"

        runtime = ToolRuntime(ContextManager(None, 64000))
        all_tools = [*TOOLS, *runtime.tools(), time_get_current_time, git_git_status, git_git_commit]
        runtime.bind_catalog(all_tools)
        time_names = {entry["function"]["name"] for entry in
                      runtime.select_schemas([HumanMessage(content="现在几点？")], all_tools)}
        self.assertIn("time_get_current_time", time_names)
        self.assertNotIn("git_git_status", time_names)
        git_names = {entry["function"]["name"] for entry in
                     runtime.select_schemas([HumanMessage(content="查看 git 状态")], all_tools)}
        self.assertIn("git_git_status", git_names)
        self.assertNotIn("git_git_commit", git_names)

    def test_cache_hit_and_file_change_invalidation(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}):
                target = Path(directory) / "note.txt"
                target.write_text("first", encoding="utf-8")
                runtime = ToolRuntime(ContextManager(None, 64000))
                calls = 0

                async def reader():
                    nonlocal calls
                    calls += 1
                    return target.read_text(encoding="utf-8")

                first = await runtime.cached_text("read_file", {"path": "note.txt"}, reader)
                second = await runtime.cached_text("read_file", {"path": "note.txt"}, reader)
                self.assertEqual((first, second, calls), ("first", "first", 1))
                target.write_text("second", encoding="utf-8")
                os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 2_000_000_000))
                third = await runtime.cached_text("read_file", {"path": "note.txt"}, reader)
                self.assertEqual((third, calls), ("second", 2))
                self.assertEqual(runtime.metrics["cache_hits"], 1)

        asyncio.run(run())

    def test_cached_agent_calls_keep_distinct_ids_and_full_external_results(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory, \
                 patch("astraea.storage.DATA_DIR", Path(directory) / "data"), \
                 patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}):
                Path(directory, "note.txt").write_text("exact content", encoding="utf-8")
                context = ContextManager(None, 64000)
                middleware = ToolOptimizationMiddleware(ToolRuntime(context))
                executions = 0

                async def handler(request):
                    nonlocal executions
                    executions += 1
                    return ToolMessage(content="exact content", tool_call_id=request.tool_call["id"])

                requests = [SimpleNamespace(tool_call={"name": "read_file", "args": {"path": "note.txt"}, "id": f"call{i}"})
                            for i in range(2)]
                first = await middleware.awrap_tool_call(requests[0], handler)
                second = await middleware.awrap_tool_call(requests[1], handler)
                self.assertEqual((first.content, second.content), ("exact content", "exact content"))
                self.assertEqual((first.tool_call_id, second.tool_call_id), ("call0", "call1"))
                self.assertEqual(executions, 1)
                self.assertEqual(len(list((Path(directory) / "data" / "context" / context.session_id / "items").glob("*.json"))), 2)

        asyncio.run(run())

    def test_prediction_prefetches_explicit_read_only_file(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}):
                Path(directory, "note.txt").write_text("prefetched fact", encoding="utf-8")
                runtime = ToolRuntime(ContextManager(None, 64000))
                tasks = runtime.prefetch("请读取 note.txt")
                self.assertEqual(len(tasks), 1)
                await asyncio.gather(*tasks)
                calls = 0

                async def should_not_run():
                    nonlocal calls
                    calls += 1
                    return "wrong"

                result = await runtime.cached_text("read_file", {"path": "note.txt"}, should_not_run)
                self.assertEqual(result, "prefetched fact")
                self.assertEqual(calls, 0)
                self.assertEqual(runtime.metrics["prefetch_hits"], 1)

        asyncio.run(run())

    def test_programmatic_plan_parallel_order_and_rejects_writes(self):
        async def run():
            class Delayed:
                async def ainvoke(self, args):
                    await asyncio.sleep(.04)
                    return f"value-{args['number']}"

            with tempfile.TemporaryDirectory() as directory, \
                 patch("astraea.storage.DATA_DIR", Path(directory)), \
                 patch.dict(READONLY, {"delayed": Delayed()}):
                runtime = ToolRuntime(ContextManager(None, 64000))
                plan = runtime.tools()[2]
                steps = [{"tool": "delayed", "args": {"number": number}} for number in range(4)]
                start = time.perf_counter()
                sequential = json.loads(await plan.ainvoke({"steps": steps, "parallel": False}))
                sequential_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                parallel = json.loads(await plan.ainvoke({"steps": steps, "parallel": True}))
                parallel_ms = (time.perf_counter() - start) * 1000
                self.assertEqual([item["content"] for item in parallel],
                                 [item["content"] for item in sequential])
                self.assertLess(parallel_ms, sequential_ms * .75)
                self.assertTrue((await plan.ainvoke({"steps": [{"tool": "write_file", "args": {}}]})).startswith("ERROR:"))

        asyncio.run(run())

    def test_long_result_is_shortened_only_for_model_view(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
                context = ContextManager(None, 64000)
                context._tool_runtime = ToolRuntime(context)
                long_result = "start\n" + "middle data\n" * 1000 + "end"
                history = [HumanMessage(content="Read large file"),
                           AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "large.txt"}, "id": "long1"}]),
                           ToolMessage(content=long_result, tool_call_id="long1")]
                context.record(history)
                visible = await context.select(context.prepare("What did large file say?"),
                                               SystemMessage(content="Assistant"), [])
                shown = next(item for item in visible if isinstance(item, ToolMessage))
                self.assertLess(len(str(shown.content)), len(long_result))
                self.assertIn("load_context", str(shown.content))
                self.assertEqual(context.history[-1].content, long_result)
                stored = context.store.search("large.txt")[0]
                self.assertEqual(context.store.get(stored["id"])["content"], long_result)

        asyncio.run(run())

    def test_mcp_calls_are_serialized_while_native_plan_can_parallelize(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
                runtime = ToolRuntime(ContextManager(None, 64000))
                middleware = ToolOptimizationMiddleware(runtime)
                active = maximum = 0

                async def handler(request):
                    nonlocal active, maximum
                    active += 1
                    maximum = max(maximum, active)
                    await asyncio.sleep(.03)
                    active -= 1
                    return ToolMessage(content="ok", tool_call_id=request.tool_call["id"])

                requests = [SimpleNamespace(tool_call={"name": "fetch_fetch", "args": {}, "id": f"c{i}"})
                            for i in range(3)]
                results = await asyncio.gather(*(middleware.awrap_tool_call(request, handler) for request in requests))
                self.assertEqual(maximum, 1)
                self.assertEqual([result.content for result in results], ["ok"] * 3)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
