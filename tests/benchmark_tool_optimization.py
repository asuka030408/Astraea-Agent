"""Repeatable tool-runtime microbenchmarks; no model API calls.

Runs controlled 30–50 ms I/O delays to make cache, parallelism and prefetch
effects measurable. This is not a real-model latency benchmark.
"""

import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astraea.context import ContextManager, _measure_tool_schemas
from astraea.tool_runtime import READONLY, ToolRuntime
from astraea.tools import TOOLS
from astraea.tools.context import context_tools


async def measure(action):
    start = time.perf_counter()
    value = await action()
    return (time.perf_counter() - start) * 1000, value


async def main():
    with tempfile.TemporaryDirectory() as directory, \
         patch("astraea.storage.DATA_DIR", Path(directory) / "data"), \
         patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}):
        path = Path(directory, "note.txt")
        path.write_text("verified fixture result\n" * 100, encoding="utf-8")
        expected = path.read_text(encoding="utf-8")
        context = ContextManager(None, 64000)
        runtime = ToolRuntime(context)
        tools = [*TOOLS, *context_tools(context.session_id), *runtime.tools()]
        runtime.bind_catalog(tools)
        selected = runtime.select_schemas([HumanMessage(content="请读取 note.txt")], tools)
        schema = {"full_tools": len(tools), "selected_tools": len(selected),
                  "full_tokens": _measure_tool_schemas(tools),
                  "selected_tokens": _measure_tool_schemas(selected)}

        async def delayed_read():
            await asyncio.sleep(.03)
            return path.read_text(encoding="utf-8")

        cache_before, cache_after = [], []
        cache_correct = 0
        for _ in range(4):
            async def baseline_sequential():
                return [await delayed_read() for _ in range(20)]
            baseline_ms, baseline = await measure(baseline_sequential)
            fresh_runtime = ToolRuntime(ContextManager(None, 64000))
            async def optimized():
                return [await fresh_runtime.cached_text("read_file", {"path": "note.txt"}, delayed_read)
                        for _ in range(20)]
            optimized_ms, after = await measure(optimized)
            cache_before.append(baseline_ms)
            cache_after.append(optimized_ms)
            cache_correct += sum(a == b == expected for a, b in zip(baseline, after))

        class Delayed:
            async def ainvoke(self, args):
                await asyncio.sleep(.05)
                return f"result-{args['number']}"

        plan_before, plan_after = [], []
        plan_correct = 0
        with patch.dict(READONLY, {"delayed": Delayed()}):
            for _ in range(4):
                plan = ToolRuntime(ContextManager(None, 64000)).tools()[2]
                steps = [{"tool": "delayed", "args": {"number": number}} for number in range(4)]
                before_ms, before = await measure(lambda: plan.ainvoke({"steps": steps, "parallel": False}))
                after_ms, after = await measure(lambda: plan.ainvoke({"steps": steps, "parallel": True}))
                plan_before.append(before_ms)
                plan_after.append(after_ms)
                b = json.loads(before)
                a = json.loads(after)
                plan_correct += sum(x["content"] == y["content"] == f"result-{i}"
                                    for i, (x, y) in enumerate(zip(b, a)))

        class DelayedRead:
            async def ainvoke(self, args):
                await asyncio.sleep(.03)
                return path.read_text(encoding="utf-8")

        predict_before, predict_after = [], []
        predict_correct = 0
        with patch.dict(READONLY, {"read_file": DelayedRead()}):
            for _ in range(4):
                async def baseline():
                    await asyncio.sleep(.05)  # Simulated model planning time.
                    return await READONLY["read_file"].ainvoke({"path": "note.txt"})
                before_ms, before = await measure(baseline)
                fresh_runtime = ToolRuntime(ContextManager(None, 64000))
                async def optimized():
                    tasks = fresh_runtime.prefetch("请读取 note.txt")
                    await asyncio.sleep(.05)
                    after = await fresh_runtime.cached_text("read_file", {"path": "note.txt"},
                                                            lambda: READONLY["read_file"].ainvoke({"path": "note.txt"}))
                    await asyncio.gather(*tasks)
                    return after
                after_ms, after = await measure(optimized)
                predict_before.append(before_ms)
                predict_after.append(after_ms)
                predict_correct += int(before == after == expected)

        long_result = "header\n" + "middle detail\n" * 1000 + "footer"
        call = {"id": "long-result", "name": "read_file", "args": {"path": "note.txt"}}
        item_id = context.store.put_result(call, ToolMessage(content=long_result, tool_call_id=call["id"]))
        compressed = runtime.compress_result(ToolMessage(content=long_result, tool_call_id=call["id"]), item_id)

        report = {
            "schema": schema,
            "cache": {"baseline_ms_median": round(statistics.median(cache_before), 2),
                      "optimized_ms_median": round(statistics.median(cache_after), 2),
                      "correct": cache_correct, "total": 80},
            "parallel_plan": {"sequential_ms_median": round(statistics.median(plan_before), 2),
                              "parallel_ms_median": round(statistics.median(plan_after), 2),
                              "correct": plan_correct, "total": 16},
            "prediction": {"baseline_ms_median": round(statistics.median(predict_before), 2),
                           "prefetch_ms_median": round(statistics.median(predict_after), 2),
                           "correct": predict_correct, "total": 4},
            "compression": {"raw_chars": len(long_result), "visible_chars": len(str(compressed.content)),
                            "full_result_preserved": context.store.get(item_id)["content"] == long_result},
        }
        output = Path(__file__).resolve().parent.parent / "docs" / "tool-optimization-benchmark-results.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
