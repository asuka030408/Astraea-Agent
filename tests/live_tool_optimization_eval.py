"""Optional paired real-model tool evaluation with a temporary workspace."""

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from langchain.agents import create_agent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astraea.agent import SYSTEM_PROMPT, build_agent, build_model
from astraea.config import load_settings
from astraea.context import ContextManager, ContextMiddleware
from astraea.tools import TOOLS
from astraea.tools.context import context_tools


TASKS = [
    ("读取 note.txt，只回答里面的校验词。", "青鸟17"),
    ("读取 data.json，只回答 count 的数值。", "42"),
    ("列出当前工作区的文件，只回答 .txt 文件有几个。", "2"),
    ("读取 note.txt 和 second.txt，按顺序回答两个校验词。", "青鸟17|白帆29"),
    ("在工作区创建 result.txt，内容恰好是 VERIFIED。完成后回答已创建。", "write"),
    ("把 note.txt 复制为 copy.txt，完成后回答已复制。", "copy"),
    ("删除工作区的 to_delete.txt，完成后回答已删除。", "delete"),
]


def correct(answer: str, expected: str) -> bool:
    if "|" in expected:
        return all(part in answer for part in expected.split("|"))
    if expected.isdigit():
        return bool(re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", answer))
    return expected in answer


async def main():
    settings = load_settings()
    model = build_model(settings)
    rows = []
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as data_directory, \
         patch.dict(os.environ, {"ASTRAEA_WORKSPACE": directory}), \
         patch("astraea.config.USER_SETTINGS_FILE", Path(directory) / "unused-settings.json"), \
         patch("astraea.storage.DATA_DIR", Path(data_directory)):
        Path(directory, "note.txt").write_text("校验词：青鸟17。\n", encoding="utf-8")
        Path(directory, "second.txt").write_text("校验词：白帆29。\n", encoding="utf-8")
        Path(directory, "data.json").write_text('{"count":42,"status":"ready"}', encoding="utf-8")
        for index, (question, expected) in enumerate(TASKS):
            order = ("baseline", "optimized") if index % 2 == 0 else ("optimized", "baseline")
            for mode in order:
                for transient in ("result.txt", "copy.txt"):
                    (Path(directory) / transient).unlink(missing_ok=True)
                (Path(directory) / "to_delete.txt").unlink(missing_ok=True)
                if expected == "delete":
                    Path(directory, "to_delete.txt").write_text("temporary", encoding="utf-8")
                context = ContextManager(model, settings.context_window)
                all_tools = [*TOOLS, *context_tools(context.session_id)]
                agent = (create_agent(model=model, tools=all_tools, system_prompt=SYSTEM_PROMPT,
                                      middleware=[ContextMiddleware(context)]) if mode == "baseline"
                         else build_agent(model, context=context))
                start = time.perf_counter()
                runtime = getattr(context, "_tool_runtime", None)
                prefetch_tasks = runtime.prefetch(question) if runtime is not None else []
                try:
                    result = await agent.ainvoke({"messages": context.prepare(question)},
                                                 config={"recursion_limit": 12})
                    elapsed = (time.perf_counter() - start) * 1000
                    answer = str(result["messages"][-1].content)
                    calls = [call["name"] for message in result["messages"]
                             for call in (getattr(message, "tool_calls", None) or [])]
                    if expected == "write":
                        accurate = (Path(directory, "result.txt").is_file() and
                                    Path(directory, "result.txt").read_text(encoding="utf-8") == "VERIFIED")
                    elif expected == "copy":
                        accurate = (Path(directory, "copy.txt").is_file() and
                                    Path(directory, "copy.txt").read_text(encoding="utf-8") ==
                                    Path(directory, "note.txt").read_text(encoding="utf-8"))
                    elif expected == "delete":
                        accurate = not Path(directory, "to_delete.txt").exists()
                    else:
                        accurate = correct(answer, expected)
                    row = {"task": index + 1, "mode": mode, "ms": round(elapsed, 1),
                           "correct": accurate, "answer": answer[:300],
                           "tool_calls": calls, "schema_tokens": context.last_budget.get("tools")}
                except Exception as exc:
                    row = {"task": index + 1, "mode": mode, "correct": False,
                           "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                finally:
                    for task in prefetch_tasks:
                        if not task.done():
                            task.cancel()
                    if prefetch_tasks:
                        await asyncio.gather(*prefetch_tasks, return_exceptions=True)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    output = Path(__file__).resolve().parent.parent / "docs" / "tool-optimization-live-results.json"
    output.write_text(json.dumps({"model": settings.model, "tasks": len(TASKS), "results": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    asyncio.run(main())
