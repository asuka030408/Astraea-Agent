"""Optional real-model continuity check for adaptive context selection."""

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astraea.agent import build_agent, build_model
from astraea.config import load_settings
from astraea.context import ContextManager


async def main() -> None:
    settings = load_settings()
    model = build_model(settings)
    compactor = build_model(settings, settings.compactor_model)
    with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
        context = ContextManager(compactor, min(settings.context_window, 16000))
        history = [HumanMessage(content="本次测试的校验词是紫色彗星，请记住。"), AIMessage(content="已记录。")]
        for number in range(12):
            history.extend([HumanMessage(content=f"第{number}轮讨论项目结构。" + "这里是无关的背景。" * 160),
                            AIMessage(content="继续讨论结构。" * 100)])
        context.record(history)
        request = context.prepare("最开始指定的校验词是什么？只回答校验词。")
        result = await build_agent(model, context=context).ainvoke({"messages": request})
        answer = str(result["messages"][-1].content)
        assert "紫色彗星" in answer, f"Early fact was not recalled: {answer}"
        print(f"PASS: recalled early fact; model input {context.last_budget['input_total']} / "
              f"{context.last_budget['input_limit']} tokens; checkpoint={bool(context.summary)}")


if __name__ == "__main__":
    asyncio.run(main())
