"""Run a small, real-model check of T4 context compaction.

Usage: .\.venv\Scripts\python.exe tests\live_smoke.py
This makes two model requests and may incur API charges.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astraea.agent import build_agent, build_model
from astraea.config import load_settings
from astraea.context import ContextManager, usage_tokens


def main() -> None:
    settings = load_settings()
    model = build_model(settings)
    agent = build_agent(model)
    history = [
        HumanMessage(content="请记住：本次测试的校验词是紫色彗星。" + "这是较早的讨论内容。" * 120),
        AIMessage(content="我记住了校验词，并将继续处理任务。" + "这是较早的回答。" * 100),
        HumanMessage(content="现在先讨论项目结构。"),
        AIMessage(content="建议把入口、配置、工具和上下文管理分开。"),
        HumanMessage(content="再讨论测试方式。"),
        AIMessage(content="可以先用单元测试，再做真实模型的冒烟测试。"),
    ]
    question = "最开始指定的校验词是什么？只回答校验词。"
    planned = [*history, HumanMessage(content=question)]
    budget = int(usage_tokens(planned) / 0.91)

    with tempfile.TemporaryDirectory() as directory, patch("astraea.storage.DATA_DIR", Path(directory)):
        context = ContextManager(model, budget)
        context.record(history)
        request = context.prepare(question)
        assert context.tier == 4, f"Expected T4, got T{context.tier}"
        assert isinstance(request[0], SystemMessage), "Summary was not injected into messages"
        answer = str(agent.invoke({"messages": request})["messages"][-1].content)
        assert "紫色彗星" in answer, f"Important context was lost: {answer}"
        print("PASS: T4 triggered, summary entered messages, and the model retained the early constraint.")


if __name__ == "__main__":
    main()
