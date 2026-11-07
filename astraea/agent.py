"""Build Astraea's LangChain agent."""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from astraea.config import Settings
from astraea.tools import TOOLS


SYSTEM_PROMPT = (
    "You are Astraea, a concise and helpful assistant. "
    "Answer in the user's language. Use tools when they help. "
    "When the user asks about an earlier event or decision, use search_memory if the current chat does not contain it. "
    "Use search_web for current web information, then read relevant results with fetch_fetch when detail matters. "
    "Do not claim to have read files unless a tool returned them."
)


def build_model(settings: Settings, model_name: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=60,
        max_retries=2,
        streaming=True,
    )


def build_agent(model: ChatOpenAI, extra_tools=None):
    return create_agent(model=model, tools=[*TOOLS, *(extra_tools or [])], system_prompt=SYSTEM_PROMPT)
