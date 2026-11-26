"""Build Astraea's LangChain agent."""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from astraea.config import Settings
from astraea.context import ContextMiddleware
from astraea.tools import TOOLS
from astraea.tools.context import context_tools
from astraea.tool_runtime import ToolOptimizationMiddleware, ToolRuntime


SYSTEM_PROMPT = (
    "You are Astraea, a concise and helpful assistant. "
    "Answer in the user's language. Use tools when they help. "
    "When the user asks about an earlier event or decision, use search_memory if the current chat does not contain it. "
    "Use search_web for current web information, then read relevant results with fetch_fetch when detail matters. "
    "Do not claim to have read files unless a tool returned them."
    " Earlier tool results may be stored externally; use search_context and load_context when needed."
    " Use search_history and load_history when an older dialogue detail is missing from the current request."
    " If a needed tool is not visible, call search_tools and then load_tools to activate it."
    " Issue independent read-only tool calls together when possible; run_readonly_plan is useful for larger batches."
    " For explicit copy, move, or delete requests, use the direct file tool without a preliminary read unless file contents are needed."
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


def build_agent(model: ChatOpenAI, extra_tools=None, context=None):
    session_tools = context_tools(context.session_id) if context is not None else []
    runtime = None
    if context is not None:
        runtime = getattr(context, "_tool_runtime", None)
        if runtime is None:
            runtime = context._tool_runtime = ToolRuntime(context)
    runtime_tools = runtime.tools() if runtime is not None else []
    all_tools = [*TOOLS, *session_tools, *runtime_tools, *(extra_tools or [])]
    if runtime is not None:
        runtime.bind_catalog(all_tools)
    middleware = [ToolOptimizationMiddleware(runtime), ContextMiddleware(context)] if runtime else []
    return create_agent(model=model, tools=all_tools,
                        system_prompt=SYSTEM_PROMPT, middleware=middleware)
