"""Terminal interaction; the agent loop itself is provided by LangChain."""

import asyncio

from astraea.agent import build_model
from astraea.config import load_settings
from astraea.context import ContextManager
from astraea.memory import MemoryManager, read_profile_markdown
from astraea.mcp_tools import invoke_with_mcp


def main() -> None:
    try:
        settings = load_settings()
        model = build_model(settings)
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return

    print("Astraea Agent — type /exit to quit, /clear to reset this conversation.")
    context = ContextManager(build_model(settings, settings.compactor_model), settings.context_window)
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in {"/exit", "/quit"}:
            print("Goodbye.")
            break
        if user_input.lower() == "/clear":
            context = ContextManager(build_model(settings, settings.compactor_model), settings.context_window)
            print("Conversation cleared.")
            continue
        if not user_input:
            continue

        try:
            result = asyncio.run(invoke_with_mcp(model, settings,
                                                 context.prepare(user_input, read_profile_markdown())))
        except Exception as exc:
            print(f"Request failed: {exc}")
            continue

        print(f"\nAstraea > {result['messages'][-1].content}")
        try:
            context.record(result["messages"])
            MemoryManager(model).process_turn(context.session_id, context.history)
        except Exception as exc:
            print(f"[Memory/Context] Could not finish post-turn processing: {exc}")
