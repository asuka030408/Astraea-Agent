"""Persistent MCP session: reuse, loud startup failures, and one reconnect."""

import asyncio
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from langchain_core.messages import AIMessage

import astraea.mcp_tools as mcp


CONFIG = {"mcpServers": {"fetch": {}, "filesystem": {}, "time": {}}}
ALL_TOOLS = ["fetch_fetch", "filesystem_read_text_file", "time_get_current_time"]


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeAdapter:
    starts = 0
    tool_names = None

    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        type(self).starts += 1
        return self

    async def __aexit__(self, *args):
        return False

    async def list_tools(self):
        return [FakeTool(name) for name in (type(self).tool_names or ALL_TOOLS)]


class FakeAgent:
    async def ainvoke(self, payload, config=None):
        return {"messages": [*payload["messages"], AIMessage(content="ok")]}


class FlakyAgent:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, payload, config=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("stdio stream connection closed")
        return {"messages": [*payload["messages"], AIMessage(content="ok")]}


class OverlapAgent:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def ainvoke(self, payload, config=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.calls += 1
        self.active -= 1
        return {"messages": [*payload["messages"], AIMessage(content="ok")]}


@contextmanager
def mcp_patches(agent):
    with patch.object(mcp, "MCPAdapter", FakeAdapter), \
         patch.object(mcp, "mcp_config", return_value=CONFIG), \
         patch.object(mcp, "set_discovered_tools"), \
         patch("astraea.agent.build_agent", return_value=agent):
        yield


class MCPPoolTests(unittest.TestCase):
    def setUp(self):
        FakeAdapter.starts = 0
        FakeAdapter.tool_names = None
        mcp.bind_event_loop(None)

    def tearDown(self):
        mcp.bind_event_loop(None)

    def _scenario(self, body):
        async def run():
            mcp.bind_event_loop(asyncio.get_running_loop())
            try:
                return await body()
            finally:
                await mcp.close_mcp_session()
                mcp.bind_event_loop(None)
        return asyncio.run(run())

    def test_two_turns_start_servers_only_once(self):
        async def body():
            with mcp_patches(FakeAgent()):
                for _ in range(2):
                    await mcp.invoke_with_mcp(object(), object(), [])
                return FakeAdapter.starts
        self.assertEqual(self._scenario(body), 1)

    def test_missing_server_fails_loudly(self):
        FakeAdapter.tool_names = ["fetch_fetch", "filesystem_read_text_file"]

        async def body():
            with mcp_patches(FakeAgent()):
                with self.assertRaises(RuntimeError) as caught:
                    await mcp.invoke_with_mcp(object(), object(), [])
                return str(caught.exception)
        self.assertIn("time", self._scenario(body))

    def test_disconnect_reconnects_once(self):
        agent = FlakyAgent()

        async def body():
            with mcp_patches(agent):
                result = await mcp.invoke_with_mcp(object(), object(), [])
                return FakeAdapter.starts, result["messages"][-1].content
        self.assertEqual(self._scenario(body), (2, "ok"))

    def test_concurrent_turns_share_session_without_overlapping(self):
        agent = OverlapAgent()

        async def body():
            with mcp_patches(agent):
                await asyncio.gather(*(mcp.invoke_with_mcp(object(), object(), []) for _ in range(3)))
                return agent.max_active, agent.calls, FakeAdapter.starts
        self.assertEqual(self._scenario(body), (1, 3, 1))

    def test_unbound_loop_keeps_per_turn_connection(self):
        async def body():
            with mcp_patches(FakeAgent()):
                for _ in range(2):
                    await mcp.invoke_with_mcp(object(), object(), [])
                return FakeAdapter.starts
        self.assertEqual(asyncio.run(body()), 2)


if __name__ == "__main__":
    unittest.main()
