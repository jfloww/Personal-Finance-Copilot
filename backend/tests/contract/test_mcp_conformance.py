from __future__ import annotations

import asyncio

from mcp import Client

from offerdelta.agent.tools.definitions import build_tool_registry
from offerdelta.agent.tools.operations import build_operations_registry
from offerdelta.infrastructure.mcp.server import create_mcp_server


def test_mcp_advertises_the_registry_byte_for_byte_and_returns_the_same_results() -> None:
    asyncio.run(_assert_conformance())


async def _assert_conformance() -> None:
    registry = build_tool_registry()
    server = create_mcp_server(registry)

    async with Client(server, raise_exceptions=True) as client:
        listed = await client.list_tools()
        advertised = {tool.name: tool for tool in listed.tools}

        assert set(advertised) == set(registry.names)
        for tool in registry.tools:
            remote = advertised[tool.name]
            assert remote.description == tool.description
            assert remote.input_schema == tool.input_schema
            assert remote.annotations is not None
            assert remote.annotations.read_only_hint is True
            assert remote.annotations.destructive_hint is False

        calls: tuple[tuple[str, dict[str, object]], ...] = (
            ("list_profiles", {}),
            (
                "break_even",
                {"current": "auburn_current", "candidate": "new_jersey_candidate"},
            ),
            (
                "compare_offers",
                {
                    "current": "auburn_current",
                    "candidate": "new_jersey_candidate",
                    "horizon_months": 12,
                    "move_date": "2026-07-01",
                },
            ),
        )
        for name, arguments in calls:
            local = registry.call(name, arguments)
            remote_result = await client.call_tool(name, arguments)
            assert remote_result.is_error is (not local.ok)
            assert remote_result.structured_content == local.as_json()


def test_mcp_returns_schema_and_domain_errors_as_tool_errors() -> None:
    asyncio.run(_assert_tool_errors())


async def _assert_tool_errors() -> None:
    async with Client(create_mcp_server(), raise_exceptions=True) as client:
        invalid = await client.call_tool("search_transactions", {})
        unknown = await client.call_tool("not_a_tool", {})

        assert invalid.is_error
        assert invalid.structured_content is not None
        assert "missing required field" in invalid.structured_content["error"]
        assert unknown.is_error
        assert unknown.structured_content is not None
        assert "unknown tool" in unknown.structured_content["error"]


def test_default_mcp_exposes_six_read_only_operations_tools() -> None:
    asyncio.run(_assert_default_operations())


async def _assert_default_operations() -> None:
    registry = build_operations_registry()
    async with Client(create_mcp_server(), raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools
        assert {tool.name for tool in tools} == set(registry.names)
        for tool in tools:
            local = registry.get(tool.name)
            assert local is not None
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.destructive_hint is False
            assert tool.input_schema == local.input_schema
        args = {"transaction_id": "TX-9825-B2", "reason": "possible_duplicate"}
        remote = await client.call_tool("propose_review_case", args)
        assert remote.structured_content == registry.call("propose_review_case", args).as_json()
        assert remote.structured_content["payload"]["ledger_mutated"] is False
