"""Low-level MCP server that preserves the registry's explicit schemas exactly."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    ToolAnnotations,
)
from mcp.types import (
    Tool as MCPTool,
)

from offerdelta.agent.tools.operations import build_operations_registry
from offerdelta.agent.tools.registry import ToolRegistry

SERVER_NAME = "offerdelta-transaction-operations"
SERVER_VERSION = "0.1.0"


@asynccontextmanager
async def _lifespan(_server: Server[None]) -> AsyncIterator[None]:
    yield None


def create_mcp_server(registry: ToolRegistry | None = None) -> Server[None]:
    """Build an MCP server over exactly one registry instance."""
    registry = registry or build_operations_registry()

    async def list_tools(
        _context: ServerRequestContext[None], _params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                MCPTool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema,
                    annotations=ToolAnnotations(
                        readOnlyHint=True,
                        destructiveHint=False,
                        idempotentHint=True,
                        openWorldHint=False,
                    ),
                )
                for tool in registry.tools
            ],
            cacheScope="public",
        )

    async def call_tool(
        _context: ServerRequestContext[None], params: CallToolRequestParams
    ) -> CallToolResult:
        arguments: dict[str, object] = dict(params.arguments or {})
        result = registry.call(params.name, arguments)
        structured = result.as_json()
        return CallToolResult(
            content=[
                TextContent(text=json.dumps(structured, separators=(",", ":"), sort_keys=True))
            ],
            structuredContent=structured,
            isError=not result.ok,
        )

    return Server[None](
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "Read-only investigation over synthetic demonstration transactions and a synthetic "
            "policy excerpt. A review proposal is not persisted; no ledger or queue is modified. "
            "This is not a connection to real accounts, tenant records, or vector RAG."
        ),
        lifespan=_lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def run_stdio() -> None:
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    asyncio.run(run_stdio())


if __name__ == "__main__":  # pragma: no cover - exercised by an MCP host
    main()
