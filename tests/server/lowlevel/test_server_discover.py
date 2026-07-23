"""Tests for the `server/discover` handler and DiscoverResult type.

Verifies that:
- The handler is registered on every Server instance by default.
- The response contains supportedVersions and capabilities.
- serverInfo is stamped into the result _meta field, NOT exposed as a
  top-level field, in accordance with spec #3002.
"""

import pytest

from mcp import Client
from mcp.server import Server, ServerRequestContext
from mcp.types import (
    Icon,
    Implementation,
    PromptsCapability,
    RequestParams,
    SERVER_INFO_META_KEY,
    ToolsCapability,
    ListPromptsResult,
    ListToolsResult,
    PaginatedRequestParams,
    Prompt,
    Tool,
)


@pytest.mark.anyio
async def test_server_discover_returns_supported_versions() -> None:
    """server/discover result contains a non-empty list of supported protocol versions."""
    server = Server("test-server")
    async with Client(server) as client:
        result = await client.session.server_discover()
        assert isinstance(result.supported_versions, list)
        assert len(result.supported_versions) > 0
        # The current protocol version must be included
        assert any("2025" in v or "2026" in v for v in result.supported_versions)


@pytest.mark.anyio
async def test_server_discover_server_info_in_meta_not_top_level() -> None:
    """serverInfo MUST be in _meta, not a top-level field of DiscoverResult (spec #3002)."""
    server = Server("my-server", version="1.2.3")
    async with Client(server) as client:
        result = await client.session.server_discover()

    # serverInfo must not be a direct attribute of DiscoverResult
    assert not hasattr(result, "server_info"), (
        "DiscoverResult MUST NOT have a top-level 'server_info' field; "
        "server identity belongs in _meta per spec #3002"
    )

    # serverInfo MUST be in _meta
    assert result.meta is not None, "result._meta must be set"
    assert SERVER_INFO_META_KEY in result.meta, (
        f"result._meta must contain '{SERVER_INFO_META_KEY}'"
    )


@pytest.mark.anyio
async def test_server_discover_server_info_fields() -> None:
    """serverInfo stamp in _meta reflects the Server constructor arguments."""
    icons = [Icon(src="https://example.test/icon.png")]
    server = Server(
        "info-server",
        version="9.9.9",
        title="Info Server",
        description="A server for testing discover.",
        website_url="https://example.test",
        icons=icons,
    )
    async with Client(server) as client:
        result = await client.session.server_discover()

    assert result.meta is not None
    raw_stamp = result.meta[SERVER_INFO_META_KEY]
    stamped = Implementation.model_validate(raw_stamp)
    assert stamped == Implementation(
        name="info-server",
        version="9.9.9",
        title="Info Server",
        description="A server for testing discover.",
        website_url="https://example.test",
        icons=icons,
    )


@pytest.mark.anyio
async def test_server_discover_unversioned_server_reports_empty_version() -> None:
    """An unversioned server reports version='' rather than the SDK's own version."""
    server = Server("unversioned")
    async with Client(server) as client:
        result = await client.session.server_discover()

    assert result.meta is not None
    stamp = result.meta[SERVER_INFO_META_KEY]
    assert stamp["name"] == "unversioned"
    assert stamp["version"] == ""


@pytest.mark.anyio
async def test_server_discover_capabilities_reflect_registered_handlers() -> None:
    """Capabilities in DiscoverResult match what the server has registered."""

    async def handle_list_prompts(
        ctx: ServerRequestContext, params: PaginatedRequestParams | None
    ) -> ListPromptsResult:
        return ListPromptsResult(prompts=[Prompt(name="p")])

    async def handle_list_tools(
        ctx: ServerRequestContext, params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        return ListToolsResult(tools=[Tool(name="t", inputSchema={})])

    server = Server(
        "caps-server",
        on_list_prompts=handle_list_prompts,
        on_list_tools=handle_list_tools,
    )
    async with Client(server) as client:
        result = await client.session.server_discover()

    assert result.capabilities.prompts is not None
    assert result.capabilities.tools is not None
    # Resources were not registered
    assert result.capabilities.resources is None


@pytest.mark.anyio
async def test_server_discover_instructions_threaded_through() -> None:
    """instructions in DiscoverResult match what the Server was constructed with."""
    server = Server("inst-server", instructions="Read the docs first.")
    async with Client(server) as client:
        result = await client.session.server_discover()
    assert result.instructions == "Read the docs first."


@pytest.mark.anyio
async def test_server_discover_no_instructions_by_default() -> None:
    """instructions defaults to None when not set."""
    server = Server("bare")
    async with Client(server) as client:
        result = await client.session.server_discover()
    assert result.instructions is None
