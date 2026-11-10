from __future__ import annotations

import json
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

from autogen_ext.tools.mcp import McpWorkbench, StreamableHttpServerParams


def safe_mcp_url_for_display(server_url: str) -> str:
    """Return an endpoint label without URL user-info or query fragments."""
    parsed = urlparse(server_url)
    if not parsed.scheme or not parsed.hostname:
        return "Invalid MCP endpoint"
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or "/"
    return f"{parsed.scheme}://{host}{port}{path}"


class SupportMcpClient:
    """Typed client for the standalone SupportIQ Streamable HTTP MCP server."""

    def __init__(self, server_url: str, timeout: float = 20.0) -> None:
        parsed = urlparse(server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("MCP_SERVER_URL must be a valid HTTP or HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("MCP_SERVER_URL must not contain embedded credentials")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Plain HTTP MCP endpoints must use localhost; use HTTPS for remote servers")
        self.server_url = server_url.rstrip("/")
        self.params = StreamableHttpServerParams(
            url=self.server_url,
            timeout=timeout,
            sse_read_timeout=max(timeout, 60.0),
            terminate_on_close=True,
        )
        self.workbench = McpWorkbench(self.params)
        self._started = False

    async def __aenter__(self) -> "SupportMcpClient":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        if not self._started:
            try:
                await self.workbench.start()
            except Exception as exc:
                raise ConnectionError(
                    f"Cannot connect to MCP server at {self.server_url}. "
                    "Start it first with: python mcp_server.py"
                ) from exc
            self._started = True

    async def stop(self) -> None:
        if self._started:
            await self.workbench.stop()
            self._started = False

    async def list_tools(self) -> list[str]:
        self._ensure_started()
        tools = await self.workbench.list_tools()
        return [tool["name"] for tool in tools]

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("MCP client is not started; use 'async with SupportMcpClient(...)'")

    @staticmethod
    def _decode_result(tool_result: Any) -> dict[str, Any]:
        if tool_result.is_error:
            details = " ".join(
                getattr(item, "content", str(item)) for item in tool_result.result
            )
            raise RuntimeError(f"MCP tool {tool_result.name} failed: {details}")
        for item in tool_result.result:
            content = getattr(item, "content", None)
            if isinstance(content, str):
                try:
                    decoded = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    return decoded
        raise RuntimeError(f"MCP tool {tool_result.name} returned no JSON object")

    async def _call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_started()
        result = await self.workbench.call_tool(name, arguments or {})
        return self._decode_result(result)

    async def health_check(self) -> dict[str, Any]:
        return await self._call("health_check")

    async def search_product_scope(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        payload = await self._call("search_product_scope", {"query": query, "top_k": top_k})
        return list(payload.get("hits", []))

    async def search_support_knowledge(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        payload = await self._call(
            "search_support_knowledge", {"query": query, "top_k": top_k}
        )
        return list(payload.get("hits", []))

    async def get_knowledge_stats(self) -> dict[str, Any]:
        return await self._call("get_knowledge_stats")

    async def get_recent_ticket_summary(self, limit: int = 10) -> dict[str, Any]:
        return await self._call("get_recent_ticket_summary", {"limit": limit})
