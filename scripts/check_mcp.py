from __future__ import annotations

import asyncio

from support_system.config import Settings
from support_system.mcp_client import SupportMcpClient


async def check() -> None:
    settings = Settings.from_env()
    async with SupportMcpClient(
        settings.mcp_server_url,
        timeout=min(settings.mcp_timeout_seconds, 5.0),
    ) as client:
        health = await client.health_check()
        if health.get("status") != "ok":
            raise RuntimeError("MCP health_check did not return status=ok")
        tools = await client.list_tools()
        required = {
            "health_check",
            "search_product_scope",
            "search_support_knowledge",
            "get_knowledge_stats",
            "get_recent_ticket_summary",
            "get_graph_stats",
            "search_knowledge_graph",
        }
        actual = set(tools)
        missing = required.difference(actual)
        extra = actual.difference(required)
        if missing or extra:
            raise RuntimeError(
                "MCP tool contract mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(extra)}"
            )
        print(f"MCP ready: {settings.mcp_server_url} ({len(tools)} tools)")


def main() -> None:
    try:
        asyncio.run(check())
    except Exception as exc:
        raise SystemExit(f"MCP readiness check failed: {exc}") from exc


if __name__ == "__main__":
    main()
