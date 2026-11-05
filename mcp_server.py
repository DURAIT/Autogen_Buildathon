from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from support_system.config import Settings
from support_system.knowledge_graph import KnowledgeGraph
from support_system.rag import Embedder, LocalFaissStore
from support_system.storage import TicketLogger


class SupportMcpService:
    """Read-only support tools shared by AutoGen agents through MCP."""

    def __init__(
        self,
        data_dir: str | Path,
        min_relevance_score: float = 0.20,
        embedder: Embedder | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.min_relevance_score = min_relevance_score
        self.embedder = embedder
        self.product_store = LocalFaissStore(
            self.data_dir / "product_scope_faiss", embedder=self.embedder
        )
        self.support_store = LocalFaissStore(
            self.data_dir / "support_knowledge_faiss", embedder=self.embedder
        )
        self.knowledge_graph = KnowledgeGraph(self.data_dir / "knowledge_graph.json")
        self.ticket_logger = TicketLogger(self.data_dir / "tickets.csv")

    def _refresh_stores(self) -> None:
        """Refresh process-local snapshots so app-side ingestion is visible immediately."""
        self.product_store = LocalFaissStore(
            self.data_dir / "product_scope_faiss", embedder=self.embedder
        )
        self.support_store = LocalFaissStore(
            self.data_dir / "support_knowledge_faiss", embedder=self.embedder
        )

    def health_check(self) -> dict[str, Any]:
        """Return MCP service readiness and index counts."""
        self._refresh_stores()
        return {
            "status": "ok",
            "service": "supportiq-mcp",
            "product_vectors": int(self.product_store.index.ntotal),
            "support_vectors": int(self.support_store.index.ntotal),
        }

    def search_product_scope(self, query: str, top_k: int = 4) -> dict[str, Any]:
        """Search product manuals and catalogs to determine whether a query is in scope."""
        safe_top_k = min(max(int(top_k), 1), 12)
        hits = self.product_store.search(
            query, top_k=safe_top_k, min_score=self.min_relevance_score
        )
        return {"query": query, "count": len(hits), "hits": hits}

    def search_support_knowledge(self, query: str, top_k: int = 6) -> dict[str, Any]:
        """Search support policies, FAQs, catalogs, and auto-resolved-case memory."""
        safe_top_k = min(max(int(top_k), 1), 20)
        hits = self.support_store.search(
            query, top_k=safe_top_k, min_score=self.min_relevance_score
        )
        return {"query": query, "count": len(hits), "hits": hits}

    def get_knowledge_stats(self) -> dict[str, Any]:
        """Return vector counts and indexed source counts without exposing document text."""
        self._refresh_stores()
        product_sources = {
            str(record.get("source"))
            for record in self.product_store.records
            if record.get("source")
        }
        support_sources = {
            str(record.get("source"))
            for record in self.support_store.records
            if record.get("source")
        }
        return {
            "product_vectors": int(self.product_store.index.ntotal),
            "support_vectors": int(self.support_store.index.ntotal),
            "product_sources": len(product_sources),
            "support_sources": len(support_sources),
        }

    def get_recent_ticket_summary(self, limit: int = 10) -> dict[str, Any]:
        """Return privacy-reduced recent ticket metadata for operational analytics."""
        safe_limit = min(max(int(limit), 1), 100)
        rows = self.ticket_logger.read_all()[-safe_limit:]
        allowed = {
            "ticket_id",
            "timestamp_utc",
            "category",
            "urgency",
            "sentiment",
            "priority",
            "confidence",
            "sources",
            "status",
            "escalation_reason",
        }
        summaries = [{key: value for key, value in row.items() if key in allowed} for row in rows]
        return {"count": len(summaries), "tickets": list(reversed(summaries))}

    def get_graph_stats(self) -> dict[str, Any]:
        """Return typed provenance graph counts without exposing document contents."""
        return self.knowledge_graph.stats()

    def search_knowledge_graph(
        self, query: str, limit: int = 10, max_nodes: int = 50
    ) -> dict[str, Any]:
        """Search bounded, sanitized provenance previews and their evidence subgraph."""
        return self.knowledge_graph.search(query, limit=limit, max_nodes=max_nodes)


def create_mcp_server(
    data_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    min_relevance_score: float = 0.20,
    embedder: Embedder | None = None,
) -> FastMCP:
    normalized_host = host.strip().lower()
    if normalized_host != "localhost":
        try:
            is_loopback = ipaddress.ip_address(normalized_host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError(
                "Unauthenticated MCP server must bind to a loopback address only"
            )
    if not 1 <= int(port) <= 65535:
        raise ValueError("MCP port must be between 1 and 65535")
    service = SupportMcpService(
        data_dir,
        min_relevance_score=min_relevance_score,
        embedder=embedder,
    )
    server = FastMCP(
        name="SupportIQ MCP",
        instructions=(
            "Read-only tools for product-scope retrieval, support knowledge retrieval, "
            "knowledge statistics, and privacy-reduced ticket analytics."
        ),
        host=host,
        port=port,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        max_request_body_size=256 * 1024,
        log_level="INFO",
    )
    server.tool()(service.health_check)
    server.tool()(service.search_product_scope)
    server.tool()(service.search_support_knowledge)
    server.tool()(service.get_knowledge_stats)
    server.tool()(service.get_recent_ticket_summary)
    server.tool()(service.get_graph_stats)
    server.tool()(service.search_knowledge_graph)
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SupportIQ MCP tool server.")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8765")))
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    parser.add_argument(
        "--min-score",
        type=float,
        default=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.20")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    server = create_mcp_server(
        args.data_dir,
        host=args.host,
        port=args.port,
        min_relevance_score=args.min_score,
        embedder=settings.build_embedder(),
    )
    print(f"SupportIQ MCP ready at http://{args.host}:{args.port}/mcp")
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
