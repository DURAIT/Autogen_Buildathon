from __future__ import annotations

from pathlib import Path

import pytest

from mcp_server import SupportMcpService, create_mcp_server
from support_system.knowledge_graph import KnowledgeGraph
from support_system.mcp_client import SupportMcpClient, safe_mcp_url_for_display
from support_system.rag import LocalFaissStore


def _seed(data_dir: Path) -> None:
    product = LocalFaissStore(data_dir / "product_scope_faiss")
    support = LocalFaissStore(data_dir / "support_knowledge_faiss")
    product.add_texts(
        ["Acme Cloud Backup provides encrypted scheduled backups."],
        [{"source": "product_manual.pdf", "page": 1, "kind": "manual"}],
    )
    support.add_texts(
        ["Eligible refunds are returned within five business days."],
        [{"source": "refund_policy.pdf", "page": 2, "kind": "policy"}],
    )
    KnowledgeGraph(data_dir / "knowledge_graph.json").add_document(
        "refund_policy.pdf",
        [
            {
                "source": "refund_policy.pdf",
                "page": 2,
                "chunk": 1,
                "kind": "policy",
                "text": "Eligible refunds are returned within five business days.",
            }
        ],
    )


def test_mcp_service_exposes_multiple_support_tools(tmp_path: Path):
    _seed(tmp_path)
    service = SupportMcpService(tmp_path, min_relevance_score=0.05)

    product = service.search_product_scope("encrypted cloud backup", top_k=4)
    support = service.search_support_knowledge("refund five business days", top_k=6)
    stats = service.get_knowledge_stats()
    health = service.health_check()

    assert product["hits"][0]["source"] == "product_manual.pdf"
    assert support["hits"][0]["source"] == "refund_policy.pdf"
    assert {
        "chunk_id",
        "vector_score",
        "vector_rank",
        "bm25_score",
        "bm25_rank",
        "fusion_score",
        "score",
    }.issubset(support["hits"][0])
    assert stats["product_vectors"] == 1
    assert stats["support_vectors"] == 1
    assert health["status"] == "ok"


def test_mcp_stats_refresh_after_external_index_write(tmp_path: Path):
    service = SupportMcpService(tmp_path, min_relevance_score=0.05)
    external_store = LocalFaissStore(tmp_path / "support_knowledge_faiss")
    external_store.add_texts(
        ["A newly uploaded support policy."],
        [{"source": "new-policy.pdf", "page": 1, "kind": "policy"}],
    )

    stats = service.get_knowledge_stats()

    assert stats["support_vectors"] == 1
    assert stats["support_sources"] == 1


def test_mcp_server_registers_expected_tools(tmp_path: Path):
    _seed(tmp_path)
    server = create_mcp_server(tmp_path, host="127.0.0.1", port=8765)

    tool_names = {tool.name for tool in server._tool_manager.list_tools()}

    assert tool_names == {
        "health_check",
        "search_product_scope",
        "search_support_knowledge",
        "get_knowledge_stats",
        "get_recent_ticket_summary",
        "get_graph_stats",
        "search_knowledge_graph",
    }


def test_mcp_graph_tools_return_bounded_provenance_only(tmp_path: Path):
    _seed(tmp_path)
    service = SupportMcpService(tmp_path)

    stats = service.get_graph_stats()
    result = service.search_knowledge_graph("refund five", limit=500)

    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    assert result["limit"] == 20
    assert result["hits"][0]["source"] == "refund_policy.pdf"
    assert "text" not in result["hits"][0]


def test_mcp_server_rejects_unauthenticated_non_loopback_bind(tmp_path: Path):
    with pytest.raises(ValueError, match="loopback"):
        create_mcp_server(tmp_path, host="0.0.0.0", port=8765)


def test_mcp_url_display_never_includes_credentials():
    display = safe_mcp_url_for_display("https://user:secret@example.com:9443/mcp")

    assert display == "https://example.com:9443/mcp"
    assert "secret" not in display


def test_mcp_client_rejects_url_credentials():
    with pytest.raises(ValueError, match="credentials"):
        SupportMcpClient("https://user:secret@example.com/mcp")


def test_mcp_client_rejects_non_local_plain_http_endpoint():
    with pytest.raises(ValueError, match="localhost"):
        SupportMcpClient("http://example.com:8765/mcp")
