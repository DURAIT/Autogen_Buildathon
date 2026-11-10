from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from support_system import ingestion
from support_system.ingestion import ingest_pdf
from support_system.knowledge_graph import KnowledgeGraph
from support_system.rag import LocalFaissStore


def _evidence(source: str = "refund_policy.pdf") -> list[dict]:
    return [
        {
            "source": source,
            "page": 2,
            "chunk": 1,
            "kind": "policy",
            "text": (
                "Contact alice@example.com or +1 212 555 0100. "
                "password=hunter2 api_key=sk-abcdefghijklmnopqrstuvwxyz Refunds arrive in five days."
            ),
        }
    ]


def test_graph_persists_typed_provenance_as_safe_json_and_is_idempotent(tmp_path: Path):
    path = tmp_path / "knowledge_graph.json"
    graph = KnowledgeGraph(path)

    first = graph.add_document("refund_policy.pdf", _evidence())
    second = graph.add_document("refund_policy.pdf", _evidence())
    reloaded = KnowledgeGraph(path)

    assert first == {"documents": 1, "pages": 1, "chunks": 1, "kinds": 1}
    assert second == {"documents": 0, "pages": 0, "chunks": 0, "kinds": 0}
    assert reloaded.stats() == {
        "documents": 1,
        "pages": 1,
        "chunks": 1,
        "kinds": 1,
        "nodes": 4,
        "edges": 3,
    }
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "supportiq-provenance-v1"
    assert {node["type"] for node in payload["nodes"]} == {"Document", "Page", "Chunk", "Kind"}
    assert {edge["relation"] for edge in payload["edges"]} == {
        "HAS_PAGE",
        "HAS_CHUNK",
        "HAS_KIND",
    }
    serialized = path.read_text(encoding="utf-8")
    assert "alice@example.com" not in serialized
    assert "212 555 0100" not in serialized
    assert "hunter2" not in serialized
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "[EMAIL]" in serialized and "[PHONE]" in serialized
    assert "[SECRET]" in serialized
    assert not path.with_suffix(".pickle").exists()


def test_graph_repairs_and_persists_missing_edges_without_new_nodes(tmp_path: Path):
    path = tmp_path / "knowledge_graph.json"
    graph = KnowledgeGraph(path)
    graph.add_document("refund_policy.pdf", _evidence())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["edges"] = payload["edges"][1:]
    path.write_text(json.dumps(payload), encoding="utf-8")

    added = KnowledgeGraph(path).add_document("refund_policy.pdf", _evidence())

    assert added == {"documents": 0, "pages": 0, "chunks": 0, "kinds": 0}
    assert KnowledgeGraph(path).stats()["edges"] == 3


def test_graph_search_is_bounded_and_returns_evidence_subgraphs(tmp_path: Path):
    graph = KnowledgeGraph(tmp_path / "knowledge_graph.json")
    graph.add_document("refund_policy.pdf", _evidence())
    graph.add_document(
        "login_faq.pdf",
        [{"source": "login_faq.pdf", "page": 1, "chunk": 1, "kind": "faq", "text": "Reset login passwords from account settings."}],
    )

    result = graph.search("refund five", limit=999, max_nodes=999)

    assert result["count"] == 1
    assert result["limit"] == 20
    assert len(result["nodes"]) <= 100
    hit = result["hits"][0]
    assert hit["source"] == "refund_policy.pdf"
    assert hit["page"] == 2
    assert hit["kind"] == "policy"
    assert hit["chunk_id"].startswith("chunk_")
    assert "Refunds arrive" in hit["preview"]
    assert result["edges"]
    with pytest.raises(ValueError, match="query"):
        graph.search("   ")


def test_graph_serializes_concurrent_process_local_writers(tmp_path: Path):
    path = tmp_path / "knowledge_graph.json"
    first = KnowledgeGraph(path)
    second = KnowledgeGraph(path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: item[0].add_document(item[1], _evidence(item[1])),
                [(first, "a_policy.pdf"), (second, "b_policy.pdf")],
            )
        )

    assert [result["chunks"] for result in results] == [1, 1]
    assert KnowledgeGraph(path).stats()["documents"] == 2


def test_pdf_ingestion_optionally_builds_graph_with_matching_chunk_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pages = [
        SimpleNamespace(extract_text=lambda: "Eligible refunds arrive within five business days."),
        SimpleNamespace(extract_text=lambda: "Contact support for account assistance."),
    ]
    monkeypatch.setattr(ingestion, "PdfReader", lambda _: SimpleNamespace(pages=pages))
    product = LocalFaissStore(tmp_path / "product")
    support = LocalFaissStore(tmp_path / "support")
    graph = KnowledgeGraph(tmp_path / "knowledge_graph.json")

    report = ingest_pdf(
        b"safe-pdf-placeholder",
        filename="refund_policy.pdf",
        product_store=product,
        support_store=support,
        graph=graph,
    )

    graph_result = graph.search("refund five")
    assert report["graph_added"] == {"documents": 1, "pages": 2, "chunks": 2, "kinds": 1}
    assert graph_result["hits"][0]["chunk_id"] == support.records[0]["chunk_id"]
