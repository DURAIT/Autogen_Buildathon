from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from support_system import rag
from support_system.agents import KnowledgeBaseAgent, KnowledgeSynthesis
from support_system.config import Settings
from support_system.models import Classification
from support_system.rag import LocalFaissStore


class ControlledEmbedder:
    dimension = 2
    version = "controlled-v1"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            if text.strip().lower() == "exactterm":
                vectors.append([1.0, 0.0])
            elif "exactterm" in text.lower():
                vectors.append([0.8, 0.6])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype="float32")


class ConstantEmbedder:
    def __init__(self, dimension: int, version: str) -> None:
        self.dimension = dimension
        self.version = version

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        vectors[:, 0] = 1.0
        return vectors


def test_legacy_records_receive_stable_chunk_ids_and_persist_migration(tmp_path: Path):
    directory = tmp_path / "legacy"
    directory.mkdir()
    legacy = {
        "text": "Eligible refunds arrive in five business days.",
        "content_hash": "a" * 64,
        "source": "refund.pdf",
        "kind": "policy",
    }
    (directory / "records.json").write_text(json.dumps([legacy]), encoding="utf-8")

    first = LocalFaissStore(directory)
    first_id = first.records[0]["chunk_id"]
    second = LocalFaissStore(directory)

    assert first_id == second.records[0]["chunk_id"]
    assert first_id.startswith("chunk_")
    assert json.loads((directory / "records.json").read_text(encoding="utf-8"))[0][
        "chunk_id"
    ] == first_id


def test_caller_cannot_override_chunk_identity_fields(tmp_path: Path):
    store = LocalFaissStore(tmp_path / "identity")

    added = store.add_texts(
        ["same passage", "same passage"],
        [
            {"source": "policy.pdf", "chunk_id": "attacker-one"},
            {"source": "policy.pdf", "chunk_id": "attacker-two"},
        ],
    )

    assert added == 1
    assert len(store.records) == 1
    assert store.records[0]["chunk_id"].startswith("chunk_")


def test_hybrid_search_uses_bm25_and_rrf_with_score_provenance(tmp_path: Path):
    store = LocalFaissStore(tmp_path / "hybrid", embedder=ControlledEmbedder())
    store.add_texts(
        ["General account assistance", "The exactterm refund rule"],
        [
            {"source": "general.pdf", "kind": "faq"},
            {"source": "exact.pdf", "kind": "policy"},
        ],
    )

    hits = store.search("exactterm", top_k=2, min_score=0)

    assert [hit["source"] for hit in hits] == ["exact.pdf", "general.pdf"]
    assert hits[0]["vector_rank"] == 2
    assert hits[0]["bm25_rank"] == 1
    assert hits[0]["bm25_score"] > 0
    assert hits[0]["fusion_score"] > hits[1]["fusion_score"]
    assert all(0 <= hit["score"] <= 1 for hit in hits)


def test_unrelated_vector_only_candidate_below_raw_threshold_is_excluded(tmp_path: Path):
    class BarelySimilarEmbedder:
        dimension = 2
        version = "barely-similar-v1"

        def embed(self, texts: list[str]) -> np.ndarray:
            vectors = []
            for text in texts:
                if text == "unrelated document":
                    vectors.append([0.01, np.sqrt(1 - 0.01**2)])
                else:
                    vectors.append([1.0, 0.0])
            return np.asarray(vectors, dtype="float32")

    store = LocalFaissStore(tmp_path / "relevance-gate", embedder=BarelySimilarEmbedder())
    store.add_texts(["unrelated document"], [{"source": "unrelated.pdf"}])

    assert store.search("exactterm", min_score=0.20) == []


def test_bm25_candidate_keeps_raw_vector_relevance_separate_from_fusion(tmp_path: Path):
    class LowVectorEmbedder:
        dimension = 2
        version = "low-vector-v1"

        def embed(self, texts: list[str]) -> np.ndarray:
            return np.asarray(
                [[1.0, 0.0] if text == "exactterm" else [0.01, np.sqrt(1 - 0.01**2)] for text in texts],
                dtype="float32",
            )

    store = LocalFaissStore(tmp_path / "lexical-gate", embedder=LowVectorEmbedder())
    store.add_texts(["exactterm policy"], [{"source": "policy.pdf"}])

    hit = store.search("exactterm", min_score=0.20)[0]

    assert hit["vector_rank"] is None
    assert hit["vector_score"] == 0.01
    assert hit["bm25_score"] > 0
    assert hit["score"] != hit["fusion_score"]


def test_bm25_candidate_below_final_relevance_threshold_is_excluded(tmp_path: Path):
    class OrthogonalEmbedder:
        dimension = 2
        version = "orthogonal-v1"

        def embed(self, texts: list[str]) -> np.ndarray:
            return np.asarray(
                [[1.0, 0.0] if text == "exactterm" else [0.0, 1.0] for text in texts],
                dtype="float32",
            )

    store = LocalFaissStore(tmp_path / "tiny-lexical-score", embedder=OrthogonalEmbedder())
    store.add_texts(
        [f"exactterm common document {index}" for index in range(10)],
        [{"source": f"document-{index}.pdf"} for index in range(10)],
    )

    assert store.search("exactterm", min_score=0.20) == []


def test_store_rebuilds_for_embedder_version_and_dimension(tmp_path: Path):
    directory = tmp_path / "compatibility"
    original = LocalFaissStore(directory, embedder=ConstantEmbedder(2, "semantic-v1"))
    original.add_texts(["refund policy"], [{"source": "policy.pdf"}])

    reloaded = LocalFaissStore(directory, embedder=ConstantEmbedder(3, "semantic-v2"))
    metadata = json.loads((directory / "embedding.version").read_text(encoding="utf-8"))

    assert reloaded.index.d == 3
    assert reloaded.index.ntotal == 1
    assert metadata == {"dimension": 3, "version": "semantic-v2"}


def test_openai_embedder_fingerprint_changes_with_normalized_provider_url():
    first = rag.OpenAICompatibleEmbedder(
        base_url="HTTPS://Provider-A.example/v1/",
        model="same-model",
        dimension=2,
        api_key="secret-a",
    )
    same = rag.OpenAICompatibleEmbedder(
        base_url="https://provider-a.example/v1",
        model="same-model",
        dimension=2,
        api_key="secret-b",
    )
    other = rag.OpenAICompatibleEmbedder(
        base_url="https://provider-b.example/v1",
        model="same-model",
        dimension=2,
        api_key="secret-a",
    )

    assert first.version == same.version
    assert first.version != other.version
    assert "provider-a.example" not in first.version
    assert "secret" not in first.version


def test_openai_compatible_embedder_is_configurable_without_model_downloads():
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps({"data": [{"index": 0, "embedding": [3.0, 4.0]}]}).encode()

    def requester(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    embedder = rag.OpenAICompatibleEmbedder(
        base_url="http://localhost:8080/v1",
        model="semantic-small",
        dimension=2,
        api_key="test-key",
        timeout=7,
        requester=requester,
    )

    vectors = embedder.embed(["refund policy"])

    assert captured == {
        "url": "http://localhost:8080/v1/embeddings",
        "authorization": "Bearer test-key",
        "payload": {"input": ["refund policy"], "model": "semantic-small"},
        "timeout": 7,
    }
    assert np.allclose(vectors, [[0.6, 0.8]])
    assert embedder.version.startswith("openai-compatible:semantic-small:2:v1:provider-")


async def test_synthesis_reranks_only_known_chunk_ids_and_appends_omissions():
    class FakeAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, task: str):
            self.calls += 1
            assert "chunk_a" in task and "chunk_b" in task and "chunk_c" in task
            synthesis = KnowledgeSynthesis(
                relevant=True,
                summary="Refund policy is relevant.",
                source_filenames=["a.pdf", "b.pdf"],
                ranked_chunk_ids=["chunk_b", "invented", "chunk_b"],
            )
            return SimpleNamespace(messages=[SimpleNamespace(content=synthesis)])

    knowledge = KnowledgeBaseAgent.__new__(KnowledgeBaseAgent)
    fake_agent = FakeAgent()
    knowledge.agent = fake_agent
    knowledge.rerank_enabled = True
    knowledge.last_synthesis = None
    classification = Classification(
        category="refund",
        urgency="low",
        sentiment="neutral",
        priority="low",
        summary="Refund timing",
        is_product_related=True,
        rationale="Policy match",
    )
    passages = [
        {"chunk_id": "chunk_a", "text": "A", "source": "a.pdf", "score": 0.8},
        {"chunk_id": "chunk_b", "text": "B", "source": "b.pdf", "score": 0.7},
        {"chunk_id": "chunk_c", "text": "C", "source": "c.pdf", "score": 0.6},
    ]

    reranked = await knowledge.synthesize("When is my refund?", classification, passages)

    assert [passage["chunk_id"] for passage in reranked] == [
        "chunk_b",
        "chunk_a",
        "chunk_c",
    ]
    assert fake_agent.calls == 1

    knowledge.rerank_enabled = False
    unchanged = await knowledge.synthesize("When is my refund?", classification, passages)

    assert unchanged == passages
    assert fake_agent.calls == 2


def test_settings_builds_matching_semantic_or_hashing_embedder(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai-compatible")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("EMBEDDING_API_KEY", "local-only")
    monkeypatch.setenv("ENABLE_LLM_RERANK", "false")

    settings = Settings.from_env()
    semantic = settings.build_embedder()

    assert settings.llm_rerank_enabled is False
    assert isinstance(semantic, rag.OpenAICompatibleEmbedder)
    assert semantic.dimension == 768
    assert semantic.version.startswith(
        "openai-compatible:nomic-embed-text:768:v1:provider-"
    )

    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    hashing = Settings.from_env().build_embedder()
    assert isinstance(hashing, rag.HashingEmbedder)
