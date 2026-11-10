from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import faiss
import numpy as np
from filelock import FileLock

RRF_K = 60


class Embedder(Protocol):
    dimension: int
    version: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic lexical hashing (not semantic embedding) for offline retrieval."""

    def __init__(self, dimension: int = 2048) -> None:
        self.dimension = dimension
        self.version = f"hashing-v3-positive-stemmed-{dimension}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens: list[str] = []
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if len(token) > 4 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            tokens.append(token)
        return tokens

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            tokens = self._tokens(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "little") % self.dimension
                vectors[row, column] += 1.0
        faiss.normalize_L2(vectors)
        return vectors


class OpenAICompatibleEmbedder:
    """Semantic embedder backed by an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimension: int,
        api_key: str = "",
        timeout: float = 30.0,
        embedding_version: str | None = None,
        requester: Callable[..., Any] | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        parsed_url = urlsplit(base_url.strip())
        self.base_url = urlunsplit(
            (parsed_url.scheme.lower(), parsed_url.netloc.lower(), parsed_url.path.rstrip("/"), "", "")
        )
        self.model = model
        self.dimension = dimension
        self.api_key = api_key
        self.timeout = timeout
        provider_fingerprint = hashlib.sha256(self.base_url.encode("utf-8")).hexdigest()[:12]
        version_label = embedding_version or "v1"
        self.version = (
            f"openai-compatible:{model}:{dimension}:{version_label}:provider-{provider_fingerprint}"
        )
        self._requester = requester or urlopen

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        payload = json.dumps({"input": texts, "model": self.model}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers=headers,
            method="POST",
        )
        with self._requester(request, timeout=self.timeout) as response:
            body = json.loads(response.read())
        data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
        vectors = np.asarray([item["embedding"] for item in data], dtype="float32")
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(
                "Embedding endpoint returned an incompatible count or dimension: "
                f"expected {(len(texts), self.dimension)}, got {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding endpoint returned non-finite values")
        faiss.normalize_L2(vectors)
        return vectors


class LocalSentenceTransformerEmbedder:
    """Fully local, offline, no-API-key semantic embedder.

    Loads ``sentence-transformers/all-MiniLM-L6-v2`` once per process (cached on the
    instance) and produces 384-dimensional sentence embeddings without any network
    call or API key, matching the ``Embedder`` protocol used by ``LocalFaissStore``.
    """

    _MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    _MODEL_DIMENSION = 384
    _shared_model: Any = None
    _shared_model_name: str | None = None

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self._MODEL_NAME
        # The configured EMBEDDING_DIMENSION is intentionally ignored: this backend
        # always reports its own fixed, actual model dimension, following the same
        # self-describing `.dimension` pattern as HashingEmbedder/OpenAICompatibleEmbedder.
        self.dimension = self._MODEL_DIMENSION
        self.version = f"local-sentence-transformer:{self.model_name}:{self.dimension}"
        self._model = self._load_model(self.model_name)

    @classmethod
    def _load_model(cls, model_name: str) -> Any:
        # Cache the loaded model at class level so repeated LocalFaissStore/embedder
        # construction within a process never reloads model weights from disk.
        if cls._shared_model is None or cls._shared_model_name != model_name:
            from sentence_transformers import SentenceTransformer

            cls._shared_model = SentenceTransformer(model_name)
            cls._shared_model_name = model_name
        return cls._shared_model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        vectors = np.asarray(
            self._model.encode(texts, convert_to_numpy=True), dtype="float32"
        )
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(
                "Local sentence-transformer model returned an unexpected shape: "
                f"expected {(len(texts), self.dimension)}, got {vectors.shape}"
            )
        # Normalize consistent with HashingEmbedder/OpenAICompatibleEmbedder, which
        # both L2-normalize so inner-product FAISS search behaves as cosine similarity.
        faiss.normalize_L2(vectors)
        return vectors


class LocalFaissStore:
    """Persistent FAISS index with synchronized, transactional local writes."""

    def __init__(self, directory: str | Path, embedder: Embedder | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self.index_path = self.directory / "index.faiss"
        self.records_path = self.directory / "records.json"
        self.version_path = self.directory / "embedding.version"
        self._thread_lock = RLock()
        self._file_lock = FileLock(str(self.directory / ".store.lock"))
        self.records: list[dict[str, Any]] = []
        self._hashes: set[str] = set()
        self.index = faiss.IndexFlatIP(self.embedder.dimension)
        with self._file_lock:
            self._load()

    def _load(self) -> None:
        self.records = []
        self._hashes = set()
        self.index = faiss.IndexFlatIP(self.embedder.dimension)
        migrated_records = False
        if self.records_path.exists():
            loaded = json.loads(self.records_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError(f"Invalid records file: {self.records_path}")
            self.records = loaded
            for record in self.records:
                if not record.get("chunk_id"):
                    record["chunk_id"] = self._chunk_id(record["content_hash"])
                    migrated_records = True
            self._hashes = {record["content_hash"] for record in self.records}
        stored_metadata: dict[str, Any] = {}
        if self.version_path.exists():
            raw_metadata = self.version_path.read_text(encoding="utf-8").strip()
            try:
                parsed_metadata = json.loads(raw_metadata)
                if isinstance(parsed_metadata, dict):
                    stored_metadata = parsed_metadata
            except json.JSONDecodeError:
                stored_metadata = {"version": raw_metadata}
        expected_metadata = {
            "dimension": self.embedder.dimension,
            "version": self.embedder.version,
        }
        needs_rebuild = bool(
            self.records
            and (not self.index_path.exists() or stored_metadata != expected_metadata)
        )
        if self.index_path.exists() and not needs_rebuild:
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception:
                needs_rebuild = True
        if self.index.d != self.embedder.dimension or self.index.ntotal != len(self.records):
            needs_rebuild = True
        if needs_rebuild:
            self.index = faiss.IndexFlatIP(self.embedder.dimension)
            if self.records:
                self.index.add(self.embedder.embed([record["text"] for record in self.records]))
            self._save()
        elif migrated_records:
            self._save()

    @staticmethod
    def _chunk_id(content_hash: str) -> str:
        return f"chunk_{content_hash[:24]}"

    def _save(self) -> None:
        index_tmp = self.index_path.with_suffix(".faiss.tmp")
        records_tmp = self.records_path.with_suffix(".json.tmp")
        version_tmp = self.version_path.with_suffix(".version.tmp")
        faiss.write_index(self.index, str(index_tmp))
        records_tmp.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        version_tmp.write_text(
            json.dumps(
                {"dimension": self.embedder.dimension, "version": self.embedder.version},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        index_tmp.replace(self.index_path)
        records_tmp.replace(self.records_path)
        version_tmp.replace(self.version_path)

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]]) -> int:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have equal lengths")
        with self._thread_lock, self._file_lock:
            self._load()
            pending_texts: list[str] = []
            pending_records: list[dict[str, Any]] = []
            for text, metadata in zip(texts, metadatas):
                clean_text = " ".join(text.split())
                if not clean_text:
                    continue
                safe_metadata = {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"text", "content_hash", "chunk_id"}
                }
                identity = hashlib.sha256(
                    (clean_text + json.dumps(safe_metadata, sort_keys=True, default=str)).encode(
                        "utf-8"
                    )
                ).hexdigest()
                if identity in self._hashes:
                    continue
                pending_texts.append(clean_text)
                pending_records.append(
                    {
                        **safe_metadata,
                        "text": clean_text,
                        "content_hash": identity,
                        "chunk_id": self._chunk_id(identity),
                    }
                )
                self._hashes.add(identity)
            if not pending_texts:
                return 0
            self.index.add(self.embedder.embed(pending_texts))
            self.records.extend(pending_records)
            self._save()
            return len(pending_texts)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.20) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        with self._thread_lock, self._file_lock:
            self._load()
            if self.index.ntotal == 0:
                return []
            count = min(max(top_k, 1), self.index.ntotal)
            vector_scores, vector_indexes = self.index.search(
                self.embedder.embed([query]), self.index.ntotal
            )
            raw_vector_values = {
                int(index): float(score)
                for score, index in zip(vector_scores[0], vector_indexes[0])
                if index >= 0
            }
            vector_values = {
                index: score
                for index, score in raw_vector_values.items()
                if score >= min_score
            }
            vector_order = sorted(
                vector_values,
                key=lambda index: (-vector_values[index], self.records[index]["chunk_id"]),
            )
            vector_ranks = {index: rank for rank, index in enumerate(vector_order, start=1)}

            bm25_values = self._bm25_scores(query)
            bm25_order = sorted(
                (index for index, score in enumerate(bm25_values) if score > 0),
                key=lambda index: (-bm25_values[index], self.records[index]["chunk_id"]),
            )
            bm25_ranks = {index: rank for rank, index in enumerate(bm25_order, start=1)}

            candidates = set(vector_ranks) | set(bm25_ranks)
            ranked: list[tuple[float, str, int]] = []
            for index in candidates:
                fusion_score = 0.0
                if index in vector_ranks:
                    fusion_score += 1.0 / (RRF_K + vector_ranks[index])
                if index in bm25_ranks:
                    fusion_score += 1.0 / (RRF_K + bm25_ranks[index])
                ranked.append((fusion_score, self.records[index]["chunk_id"], index))
            ranked.sort(key=lambda item: (-item[0], item[1]))

            hits: list[dict[str, Any]] = []
            for fusion_score, _, index in ranked:
                vector_relevance = min(max(raw_vector_values.get(index, 0.0), 0.0), 1.0)
                lexical_relevance = bm25_values[index] / (1.0 + bm25_values[index])
                score = max(vector_relevance, lexical_relevance)
                if score < min_score:
                    continue
                hits.append(
                    {
                        **self.records[index],
                        "vector_score": round(raw_vector_values.get(index, 0.0), 6),
                        "vector_rank": vector_ranks.get(index),
                        "bm25_score": round(bm25_values[index], 6),
                        "bm25_rank": bm25_ranks.get(index),
                        "fusion_score": round(fusion_score, 8),
                        "score": round(score, 6),
                    }
                )
                if len(hits) == count:
                    break
            return hits

    def _bm25_scores(self, query: str, *, k1: float = 1.5, b: float = 0.75) -> list[float]:
        documents = [HashingEmbedder._tokens(record["text"]) for record in self.records]
        query_terms = HashingEmbedder._tokens(query)
        if not documents or not query_terms:
            return [0.0] * len(documents)
        average_length = sum(map(len, documents)) / len(documents)
        document_frequency = Counter(
            term for document in documents for term in set(document)
        )
        query_frequency = Counter(query_terms)
        scores: list[float] = []
        for document in documents:
            term_frequency = Counter(document)
            score = 0.0
            for term, query_count in query_frequency.items():
                frequency = term_frequency.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(
                    1.0
                    + (len(documents) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                length_ratio = len(document) / average_length if average_length else 0.0
                score += query_count * idf * (
                    frequency * (k1 + 1.0)
                    / (frequency + k1 * (1.0 - b + b * length_ratio))
                )
            scores.append(score)
        return scores
