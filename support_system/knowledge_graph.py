from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import networkx as nx
from filelock import FileLock

_FORMAT = "supportiq-provenance-v1"
_MAX_PREVIEW = 240
_MAX_RESULTS = 20
_MAX_SUBGRAPH_NODES = 100
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _sanitize_preview(value: object) -> str:
    text = _clean_text(value)
    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", text, flags=re.I)
    text = re.sub(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)", "[PHONE]", text)
    text = re.sub(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", "[PAYMENT_NUMBER]", text)
    text = re.sub(
        r"(?i)\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*[^\s,;]+",
        "[SECRET]",
        text,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{16,}\b", "[SECRET]", text)
    return text[:_MAX_PREVIEW]


def _digest(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


class KnowledgeGraph:
    """Persistent read-mostly provenance index; source documents remain the truth source."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = _thread_lock(self.path)
        self._file_lock = FileLock(str(self.path) + ".lock")
        self._graph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        with self._thread_lock, self._file_lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        graph = nx.DiGraph()
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
                raise ValueError(f"Invalid knowledge graph format: {self.path}")
            nodes = payload.get("nodes")
            edges = payload.get("edges")
            if not isinstance(nodes, list) or not isinstance(edges, list):
                raise ValueError(f"Invalid knowledge graph data: {self.path}")
            for node in nodes:
                if not isinstance(node, dict) or not isinstance(node.get("id"), str):
                    raise ValueError(f"Invalid knowledge graph node: {self.path}")
                attributes = {key: value for key, value in node.items() if key != "id"}
                graph.add_node(node["id"], **attributes)
            for edge in edges:
                if not isinstance(edge, dict) or not all(
                    isinstance(edge.get(key), str) for key in ("source_id", "target_id", "relation")
                ):
                    raise ValueError(f"Invalid knowledge graph edge: {self.path}")
                attributes = {
                    key: value for key, value in edge.items() if key not in {"source_id", "target_id"}
                }
                graph.add_edge(edge["source_id"], edge["target_id"], **attributes)
        self._graph = graph

    def _save_unlocked(self) -> None:
        payload = {
            "format": _FORMAT,
            "nodes": [
                {"id": node_id, **attributes}
                for node_id, attributes in sorted(self._graph.nodes(data=True))
            ],
            "edges": [
                {"source_id": source, "target_id": target, **attributes}
                for source, target, attributes in sorted(
                    self._graph.edges(data=True), key=lambda item: (item[0], item[1])
                )
            ],
        }
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def add_document(self, source: str, chunks: Iterable[dict[str, Any]]) -> dict[str, int]:
        safe_source = Path(_clean_text(source)).name
        if not safe_source:
            raise ValueError("source is required")
        evidence = list(chunks)
        if not evidence:
            raise ValueError("at least one chunk is required")
        added = Counter({"documents": 0, "pages": 0, "chunks": 0, "kinds": 0})
        repaired_edge = False
        with self._thread_lock, self._file_lock:
            self._load_unlocked()
            document_id = _digest("document", safe_source)
            if document_id not in self._graph:
                self._graph.add_node(document_id, type="Document", source=safe_source)
                added["documents"] += 1
            for item in evidence:
                page = int(item["page"])
                if page < 1:
                    raise ValueError("page must be positive")
                kind = _clean_text(item.get("kind") or "document")[:80]
                text = _clean_text(item.get("text"))
                if not text:
                    continue
                metadata = {
                    "source": safe_source,
                    "page": page,
                    "chunk": int(item.get("chunk", 1)),
                    "kind": kind,
                }
                identity = hashlib.sha256(
                    (text + json.dumps(metadata, sort_keys=True, default=str)).encode("utf-8")
                ).hexdigest()
                chunk_id = f"chunk_{identity[:24]}"
                page_id = _digest("page", f"{safe_source}\0{page}")
                kind_id = _digest("kind", kind.casefold())
                if page_id not in self._graph:
                    self._graph.add_node(page_id, type="Page", source=safe_source, page=page)
                    added["pages"] += 1
                if kind_id not in self._graph:
                    self._graph.add_node(kind_id, type="Kind", kind=kind)
                    added["kinds"] += 1
                if chunk_id not in self._graph:
                    self._graph.add_node(
                        chunk_id,
                        type="Chunk",
                        source=safe_source,
                        page=page,
                        chunk_id=chunk_id,
                        kind=kind,
                        preview=_sanitize_preview(text),
                    )
                    added["chunks"] += 1
                edge_evidence = {
                    "source": safe_source,
                    "page": page,
                    "chunk_id": chunk_id,
                    "kind": kind,
                }
                for source_id, target_id, relation in (
                    (document_id, page_id, "HAS_PAGE"),
                    (page_id, chunk_id, "HAS_CHUNK"),
                    (chunk_id, kind_id, "HAS_KIND"),
                ):
                    repaired_edge = repaired_edge or not self._graph.has_edge(source_id, target_id)
                    self._graph.add_edge(
                        source_id, target_id, relation=relation, **edge_evidence
                    )
            if any(added.values()) or repaired_edge:
                self._save_unlocked()
        return dict(added)

    def stats(self) -> dict[str, int]:
        self._load()
        counts = Counter(attributes.get("type") for _, attributes in self._graph.nodes(data=True))
        return {
            "documents": counts["Document"],
            "pages": counts["Page"],
            "chunks": counts["Chunk"],
            "kinds": counts["Kind"],
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }

    def search(self, query: str, limit: int = 10, max_nodes: int = 50) -> dict[str, Any]:
        terms = set(re.findall(r"[\w-]+", _clean_text(query).casefold()))
        if not terms:
            raise ValueError("query is required")
        safe_limit = min(max(int(limit), 1), _MAX_RESULTS)
        safe_max_nodes = min(max(int(max_nodes), 1), _MAX_SUBGRAPH_NODES)
        self._load()
        matches: list[tuple[int, str, dict[str, Any]]] = []
        for node_id, attributes in self._graph.nodes(data=True):
            if attributes.get("type") != "Chunk":
                continue
            searchable = " ".join(
                str(attributes.get(key, "")) for key in ("source", "kind", "preview")
            ).casefold()
            score = sum(searchable.count(term) for term in terms)
            if score:
                matches.append((score, node_id, attributes))
        matches.sort(key=lambda item: (-item[0], item[1]))
        selected = matches[:safe_limit]
        node_ids: list[str] = []
        for _, chunk_id, _ in selected:
            related = set(self._graph.predecessors(chunk_id)) | set(self._graph.successors(chunk_id))
            for predecessor in self._graph.predecessors(chunk_id):
                related.update(self._graph.predecessors(predecessor))
            for candidate in [chunk_id, *sorted(related)]:
                if candidate not in node_ids and len(node_ids) < safe_max_nodes:
                    node_ids.append(candidate)
        selected_nodes = set(node_ids)
        return {
            "query": _clean_text(query),
            "count": len(selected),
            "limit": safe_limit,
            "hits": [
                {
                    "source": attributes["source"],
                    "page": attributes["page"],
                    "chunk_id": attributes["chunk_id"],
                    "kind": attributes["kind"],
                    "preview": attributes["preview"],
                    "match_score": score,
                }
                for score, _, attributes in selected
            ],
            "nodes": [
                {"id": node_id, **self._graph.nodes[node_id]} for node_id in node_ids
            ],
            "edges": [
                {"source_id": source, "target_id": target, **attributes}
                for source, target, attributes in self._graph.edges(data=True)
                if source in selected_nodes and target in selected_nodes
            ],
        }
