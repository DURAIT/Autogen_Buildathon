from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal

from pypdf import PdfReader

from .rag import LocalFaissStore

if TYPE_CHECKING:
    from .knowledge_graph import KnowledgeGraph

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_EXTRACTED_CHARACTERS = 5_000_000
MAX_CHUNKS = 5_000


def _kind_from_filename(filename: str) -> str:
    lowered = filename.lower()
    for kind in ("faq", "catalog", "policy", "manual", "resolved"):
        if kind in lowered:
            return kind
    return "document"


def _chunks(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    parts: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        if end < len(clean):
            boundary = clean.rfind(". ", start + size // 2, end)
            if boundary > start:
                end = boundary + 1
        parts.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return parts


def ingest_pdf(
    source: str | Path | bytes | BinaryIO,
    *,
    filename: str | None,
    product_store: LocalFaissStore,
    support_store: LocalFaissStore,
    target: Literal["product", "support", "both"] = "both",
    graph: KnowledgeGraph | None = None,
) -> dict:
    if target not in {"product", "support", "both"}:
        raise ValueError("target must be product, support, or both")
    if isinstance(source, bytes):
        if len(source) > MAX_PDF_BYTES:
            raise ValueError("PDF exceeds the maximum allowed size of 20 MB")
        reader = PdfReader(BytesIO(source))
        safe_name = filename or "uploaded.pdf"
    else:
        if isinstance(source, (str, Path)):
            if Path(source).stat().st_size > MAX_PDF_BYTES:
                raise ValueError("PDF exceeds the maximum allowed size of 20 MB")
        elif hasattr(source, "seek") and hasattr(source, "tell"):
            original_position = source.tell()
            source.seek(0, 2)
            stream_size = source.tell()
            source.seek(original_position)
            if stream_size > MAX_PDF_BYTES:
                raise ValueError("PDF exceeds the maximum allowed size of 20 MB")
        reader = PdfReader(source)
        stream_name = getattr(source, "name", None)
        safe_name = filename or (Path(stream_name).name if stream_name else "uploaded.pdf")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the maximum allowed page count of {MAX_PDF_PAGES}")

    safe_name = Path(safe_name).name
    texts: list[str] = []
    metadata: list[dict] = []
    extracted_characters = 0
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        extracted_characters += len(page_text)
        if extracted_characters > MAX_EXTRACTED_CHARACTERS:
            raise ValueError("PDF extracted text exceeds the safe processing limit")
        for chunk_number, chunk in enumerate(_chunks(page_text), start=1):
            texts.append(chunk)
            metadata.append(
                {
                    "source": safe_name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "kind": _kind_from_filename(safe_name),
                }
            )
            if len(texts) > MAX_CHUNKS:
                raise ValueError("PDF creates too many chunks for safe processing")
    if not texts:
        raise ValueError(f"No extractable text found in {safe_name}; OCR the PDF first.")

    product_added = product_store.add_texts(texts, metadata) if target in {"product", "both"} else 0
    support_added = support_store.add_texts(texts, metadata) if target in {"support", "both"} else 0
    graph_added = (
        graph.add_document(
            safe_name,
            [{**item, "text": text} for text, item in zip(texts, metadata)],
        )
        if graph is not None
        else None
    )
    return {
        "filename": safe_name,
        "pages": len(reader.pages),
        "chunks": len(texts),
        "target": target,
        "product_added": product_added,
        "support_added": support_added,
        "graph_added": graph_added,
    }
