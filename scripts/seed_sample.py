from __future__ import annotations

from pathlib import Path

from support_system.rag import LocalFaissStore

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT / "data"
text = (ROOT / "knowledge" / "sample_product_support.md").read_text(encoding="utf-8")
metadata = [{"source": "sample_product_support.md", "page": 1, "kind": "sample"}]
product = LocalFaissStore(data_dir / "product_scope_faiss")
support = LocalFaissStore(data_dir / "support_knowledge_faiss")
print({"product_added": product.add_texts([text], metadata), "support_added": support.add_texts([text], metadata)})
