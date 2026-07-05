from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Iterable

from app.config import get_settings


KNOWLEDGE_DIR = Path("data/knowledge")
VECTOR_STORE_DIR = Path("data/vectorstore/faiss")

AGENT_SOURCES = {
    "health_agent": {"eldercare_safety.md", "fall_prevention.md"},
    "medication_agent": {"medication_safety.md"},
    "fraud_agent": {"scam_prevention.md"},
    "companion_agent": {"emotional_support.md"},
}


class LocalHashEmbeddings:
    """Small deterministic embedding model so RAG works without paid APIs."""

    dimension = 384

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _load_documents():
    from langchain_community.document_loaders import CSVLoader, DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from app.kaggle_data import get_csv_files

    documents = []

    if KNOWLEDGE_DIR.exists():
        loader = DirectoryLoader(
            str(KNOWLEDGE_DIR),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        documents.extend(loader.load())

    for csv_file in get_csv_files():
        try:
            documents.extend(CSVLoader(str(csv_file), encoding="utf-8").load())
        except Exception:
            continue

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    return splitter.split_documents(documents)


def _load_vector_store():
    from langchain_community.vectorstores import FAISS

    if not VECTOR_STORE_DIR.exists():
        return None
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        LocalHashEmbeddings(),
        allow_dangerous_deserialization=True,
    )


def build_vector_store() -> dict[str, object]:
    """Build or rebuild the local FAISS vector store from data/knowledge."""

    # Vietnamese note: RAG dung knowledge noi bo/local data, khong goi embedding API ben ngoai.
    try:
        from langchain_community.vectorstores import FAISS
    except Exception as exc:
        return {
            "ok": False,
            "message": f"LangChain/FAISS dependencies are not installed: {exc}",
            "documents": 0,
            "path": str(VECTOR_STORE_DIR),
        }

    documents = _load_documents()
    if not documents:
        return {
            "ok": False,
            "message": "No knowledge documents found in data/knowledge.",
            "documents": 0,
            "path": str(VECTOR_STORE_DIR),
        }

    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)
    VECTOR_STORE_DIR.parent.mkdir(parents=True, exist_ok=True)

    store = FAISS.from_documents(documents, LocalHashEmbeddings())
    store.save_local(str(VECTOR_STORE_DIR))
    return {
        "ok": True,
        "message": "Vector store rebuilt.",
        "documents": len(documents),
        "path": str(VECTOR_STORE_DIR),
    }


def retrieve_context(query: str) -> str:
    """Return relevant knowledge snippets, or an empty string if RAG is unavailable."""

    # Vietnamese note: Neu vector store chua san sang, app van chay o che do an toan.
    try:
        store = _load_vector_store()
        if store is None:
            result = build_vector_store()
            if not result.get("ok"):
                return ""
            store = _load_vector_store()
            if store is None:
                return ""

        docs = store.similarity_search(query, k=4)
        return _format_documents(docs)
    except Exception:
        return ""


def get_agent_context(agent_name: str, query: str) -> str:
    """Retrieve context for a specialist agent from its allowed knowledge files."""

    allowed_sources = AGENT_SOURCES.get(agent_name)
    if not allowed_sources:
        return retrieve_context(query)

    try:
        store = _load_vector_store()
        if store is None:
            result = build_vector_store()
            if not result.get("ok"):
                return ""
            store = _load_vector_store()
            if store is None:
                return ""

        docs = store.similarity_search(query, k=8)
        filtered = [
            doc
            for doc in docs
            if Path(str(doc.metadata.get("source", ""))).name in allowed_sources
            or "data/raw" in str(doc.metadata.get("source", "")).replace("\\", "/")
        ]
        return _format_documents(filtered[:4])
    except Exception:
        return ""


def _format_documents(documents: Iterable[object]) -> str:
    parts: list[str] = []
    for doc in documents:
        source = Path(str(doc.metadata.get("source", "knowledge"))).name
        content = " ".join(str(doc.page_content).split())
        parts.append(f"[{source}] {content}")
    return "\n\n".join(parts)
