# Local retrieval helpers for knowledge files and optional CSV data.
from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Iterable

from app.config import get_settings


# Keep knowledge and FAISS files in predictable local folders.
KNOWLEDGE_DIR = Path("data/knowledge")
VECTOR_STORE_DIR = Path("data/vectorstore/faiss")

# Restrict specialist agents to the knowledge files that fit their topic.
AGENT_SOURCES = {
    "health_agent": {"eldercare_safety.md", "fall_prevention.md"},
    "medication_agent": {"medication_safety.md"},
    "fraud_agent": {"scam_prevention.md"},
    "companion_agent": {"emotional_support.md"},
}


# Deterministic embedding model that avoids external embedding API calls.
class LocalHashEmbeddings:
    """Small deterministic embedding model so RAG works without paid APIs."""

    dimension = 384

    # Convert text into a normalized signed hashing vector.
    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            # Hash each token to a stable index and sign.
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        # Normalize vectors so cosine similarity behaves consistently.
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    # Embed document chunks for FAISS indexing.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    # Embed a single query for similarity search.
    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# Load markdown knowledge and optional CSV rows into LangChain documents.
def _load_documents():
    # Import lazily so app startup works before RAG dependencies are installed.
    from langchain_community.document_loaders import CSVLoader, DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from app.kaggle_data import get_csv_files

    documents = []

    # Add curated local safety knowledge first.
    if KNOWLEDGE_DIR.exists():
        loader = DirectoryLoader(
            str(KNOWLEDGE_DIR),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        documents.extend(loader.load())

    # Add optional Kaggle CSV content when datasets have been downloaded.
    for csv_file in get_csv_files():
        try:
            documents.extend(CSVLoader(str(csv_file), encoding="utf-8").load())
        except Exception:
            # Skip bad CSVs so one data file does not disable RAG.
            continue

    # Empty document sets mean the vector store cannot be built yet.
    if not documents:
        return []

    # Split long documents so retrieval can return focused snippets.
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    return splitter.split_documents(documents)


# Load the saved FAISS index if it has already been built.
def _load_vector_store():
    from langchain_community.vectorstores import FAISS

    # Missing index is normal on first run.
    if not VECTOR_STORE_DIR.exists():
        return None
    # Local files are trusted because this app creates the vector store itself.
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        LocalHashEmbeddings(),
        allow_dangerous_deserialization=True,
    )


# Build the local FAISS vector store from current knowledge documents.
def build_vector_store() -> dict[str, object]:
    """Build or rebuild the local FAISS vector store from data/knowledge."""

    # Vietnamese note: RAG dung knowledge noi bo/local data, khong goi embedding API ben ngoai.
    # Import here so dependency errors can be returned as admin-friendly messages.
    try:
        from langchain_community.vectorstores import FAISS
    except Exception as exc:
        return {
            "ok": False,
            "message": f"LangChain/FAISS dependencies are not installed: {exc}",
            "documents": 0,
            "path": str(VECTOR_STORE_DIR),
        }

    # No documents means the admin should add knowledge files or CSV data first.
    documents = _load_documents()
    if not documents:
        return {
            "ok": False,
            "message": "No knowledge documents found in data/knowledge.",
            "documents": 0,
            "path": str(VECTOR_STORE_DIR),
        }

    # Rebuild from scratch so stale chunks are removed.
    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)
    VECTOR_STORE_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Save the FAISS index locally for fast retrieval on later turns.
    store = FAISS.from_documents(documents, LocalHashEmbeddings())
    store.save_local(str(VECTOR_STORE_DIR))
    return {
        "ok": True,
        "message": "Vector store rebuilt.",
        "documents": len(documents),
        "path": str(VECTOR_STORE_DIR),
    }


# Retrieve general context snippets for a query.
def retrieve_context(query: str) -> str:
    """Return relevant knowledge snippets, or an empty string if RAG is unavailable."""

    # Vietnamese note: Neu vector store chua san sang, app van chay o che do an toan.
    try:
        # Build the vector store on demand if it does not exist yet.
        store = _load_vector_store()
        if store is None:
            result = build_vector_store()
            if not result.get("ok"):
                return ""
            store = _load_vector_store()
            if store is None:
                return ""

        # Return the top few chunks in a compact prompt-ready format.
        docs = store.similarity_search(query, k=4)
        return _format_documents(docs)
    except Exception:
        # RAG is optional, so failures should not stop the chat workflow.
        return ""


# Retrieve context limited to the files allowed for a specialist agent.
def get_agent_context(agent_name: str, query: str) -> str:
    """Retrieve context for a specialist agent from its allowed knowledge files."""

    # Agents without a source restriction use the general retriever.
    allowed_sources = AGENT_SOURCES.get(agent_name)
    if not allowed_sources:
        return retrieve_context(query)

    try:
        # Build or load the vector store before filtering by source.
        store = _load_vector_store()
        if store is None:
            result = build_vector_store()
            if not result.get("ok"):
                return ""
            store = _load_vector_store()
            if store is None:
                return ""

        # Search more chunks, then keep only matching specialist sources.
        docs = store.similarity_search(query, k=8)
        filtered = [
            doc
            for doc in docs
            if Path(str(doc.metadata.get("source", ""))).name in allowed_sources
            or "data/raw" in str(doc.metadata.get("source", "")).replace("\\", "/")
        ]
        return _format_documents(filtered[:4])
    except Exception:
        # Missing RAG context should degrade gracefully.
        return ""


# Format retrieved documents with their source filename for explainability.
def _format_documents(documents: Iterable[object]) -> str:
    parts: list[str] = []
    for doc in documents:
        # Collapse whitespace so snippets fit cleanly inside prompts and traces.
        source = Path(str(doc.metadata.get("source", "knowledge"))).name
        content = " ".join(str(doc.page_content).split())
        parts.append(f"[{source}] {content}")
    return "\n\n".join(parts)
