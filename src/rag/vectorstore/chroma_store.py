"""Chroma-backed persistent vector store."""

from __future__ import annotations

import json
import logging
from typing import Any

from rag.core.exceptions import VectorStoreError
from rag.core.interfaces import VectorStore
from rag.core.models import Chunk, ChunkMetadata, SearchResult, SourceType

logger = logging.getLogger(__name__)


def _metadata_to_chroma(meta: ChunkMetadata) -> dict[str, Any]:
    """Flatten ChunkMetadata into Chroma's flat, JSON-primitive-only schema."""
    flat: dict[str, Any] = {
        "source_path": meta.source_path,
        "source_type": meta.source_type.value,
        "document_id": meta.document_id,
        "chunk_index": meta.chunk_index,
        "start_char": meta.start_char,
        "end_char": meta.end_char,
    }
    if meta.title is not None:
        flat["title"] = meta.title
    if meta.page_number is not None:
        flat["page_number"] = meta.page_number
    if meta.extra:
        flat["extra_json"] = json.dumps(meta.extra)
    return flat


def _metadata_from_chroma(flat: dict[str, Any]) -> ChunkMetadata:
    extra = json.loads(flat["extra_json"]) if flat.get("extra_json") else {}
    return ChunkMetadata(
        source_path=flat["source_path"],
        source_type=SourceType(flat["source_type"]),
        document_id=flat["document_id"],
        chunk_index=flat["chunk_index"],
        title=flat.get("title"),
        page_number=flat.get("page_number"),
        start_char=flat.get("start_char", 0),
        end_char=flat.get("end_char", 0),
        extra=extra,
    )


class ChromaVectorStore(VectorStore):
    """Persists chunk embeddings in a local/embedded Chroma collection."""

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        persist_dir: str = "./data/chroma",
        distance_metric: str = "cosine",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb is required for ChromaVectorStore. Install with: pip install chromadb"
            ) from exc

        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_metric},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize Chroma collection: {exc}") from exc

        self.collection_name = collection_name
        self.persist_dir = persist_dir

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk in chunks:
            if chunk.embedding is None:
                raise VectorStoreError(
                    f"Chunk {chunk.id} has no embedding; embed before upserting"
                )
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.content)
            metadatas.append(_metadata_to_chroma(chunk.metadata))

        try:
            self._collection.upsert(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma upsert failed: {exc}") from exc

        logger.info("Upserted %d chunks into Chroma collection '%s'", len(chunks), self.collection_name)

    def query(
        self, embedding: list[float], top_k: int = 5, filters: dict | None = None
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise VectorStoreError("top_k must be positive")

        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        search_results: list[SearchResult] = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
            chunk = Chunk(id=cid, content=doc, metadata=_metadata_from_chroma(meta))
            similarity = 1.0 - dist  # cosine distance -> similarity
            search_results.append(
                SearchResult(chunk=chunk, score=similarity, dense_score=similarity, rank=rank)
            )
        return search_results

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        try:
            self._collection.delete(ids=chunk_ids)
        except Exception as exc:
            raise VectorStoreError(f"Chroma delete failed: {exc}") from exc

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Chroma count failed: {exc}") from exc

    def get_all(self) -> list[Chunk]:
        """Fetch every stored chunk (content + metadata, no embeddings) —
        used to rebuild in-memory sparse indexes after a process restart.
        """
        try:
            result = self._collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Chroma get_all failed: {exc}") from exc

        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        return [
            Chunk(id=cid, content=doc, metadata=_metadata_from_chroma(meta))
            for cid, doc, meta in zip(ids, docs, metas)
        ]
