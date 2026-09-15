"""Insertion and retrieval tests for the Chroma-backed vector store."""

from __future__ import annotations

import pytest

from rag.core.exceptions import VectorStoreError
from rag.core.models import Chunk, ChunkMetadata, SourceType
from rag.vectorstore.chroma_store import ChromaVectorStore


def make_chunk(text: str, embedding: list[float], index: int = 0) -> Chunk:
    return Chunk(
        content=text,
        embedding=embedding,
        metadata=ChunkMetadata(
            source_path="doc.txt",
            source_type=SourceType.TEXT,
            document_id="doc-1",
            chunk_index=index,
        ),
    )


@pytest.fixture
def store(chroma_persist_dir: str) -> ChromaVectorStore:
    return ChromaVectorStore(collection_name="test_collection", persist_dir=chroma_persist_dir)


class TestUpsert:
    def test_upsert_increases_count(self, store: ChromaVectorStore) -> None:
        assert store.count() == 0
        chunks = [make_chunk("hello world", [1.0, 0.0, 0.0], i) for i in range(3)]
        store.upsert(chunks)
        assert store.count() == 3

    def test_upsert_without_embedding_raises(self, store: ChromaVectorStore) -> None:
        chunk = Chunk(
            content="no embedding",
            metadata=ChunkMetadata(
                source_path="doc.txt",
                source_type=SourceType.TEXT,
                document_id="doc-1",
                chunk_index=0,
            ),
        )
        with pytest.raises(VectorStoreError):
            store.upsert([chunk])

    def test_upsert_empty_list_is_noop(self, store: ChromaVectorStore) -> None:
        store.upsert([])
        assert store.count() == 0

    def test_upsert_same_id_updates_not_duplicates(self, store: ChromaVectorStore) -> None:
        chunk = make_chunk("version one", [1.0, 0.0, 0.0])
        store.upsert([chunk])
        chunk.content = "version two"
        store.upsert([chunk])
        assert store.count() == 1


class TestQuery:
    def test_query_returns_nearest_neighbor_first(self, store: ChromaVectorStore) -> None:
        close = make_chunk("close match", [1.0, 0.0, 0.0], 0)
        far = make_chunk("far match", [0.0, 1.0, 0.0], 1)
        store.upsert([close, far])

        results = store.query([0.9, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].chunk.id == close.id
        assert results[0].score >= results[1].score

    def test_query_respects_top_k(self, store: ChromaVectorStore) -> None:
        chunks = [make_chunk(f"chunk {i}", [1.0, float(i) * 0.01, 0.0], i) for i in range(10)]
        store.upsert(chunks)
        results = store.query([1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_query_rejects_non_positive_top_k(self, store: ChromaVectorStore) -> None:
        with pytest.raises(VectorStoreError):
            store.query([1.0, 0.0, 0.0], top_k=0)

    def test_query_on_empty_store_returns_empty(self, store: ChromaVectorStore) -> None:
        results = store.query([1.0, 0.0, 0.0], top_k=5)
        assert results == []


class TestDeleteAndGetAll:
    def test_delete_removes_chunk(self, store: ChromaVectorStore) -> None:
        chunk = make_chunk("to delete", [1.0, 0.0, 0.0])
        store.upsert([chunk])
        assert store.count() == 1
        store.delete([chunk.id])
        assert store.count() == 0

    def test_get_all_round_trips_content_and_metadata(self, store: ChromaVectorStore) -> None:
        chunk = make_chunk("round trip me", [1.0, 0.0, 0.0], index=2)
        store.upsert([chunk])
        fetched = store.get_all()
        assert len(fetched) == 1
        assert fetched[0].content == "round trip me"
        assert fetched[0].metadata.chunk_index == 2
        assert fetched[0].metadata.source_type == SourceType.TEXT
