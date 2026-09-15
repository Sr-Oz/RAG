"""Tests for hybrid (dense + sparse) retrieval and RRF fusion."""

from __future__ import annotations

import pytest

from rag.core.exceptions import RetrievalError
from rag.core.models import Chunk, ChunkMetadata, SourceType
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.vectorstore.chroma_store import ChromaVectorStore


def make_chunk(text: str, index: int) -> Chunk:
    return Chunk(
        content=text,
        metadata=ChunkMetadata(
            source_path=f"doc{index}.txt",
            source_type=SourceType.TEXT,
            document_id=f"doc-{index}",
            chunk_index=index,
        ),
    )


@pytest.fixture
def retriever(chroma_persist_dir: str, fake_embedder) -> HybridRetriever:
    store = ChromaVectorStore(collection_name="hybrid_test", persist_dir=chroma_persist_dir)
    return HybridRetriever(vector_store=store, embedder=fake_embedder, top_k=5, rrf_k=60)


class TestIndexingAndRetrieval:
    def test_retrieve_on_empty_index_returns_empty(self, retriever: HybridRetriever) -> None:
        assert retriever.retrieve("anything", top_k=5) == []

    def test_rejects_empty_query(self, retriever: HybridRetriever) -> None:
        with pytest.raises(RetrievalError):
            retriever.retrieve("   ")

    def test_exact_keyword_match_ranks_first(self, retriever: HybridRetriever) -> None:
        chunks = [
            make_chunk("The refund policy allows returns within 30 days.", 0),
            make_chunk("Shipping takes five to seven business days.", 1),
            make_chunk("Our warranty covers manufacturing defects only.", 2),
        ]
        retriever.index(chunks)

        results = retriever.retrieve("refund policy", top_k=3)
        assert len(results) > 0
        assert "refund" in results[0].chunk.content.lower()

    def test_fused_results_carry_both_score_components(self, retriever: HybridRetriever) -> None:
        chunks = [
            make_chunk("Python is a programming language used for data science.", 0),
            make_chunk("Bananas are a good source of potassium.", 1),
        ]
        retriever.index(chunks)

        results = retriever.retrieve("python programming", top_k=2)
        assert len(results) == 2
        top = results[0]
        assert top.dense_score is not None or top.sparse_score is not None
        assert top.score > 0

    def test_top_k_limits_result_count(self, retriever: HybridRetriever) -> None:
        chunks = [make_chunk(f"document number {i} about topic {i}", i) for i in range(10)]
        retriever.index(chunks)
        results = retriever.retrieve("topic", top_k=4)
        assert len(results) <= 4

    def test_results_sorted_descending_by_fused_score(self, retriever: HybridRetriever) -> None:
        chunks = [
            make_chunk("apple apple apple fruit snack", 0),
            make_chunk("apple mentioned once in this document", 1),
            make_chunk("completely unrelated content about cars", 2),
        ]
        retriever.index(chunks)
        results = retriever.retrieve("apple", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestLoadCorpus:
    def test_load_corpus_restores_sparse_index_without_reembedding(
        self, chroma_persist_dir: str, fake_embedder
    ) -> None:
        store = ChromaVectorStore(collection_name="restore_test", persist_dir=chroma_persist_dir)
        # rank_bm25's IDF is degenerate (near/exactly zero) when a term's
        # document frequency is a large fraction of a tiny corpus, so use
        # three diverse chunks to give "refunds" a clearly positive IDF.
        chunks = [
            make_chunk("persisted content about refunds", 0),
            make_chunk("completely unrelated topic about gardening", 1),
            make_chunk("a third document discussing automobiles and engines", 2),
        ]
        for chunk in chunks:
            chunk.embedding = fake_embedder.embed_query(chunk.content)
        store.upsert(chunks)

        new_retriever = HybridRetriever(vector_store=store, embedder=fake_embedder)
        before = new_retriever.retrieve("refunds", top_k=1)
        assert len(before) == 1  # dense search still finds it via the persisted vector store
        assert before[0].sparse_score is None  # but sparse index is empty until loaded

        new_retriever.load_corpus(store.get_all())
        after = new_retriever.retrieve("refunds", top_k=1)
        assert len(after) == 1
        assert after[0].sparse_score is not None  # now contributed by BM25 too
