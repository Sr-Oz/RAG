"""End-to-end verification of the pipeline: ingest -> retrieve -> generate,
with the Ollama embedder/generator swapped for offline fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings
from rag.pipeline import RAGPipeline
from rag.vectorstore.chroma_store import ChromaVectorStore


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "refund.md").write_text(
        "# Refund Policy\n\n"
        "Customers may request a refund within 30 days of purchase if the "
        "product is unused and in original packaging.\n"
    )
    (docs_dir / "shipping.txt").write_text(
        "Standard shipping takes five to seven business days. "
        "Express shipping is available for an additional fee.\n"
    )
    return docs_dir


@pytest.fixture
def pipeline(chroma_persist_dir: str, fake_embedder, fake_generator, sample_docs_dir: Path) -> RAGPipeline:
    settings = Settings(
        chroma_persist_dir=chroma_persist_dir,
        chroma_collection_name="e2e_test",
        chunk_size=200,
        chunk_overlap=30,
        retrieval_top_k=3,
    )
    store = ChromaVectorStore(
        collection_name=settings.chroma_collection_name, persist_dir=settings.chroma_persist_dir
    )
    return RAGPipeline(
        settings=settings,
        embedder=fake_embedder,
        vector_store=store,
        generator=fake_generator,
    )


class TestEndToEnd:
    def test_ingest_indexes_expected_chunk_count(
        self, pipeline: RAGPipeline, sample_docs_dir: Path
    ) -> None:
        count = pipeline.ingest_path(str(sample_docs_dir))
        assert count > 0
        assert pipeline.vector_store.count() == count

    def test_ask_returns_grounded_answer_with_sources(
        self, pipeline: RAGPipeline, sample_docs_dir: Path
    ) -> None:
        pipeline.ingest_path(str(sample_docs_dir))
        result = pipeline.ask("What is the refund policy?")

        assert result.answer
        assert result.model == "fake-model"
        assert len(result.sources) > 0
        assert any("refund" in s.chunk.content.lower() for s in result.sources)

    def test_ask_on_empty_index_still_returns_result(self, pipeline: RAGPipeline) -> None:
        result = pipeline.ask("anything at all")
        assert result.answer
        assert result.sources == []

    def test_ingesting_single_file_works(
        self, pipeline: RAGPipeline, sample_docs_dir: Path
    ) -> None:
        count = pipeline.ingest_path(str(sample_docs_dir / "refund.md"))
        assert count > 0

        result = pipeline.ask("How many days to request a refund?")
        assert any("30 days" in s.chunk.content or "refund" in s.chunk.content.lower() for s in result.sources)
