"""Bounds and correctness tests for the chunking strategies."""

from __future__ import annotations

import pytest

from rag.chunking.splitters import MarkdownChunker, RecursiveCharacterChunker
from rag.core.exceptions import ChunkingError
from rag.core.models import Document, DocumentMetadata, SourceType


def make_document(content: str, source_type: SourceType = SourceType.TEXT) -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(source_path="test.txt", source_type=source_type),
    )


class TestRecursiveCharacterChunkerValidation:
    def test_rejects_zero_chunk_size(self) -> None:
        with pytest.raises(ChunkingError):
            RecursiveCharacterChunker(chunk_size=0, chunk_overlap=0)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ChunkingError):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=-1)

    def test_rejects_overlap_gte_chunk_size(self) -> None:
        with pytest.raises(ChunkingError):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)

    def test_rejects_empty_document(self) -> None:
        # Document itself already rejects whitespace-only content at
        # construction time; use model_construct to bypass that validator
        # and exercise the chunker's own defensive guard directly.
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        blank_doc = Document.model_construct(
            content="   ",
            metadata=DocumentMetadata(source_path="test.txt", source_type=SourceType.TEXT),
        )
        with pytest.raises(ChunkingError):
            chunker.chunk(blank_doc)


class TestRecursiveCharacterChunkerBounds:
    def test_single_short_document_yields_one_chunk(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
        doc = make_document("This is a short document that fits in one chunk.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content.strip() == doc.content.strip()

    def test_chunks_respect_max_size(self) -> None:
        chunk_size = 100
        chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=20)
        # Long document with no convenient separators near boundaries.
        paragraph = " ".join(f"word{i}" for i in range(500))
        doc = make_document(paragraph)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= chunk_size + 1  # allow 1-char rounding from join

    def test_chunk_indices_are_sequential(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
        doc = make_document(" ".join(f"token{i}" for i in range(200)))
        chunks = chunker.chunk(doc)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_consecutive_chunks_overlap(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=20)
        doc = make_document(" ".join(f"tok{i}" for i in range(200)))
        chunks = chunker.chunk(doc)
        assert len(chunks) > 2

        # Overlap manifests as shared trailing/leading tokens between
        # consecutive chunks.
        for i in range(len(chunks) - 1):
            tail_tokens = set(chunks[i].content.split()[-3:])
            head_tokens = set(chunks[i + 1].content.split()[:5])
            assert tail_tokens & head_tokens, (
                f"Expected overlap between chunk {i} and {i + 1}"
            )

    def test_all_chunks_nonempty(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=5)
        doc = make_document("a b c d e f g h i j k l m n o p q r s t u v w x y z " * 10)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.content.strip()

    def test_metadata_carries_document_lineage(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=1000, chunk_overlap=50)
        doc = make_document("Short content.")
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata.document_id == doc.id
        assert chunks[0].metadata.source_path == doc.metadata.source_path
        assert chunks[0].metadata.source_type == SourceType.TEXT


class TestMarkdownChunker:
    def test_splits_on_headers_and_attaches_heading_metadata(self) -> None:
        chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
        content = (
            "# Title\n\nIntro text.\n\n"
            "## Section A\n\nContent A.\n\n"
            "## Section B\n\nContent B.\n"
        )
        doc = make_document(content, source_type=SourceType.MARKDOWN)
        chunks = chunker.chunk(doc)

        headings = {c.metadata.extra.get("heading") for c in chunks}
        assert "Section A" in headings
        assert "Section B" in headings

    def test_large_section_still_gets_subdivided(self) -> None:
        chunker = MarkdownChunker(chunk_size=80, chunk_overlap=10)
        big_section = "## Big\n\n" + " ".join(f"w{i}" for i in range(300))
        doc = make_document(big_section, source_type=SourceType.MARKDOWN)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 90

    def test_rejects_empty_document(self) -> None:
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=10)
        blank_doc = Document.model_construct(
            content="  \n  ",
            metadata=DocumentMetadata(source_path="test.md", source_type=SourceType.MARKDOWN),
        )
        with pytest.raises(ChunkingError):
            chunker.chunk(blank_doc)
