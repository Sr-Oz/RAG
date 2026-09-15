"""Tests for document ingestion loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.core.exceptions import IngestionError
from rag.core.models import SourceType
from rag.ingestion.loaders import LoaderRegistry, MarkdownLoader, TextLoader


class TestTextLoader:
    def test_loads_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("Hello, RAG world.")
        docs = TextLoader().load(str(f))
        assert len(docs) == 1
        assert docs[0].content == "Hello, RAG world."
        assert docs[0].metadata.source_type == SourceType.TEXT
        assert docs[0].metadata.checksum is not None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError):
            TextLoader().load(str(tmp_path / "missing.txt"))

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("   ")
        with pytest.raises(IngestionError):
            TextLoader().load(str(f))


class TestMarkdownLoader:
    def test_extracts_h1_as_title(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# My Title\n\nSome content here.\n")
        docs = MarkdownLoader().load(str(f))
        assert docs[0].metadata.title == "My Title"

    def test_falls_back_to_filename_when_no_heading(self, tmp_path: Path) -> None:
        f = tmp_path / "no_heading.md"
        f.write_text("Just a paragraph, no header.")
        docs = MarkdownLoader().load(str(f))
        assert docs[0].metadata.title == "no_heading"


class TestLoaderRegistry:
    def test_dispatches_by_extension(self, tmp_path: Path) -> None:
        txt = tmp_path / "a.txt"
        txt.write_text("text content")
        md = tmp_path / "b.md"
        md.write_text("# heading\ncontent")

        registry = LoaderRegistry()
        assert registry.load(str(txt))[0].metadata.source_type == SourceType.TEXT
        assert registry.load(str(md))[0].metadata.source_type == SourceType.MARKDOWN

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        with pytest.raises(IngestionError):
            LoaderRegistry().load(str(f))

    def test_load_directory_skips_unsupported_and_bad_files(self, tmp_path: Path) -> None:
        (tmp_path / "good.txt").write_text("readable content")
        (tmp_path / "ignored.csv").write_text("a,b,c")
        (tmp_path / "empty.md").write_text("   ")

        docs = LoaderRegistry().load_directory(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata.source_path.endswith("good.txt")
