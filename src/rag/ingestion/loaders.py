"""Document loaders for PDF, Markdown, and plain-text sources."""

from __future__ import annotations

import logging
from pathlib import Path

from rag.core.exceptions import IngestionError
from rag.core.interfaces import DocumentLoader
from rag.core.models import Document, DocumentMetadata, SourceType

logger = logging.getLogger(__name__)


class TextLoader(DocumentLoader):
    """Loads plain-text (.txt) files."""

    extensions = (".txt",)

    def supports(self, source_path: str) -> bool:
        return Path(source_path).suffix.lower() in self.extensions

    def load(self, source_path: str) -> list[Document]:
        path = Path(source_path)
        if not path.is_file():
            raise IngestionError(f"File not found: {source_path}")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise IngestionError(f"Failed to read {source_path}: {exc}") from exc

        if not content.strip():
            raise IngestionError(f"File is empty: {source_path}")

        metadata = DocumentMetadata(
            source_path=str(path),
            source_type=SourceType.TEXT,
            title=path.stem,
            checksum=Document.compute_checksum(content),
        )
        logger.info("Loaded text document: %s (%d chars)", path, len(content))
        return [Document(content=content, metadata=metadata)]


class MarkdownLoader(DocumentLoader):
    """Loads Markdown (.md, .markdown) files, preserving raw markdown text."""

    extensions = (".md", ".markdown")

    def supports(self, source_path: str) -> bool:
        return Path(source_path).suffix.lower() in self.extensions

    def load(self, source_path: str) -> list[Document]:
        path = Path(source_path)
        if not path.is_file():
            raise IngestionError(f"File not found: {source_path}")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise IngestionError(f"Failed to read {source_path}: {exc}") from exc

        if not content.strip():
            raise IngestionError(f"File is empty: {source_path}")

        title = self._extract_title(content) or path.stem
        metadata = DocumentMetadata(
            source_path=str(path),
            source_type=SourceType.MARKDOWN,
            title=title,
            checksum=Document.compute_checksum(content),
        )
        logger.info("Loaded markdown document: %s (%d chars)", path, len(content))
        return [Document(content=content, metadata=metadata)]

    @staticmethod
    def _extract_title(content: str) -> str | None:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return None


class PDFLoader(DocumentLoader):
    """Loads PDF files, extracting per-page text via pypdf."""

    extensions = (".pdf",)

    def supports(self, source_path: str) -> bool:
        return Path(source_path).suffix.lower() in self.extensions

    def load(self, source_path: str) -> list[Document]:
        path = Path(source_path)
        if not path.is_file():
            raise IngestionError(f"File not found: {source_path}")

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise IngestionError(
                "pypdf is required to load PDF files. Install with: pip install pypdf"
            ) from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise IngestionError(f"Failed to open PDF {source_path}: {exc}") from exc

        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception as exc:  # malformed page content stream
                logger.warning("Failed to extract text from a page in %s: %s", path, exc)
                pages_text.append("")

        content = "\n\n".join(t for t in pages_text if t.strip())
        if not content.strip():
            raise IngestionError(f"No extractable text in PDF: {source_path}")

        doc_meta = getattr(reader, "metadata", None)
        title = (doc_meta.title if doc_meta and doc_meta.title else None) or path.stem
        author = doc_meta.author if doc_meta and doc_meta.author else None

        metadata = DocumentMetadata(
            source_path=str(path),
            source_type=SourceType.PDF,
            title=title,
            author=author,
            page_count=len(reader.pages),
            checksum=Document.compute_checksum(content),
        )
        logger.info(
            "Loaded PDF document: %s (%d pages, %d chars)",
            path,
            len(reader.pages),
            len(content),
        )
        return [Document(content=content, metadata=metadata)]


class LoaderRegistry:
    """Dispatches source files to the first loader that supports them."""

    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        self._loaders = loaders or [TextLoader(), MarkdownLoader(), PDFLoader()]

    def load(self, source_path: str) -> list[Document]:
        for loader in self._loaders:
            if loader.supports(source_path):
                return loader.load(source_path)
        raise IngestionError(f"No loader registered for file: {source_path}")

    def load_directory(self, directory: str, recursive: bool = True) -> list[Document]:
        base = Path(directory)
        if not base.is_dir():
            raise IngestionError(f"Directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        documents: list[Document] = []
        for file_path in sorted(base.glob(pattern)):
            if not file_path.is_file():
                continue
            if not any(loader.supports(str(file_path)) for loader in self._loaders):
                continue
            try:
                documents.extend(self.load(str(file_path)))
            except IngestionError as exc:
                logger.error("Skipping %s: %s", file_path, exc)
        return documents
