"""Chunking strategies for splitting Documents into retrievable Chunks."""

from __future__ import annotations

import logging
import re

from rag.core.exceptions import ChunkingError
from rag.core.interfaces import Chunker
from rag.core.models import Chunk, ChunkMetadata, Document

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveCharacterChunker(Chunker):
    """Splits text on a hierarchy of separators, packing into `chunk_size`
    windows with `chunk_overlap` characters of overlap between consecutive
    chunks. Falls back to a hard character split when no separator yields
    a piece small enough to fit.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text or not text.strip():
            raise ChunkingError(f"Cannot chunk empty document: {document.id}")

        pieces = self._split_text(text, self.separators)
        windows = self._pack_windows(pieces)

        chunks: list[Chunk] = []
        for idx, (chunk_text, start, end) in enumerate(windows):
            metadata = ChunkMetadata(
                source_path=document.metadata.source_path,
                source_type=document.metadata.source_type,
                document_id=document.id,
                chunk_index=idx,
                title=document.metadata.title,
                start_char=start,
                end_char=end,
            )
            chunks.append(Chunk(content=chunk_text, metadata=metadata))

        logger.info(
            "Chunked document %s into %d chunks (size=%d, overlap=%d)",
            document.id,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text into pieces each <= chunk_size, preferring
        higher-priority (earlier) separators, falling back to hard slicing.
        """
        if len(text) <= self.chunk_size:
            return [text] if text else []

        if not separators:
            # Hard fallback: slice by character count.
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep, rest_seps = separators[0], separators[1:]
        if sep == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        parts = text.split(sep)
        if len(parts) == 1:
            # Separator not present; try the next one.
            return self._split_text(text, rest_seps)

        pieces: list[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) > self.chunk_size:
                pieces.extend(self._split_text(part, rest_seps))
            else:
                pieces.append(part)
        return pieces

    def _pack_windows(self, pieces: list[str]) -> list[tuple[str, int, int]]:
        """Greedily pack small pieces into windows up to chunk_size, carrying
        `chunk_overlap` characters of trailing context into the next window.
        Returns (text, start_char, end_char) tuples with approximate offsets.
        """
        if not pieces:
            return []

        windows: list[tuple[str, int, int]] = []
        current = ""
        cursor = 0
        window_start = 0

        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                windows.append((current, window_start, window_start + len(current)))
                overlap_text = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                window_start = window_start + len(current) - len(overlap_text)
                current = f"{overlap_text} {piece}".strip() if overlap_text else piece
            else:
                # Single piece already exceeds chunk_size (shouldn't normally
                # happen since _split_text bounds pieces) — emit as-is.
                windows.append((piece, window_start, window_start + len(piece)))
                window_start += len(piece)
                current = ""

            cursor += len(piece)

        if current:
            windows.append((current, window_start, window_start + len(current)))

        return windows


class MarkdownChunker(Chunker):
    """Splits Markdown by header boundaries first, then recursively packs
    each section into chunk_size windows. Keeps header context attached to
    each chunk's metadata for better retrieval grounding.
    """

    _HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self._recursive = RecursiveCharacterChunker(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text or not text.strip():
            raise ChunkingError(f"Cannot chunk empty document: {document.id}")

        sections = self._split_by_headers(text)
        chunks: list[Chunk] = []
        idx = 0
        for heading, body, start in sections:
            sub_doc = Document(
                id=document.id,
                content=body,
                metadata=document.metadata,
            )
            for sub_chunk in self._recursive.chunk(sub_doc):
                sub_chunk.metadata.chunk_index = idx
                sub_chunk.metadata.start_char += start
                sub_chunk.metadata.end_char += start
                if heading:
                    sub_chunk.metadata.extra["heading"] = heading
                chunks.append(sub_chunk)
                idx += 1

        logger.info(
            "Markdown-chunked document %s into %d chunks across %d sections",
            document.id,
            len(chunks),
            len(sections),
        )
        return chunks

    def _split_by_headers(self, text: str) -> list[tuple[str | None, str, int]]:
        matches = list(self._HEADER_RE.finditer(text))
        if not matches:
            return [(None, text, 0)]

        sections: list[tuple[str | None, str, int]] = []
        if matches[0].start() > 0:
            sections.append((None, text[: matches[0].start()], 0))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            body_start = match.start()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end]
            if body.strip():
                sections.append((heading, body, body_start))

        return sections
