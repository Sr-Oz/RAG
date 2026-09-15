"""Core data structures shared across the RAG pipeline."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class DocumentMetadata(BaseModel):
    """Provenance and tracking metadata for a source document."""

    source_path: str
    source_type: SourceType
    title: str | None = None
    author: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str | None = None
    page_count: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A single ingested source document, prior to chunking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: DocumentMetadata

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Document content must not be empty")
        return v

    @staticmethod
    def compute_checksum(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ChunkMetadata(BaseModel):
    """Metadata carried by every chunk, inherited and extended from its document."""

    source_path: str
    source_type: SourceType
    document_id: str
    chunk_index: int
    title: str | None = None
    page_number: int | None = None
    start_char: int = 0
    end_char: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit of text produced by a chunker."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk content must not be empty")
        return v


class SearchResult(BaseModel):
    """A single scored retrieval hit."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rank: int | None = None


class GenerationResult(BaseModel):
    """Final answer produced by the generation stage."""

    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
