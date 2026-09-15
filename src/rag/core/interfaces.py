"""Abstract base classes defining the pipeline's component contracts.

Every concrete implementation (loaders, chunkers, embedders, vector stores,
retrievers, generators) plugs into one of these interfaces so that stages
can be swapped independently (e.g. Chroma -> Qdrant, Ollama -> OpenAI)
without touching the orchestration layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from rag.core.models import Chunk, Document, GenerationResult, SearchResult


class DocumentLoader(ABC):
    """Loads raw source files into `Document` objects."""

    @abstractmethod
    def supports(self, source_path: str) -> bool:
        """Return True if this loader can handle the given file path."""

    @abstractmethod
    def load(self, source_path: str) -> list[Document]:
        """Parse a source file into one or more Documents."""


class Chunker(ABC):
    """Splits Documents into retrievable Chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a single document into chunks."""

    def chunk_all(self, documents: Iterable[Document]) -> list[Chunk]:
        """Convenience: chunk a collection of documents."""
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks


class EmbeddingProvider(ABC):
    """Converts text into dense vector embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Size of the embedding vectors this provider produces."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk/document texts."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class VectorStore(ABC):
    """Persists and searches chunk embeddings."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks (each must already carry an embedding)."""

    @abstractmethod
    def query(
        self, embedding: list[float], top_k: int = 5, filters: dict | None = None
    ) -> list[SearchResult]:
        """Dense similarity search returning the top_k nearest chunks."""

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by id."""

    @abstractmethod
    def count(self) -> int:
        """Total number of chunks currently stored."""

    @abstractmethod
    def get_all(self) -> list[Chunk]:
        """Fetch every stored chunk (content + metadata), used to rebuild
        auxiliary in-memory indexes (e.g. BM25) after a restart."""


class Retriever(ABC):
    """Retrieves ranked chunks relevant to a query."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the top_k most relevant chunks for the query."""


class Generator(ABC):
    """Produces a grounded answer from a query and retrieved context."""

    @abstractmethod
    def generate(self, query: str, context: list[SearchResult]) -> GenerationResult:
        """Generate an answer using the retrieved chunks as context."""
