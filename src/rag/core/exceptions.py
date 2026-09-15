"""Exception hierarchy for the RAG pipeline."""


class RAGError(Exception):
    """Base class for all RAG pipeline errors."""


class IngestionError(RAGError):
    """Raised when a document cannot be loaded or parsed."""


class ChunkingError(RAGError):
    """Raised when a document cannot be split into chunks."""


class EmbeddingError(RAGError):
    """Raised when an embedding provider fails to produce vectors."""


class VectorStoreError(RAGError):
    """Raised on vector store connection, insertion, or query failures."""


class RetrievalError(RAGError):
    """Raised when retrieval fails or returns an invalid state."""


class GenerationError(RAGError):
    """Raised when the LLM generation step fails."""


class ConfigurationError(RAGError):
    """Raised when required configuration is missing or invalid."""
