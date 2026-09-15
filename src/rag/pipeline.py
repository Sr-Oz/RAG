"""End-to-end RAG pipeline orchestrator wiring ingestion -> chunking ->
embedding/indexing -> hybrid retrieval -> generation.
"""

from __future__ import annotations

import logging

from rag.chunking.splitters import MarkdownChunker, RecursiveCharacterChunker
from rag.config import Settings, get_settings
from rag.core.interfaces import Chunker, EmbeddingProvider, Generator, VectorStore
from rag.core.models import Document, GenerationResult, SourceType
from rag.embeddings.ollama_embedder import OllamaEmbeddingProvider
from rag.generation.ollama_generator import OllamaGenerator
from rag.ingestion.loaders import LoaderRegistry
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Wires the full pipeline together and exposes `ingest()` / `ask()`."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        generator: Generator | None = None,
        chunkers: dict[SourceType, Chunker] | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.embedder = embedder or OllamaEmbeddingProvider(
            model=self.settings.ollama_embedding_model,
            base_url=self.settings.ollama_base_url,
            timeout=self.settings.ollama_request_timeout,
            max_retries=self.settings.ollama_max_retries,
        )
        self.vector_store = vector_store or ChromaVectorStore(
            collection_name=self.settings.chroma_collection_name,
            persist_dir=self.settings.chroma_persist_dir,
        )
        self.generator = generator or OllamaGenerator(
            model=self.settings.ollama_generation_model,
            base_url=self.settings.ollama_base_url,
            timeout=self.settings.ollama_request_timeout,
            max_retries=self.settings.ollama_max_retries,
            temperature=self.settings.generation_temperature,
            max_context_chunks=self.settings.generation_max_context_chunks,
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=self.settings.retrieval_top_k,
            rrf_k=self.settings.rrf_k,
        )

        default_recursive = RecursiveCharacterChunker(
            chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap
        )
        default_markdown = MarkdownChunker(
            chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap
        )
        self.chunkers = chunkers or {
            SourceType.MARKDOWN: default_markdown,
            SourceType.PDF: default_recursive,
            SourceType.TEXT: default_recursive,
        }

        self.loader_registry = LoaderRegistry()

        existing_chunks = self.vector_store.get_all()
        if existing_chunks:
            self.retriever.load_corpus(existing_chunks)
            logger.info("Restored sparse index with %d existing chunks", len(existing_chunks))

    def _chunker_for(self, document: Document) -> Chunker:
        return self.chunkers.get(document.metadata.source_type, self.chunkers[SourceType.TEXT])

    def ingest_path(self, path: str, recursive: bool = True) -> int:
        """Load, chunk, embed, and index a single file or a directory of
        files. Returns the number of chunks indexed.
        """
        import os

        if os.path.isdir(path):
            documents = self.loader_registry.load_directory(path, recursive=recursive)
        else:
            documents = self.loader_registry.load(path)

        return self.ingest_documents(documents)

    def ingest_documents(self, documents: list[Document]) -> int:
        total_chunks = 0
        for document in documents:
            chunker = self._chunker_for(document)
            chunks = chunker.chunk(document)
            self.retriever.index(chunks)
            total_chunks += len(chunks)
            logger.info(
                "Ingested document %s: %d chunks", document.metadata.source_path, len(chunks)
            )
        return total_chunks

    def ask(self, query: str, top_k: int | None = None) -> GenerationResult:
        k = top_k or self.settings.retrieval_top_k
        results = self.retriever.retrieve(query, top_k=k)
        return self.generator.generate(query, results)

    def close(self) -> None:
        if hasattr(self.embedder, "close"):
            self.embedder.close()
        if hasattr(self.generator, "close"):
            self.generator.close()

    def __enter__(self) -> "RAGPipeline":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
