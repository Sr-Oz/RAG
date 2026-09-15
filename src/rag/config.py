"""Centralized, environment-driven configuration for the RAG pipeline."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RAG_", extra="ignore"
    )

    # Ollama connection
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_embedding_model: str = Field(default="nomic-embed-text")
    ollama_generation_model: str = Field(default="qwen2.5:3b")
    ollama_request_timeout: float = Field(default=60.0)
    ollama_max_retries: int = Field(default=3)

    # Chunking
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)

    # Vector store (Chroma)
    chroma_persist_dir: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="rag_chunks")

    # Retrieval (dense + sparse fused via Reciprocal Rank Fusion)
    retrieval_top_k: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)

    # Generation
    generation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    generation_max_context_chunks: int = Field(default=5, gt=0)

    # Logging
    log_level: str = Field(default="INFO")


def get_settings() -> Settings:
    return Settings()
