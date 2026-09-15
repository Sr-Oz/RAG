"""Embedding provider backed by a local Ollama server."""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.core.exceptions import EmbeddingError
from rag.core.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeds text via Ollama's `/api/embed` endpoint, batching requests and
    retrying transient network failures with exponential backoff.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        max_retries: int = 3,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.batch_size = batch_size
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t for t in texts if t and t.strip()]
        if len(cleaned) != len(texts):
            raise EmbeddingError("Cannot embed empty text(s) in batch")

        embeddings: list[list[float]] = []
        for i in range(0, len(cleaned), self.batch_size):
            batch = cleaned[i : i + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty query text")
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._call_with_retry(texts)
        except _RETRYABLE_EXCEPTIONS as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed after {self.max_retries} attempts: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama embedding request returned {exc.response.status_code}: {exc.response.text}"
            ) from exc

        embeddings = response.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings or [])} embeddings for {len(texts)} inputs"
            )
        return embeddings

    def _call_with_retry(self, texts: list[str]) -> dict:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        )
        def _call() -> dict:
            resp = self._client.post(
                "/api/embed", json={"model": self.model, "input": texts}
            )
            resp.raise_for_status()
            return resp.json()

        return _call()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaEmbeddingProvider":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
