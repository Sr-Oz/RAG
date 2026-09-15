"""Answer generation backed by a local Ollama chat model."""

from __future__ import annotations

import logging
import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.core.exceptions import GenerationError
from rag.core.interfaces import Generator
from rag.core.models import GenerationResult, SearchResult

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "provided context. If the context does not contain enough information "
    "to answer, say so explicitly instead of guessing. Cite sources by "
    "their [n] marker when you use them."
)


class OllamaGenerator(Generator):
    """Generates grounded answers via Ollama's `/api/chat` endpoint."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.2,
        max_context_chunks: int = 5,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.max_context_chunks = max_context_chunks
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def generate(self, query: str, context: list[SearchResult]) -> GenerationResult:
        if not query or not query.strip():
            raise GenerationError("Query must not be empty")

        selected = context[: self.max_context_chunks]
        prompt = self._build_prompt(query, selected)

        start = time.perf_counter()
        try:
            response = self._call_with_retry(prompt)
        except _RETRYABLE_EXCEPTIONS as exc:
            raise GenerationError(
                f"Ollama generation request failed after {self.max_retries} attempts: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                f"Ollama generation request returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        answer = response.get("message", {}).get("content", "").strip()
        if not answer:
            raise GenerationError("Ollama returned an empty generation")

        return GenerationResult(
            answer=answer,
            sources=selected,
            model=self.model,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
            latency_ms=latency_ms,
        )

    def _build_prompt(self, query: str, context: list[SearchResult]) -> str:
        if not context:
            context_block = "(no relevant context retrieved)"
        else:
            lines = []
            for i, result in enumerate(context, start=1):
                source = result.chunk.metadata.source_path
                lines.append(f"[{i}] (source: {source})\n{result.chunk.content}")
            context_block = "\n\n".join(lines)

        return f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"

    def _call_with_retry(self, prompt: str) -> dict:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        )
        def _call() -> dict:
            resp = self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {"temperature": self.temperature},
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json()

        return _call()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaGenerator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
