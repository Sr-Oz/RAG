"""Shared pytest fixtures: deterministic fakes for Ollama-backed components
so the test suite runs fully offline, with no live Ollama server required.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest

from rag.core.interfaces import EmbeddingProvider, Generator
from rag.core.models import GenerationResult, SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")
FAKE_DIM = 32


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline stand-in for an Ollama embedder. Produces a
    bag-of-words hashed vector so texts sharing vocabulary end up with
    higher cosine similarity, without requiring a real model.
    """

    @property
    def dimension(self) -> int:
        return FAKE_DIM

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * FAKE_DIM
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            idx = hash(token) % FAKE_DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FakeGenerator(Generator):
    """Offline stand-in for an Ollama chat model: echoes the query and lists
    which sources it was given, without calling any network service.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[SearchResult]]] = []

    def generate(self, query: str, context: list[SearchResult]) -> GenerationResult:
        self.calls.append((query, context))
        if not context:
            answer = "I don't have enough information to answer that."
        else:
            snippet = context[0].chunk.content[:50]
            answer = f"Based on the retrieved context ('{snippet}...'), here is the answer to: {query}"
        return GenerationResult(answer=answer, sources=context, model="fake-model")


@pytest.fixture
def fake_embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def chroma_persist_dir(tmp_path: Any) -> str:
    return str(tmp_path / "chroma")
