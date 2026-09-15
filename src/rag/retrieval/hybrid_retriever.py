"""Hybrid retriever combining dense (vector) and sparse (BM25) search via
Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import re

from rag.core.exceptions import RetrievalError
from rag.core.interfaces import EmbeddingProvider, Retriever, VectorStore
from rag.core.models import Chunk, SearchResult

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever(Retriever):
    """Retrieves chunks using both dense embedding similarity (via a
    VectorStore) and sparse BM25 keyword search, fusing the two ranked
    lists with Reciprocal Rank Fusion so neither signal dominates on scale
    alone.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        top_k: int = 5,
        rrf_k: int = 60,
        dense_candidates: int = 20,
        sparse_candidates: int = 20,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RetrievalError(
                "rank_bm25 is required for hybrid retrieval. Install with: pip install rank_bm25"
            ) from exc

        self._vector_store = vector_store
        self._embedder = embedder
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.dense_candidates = dense_candidates
        self.sparse_candidates = sparse_candidates
        self._bm25_cls = BM25Okapi

        self._corpus_chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks into the vector store, and add them to
        the in-memory BM25 sparse index.
        """
        if not chunks:
            return

        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed_documents(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        self._vector_store.upsert(chunks)

        self._corpus_chunks.extend(chunks)
        self._corpus_tokens.extend(_tokenize(c.content) for c in chunks)
        self._bm25 = self._bm25_cls(self._corpus_tokens)

        logger.info("Indexed %d chunks (dense + sparse); corpus size=%d", len(chunks), len(self._corpus_chunks))

    def load_corpus(self, chunks: list[Chunk]) -> None:
        """Rebuild the in-memory BM25 sparse index from chunks already
        persisted in the vector store (e.g. after a process restart), without
        re-embedding or re-upserting them.
        """
        if not chunks:
            return
        self._corpus_chunks = list(chunks)
        self._corpus_tokens = [_tokenize(c.content) for c in chunks]
        self._bm25 = self._bm25_cls(self._corpus_tokens)
        logger.info("Loaded %d existing chunks into sparse index", len(chunks))

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query or not query.strip():
            raise RetrievalError("Query must not be empty")

        k = top_k or self.top_k
        dense_results = self._dense_search(query)
        sparse_results = self._sparse_search(query)
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:k]

    def _dense_search(self, query: str) -> list[SearchResult]:
        query_embedding = self._embedder.embed_query(query)
        return self._vector_store.query(query_embedding, top_k=self.dense_candidates)

    def _sparse_search(self, query: str) -> list[SearchResult]:
        if self._bm25 is None or not self._corpus_chunks:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self.sparse_candidates]

        results: list[SearchResult] = []
        for rank, idx in enumerate(ranked):
            if scores[idx] <= 0:
                continue
            results.append(
                SearchResult(
                    chunk=self._corpus_chunks[idx],
                    score=float(scores[idx]),
                    sparse_score=float(scores[idx]),
                    rank=rank,
                )
            )
        return results

    def _reciprocal_rank_fusion(
        self, dense: list[SearchResult], sparse: list[SearchResult]
    ) -> list[SearchResult]:
        """Fuse two ranked lists using RRF: score(d) = sum(1 / (k + rank))."""
        fused_scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        dense_score_by_id: dict[str, float] = {}
        sparse_score_by_id: dict[str, float] = {}

        for rank, result in enumerate(dense):
            cid = result.chunk.id
            chunk_by_id[cid] = result.chunk
            dense_score_by_id[cid] = result.score
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, result in enumerate(sparse):
            cid = result.chunk.id
            chunk_by_id[cid] = result.chunk
            sparse_score_by_id[cid] = result.score
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

        fused: list[SearchResult] = []
        for rank, cid in enumerate(ordered_ids):
            fused.append(
                SearchResult(
                    chunk=chunk_by_id[cid],
                    score=fused_scores[cid],
                    dense_score=dense_score_by_id.get(cid),
                    sparse_score=sparse_score_by_id.get(cid),
                    rank=rank,
                )
            )
        return fused
