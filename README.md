# RAG

A modular, production-grade Retrieval-Augmented Generation pipeline. Fully
local: embeddings and generation run through [Ollama](https://ollama.com),
vectors are persisted in an embedded [Chroma](https://www.trychroma.com/)
store, and retrieval fuses dense similarity with BM25 sparse keyword search.
No external API keys required.

## Architecture

```
Documents (PDF / Markdown / Text)
        │
        ▼
  DocumentLoader        rag/ingestion    — parses files into Document objects, tracks checksum/metadata
        │
        ▼
     Chunker            rag/chunking     — RecursiveCharacterChunker / MarkdownChunker, bounded + overlapping
        │
        ▼
 EmbeddingProvider       rag/embeddings  — OllamaEmbeddingProvider (batched, retried via tenacity)
        │
        ▼
    VectorStore          rag/vectorstore — ChromaVectorStore (persistent, metadata-filterable)
        │
        ▼
    Retriever            rag/retrieval   — HybridRetriever: dense (Chroma) + sparse (BM25) via Reciprocal Rank Fusion
        │
        ▼
    Generator             rag/generation — OllamaGenerator: grounded prompt, retried, latency/token tracked
        │
        ▼
  GenerationResult (answer + cited sources)
```

Every stage is an abstract interface (`rag/core/interfaces.py`), so any
component can be swapped (e.g. Chroma → Qdrant, Ollama → OpenAI) without
touching `RAGPipeline`.

## Setup

Requires Python 3.10+ and a running [Ollama](https://ollama.com) instance.

```bash
# 1. Install Ollama and pull the models this pipeline uses
ollama pull nomic-embed-text
ollama pull qwen2.5:3b

# 2. Install the Python package
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure (optional — sensible defaults are built in)
cp .env.example .env
```

## Usage

```bash
# Ingest a file or a directory of PDFs/Markdown/text
python scripts/ingest.py data/sample_docs

# Ask a question
python scripts/query.py "What is the refund policy?"
```

Or from Python:

```python
from rag.pipeline import RAGPipeline

with RAGPipeline() as pipeline:
    pipeline.ingest_path("data/sample_docs")
    result = pipeline.ask("What is the refund policy?")
    print(result.answer)
    for source in result.sources:
        print(source.chunk.metadata.source_path, source.score)
```

## Configuration

All settings are environment-driven (`RAG_` prefix, see `.env.example`),
loaded via `rag.config.Settings` (pydantic-settings). Key knobs:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `RAG_OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_OLLAMA_GENERATION_MODEL` | `qwen2.5:3b` | Chat/generation model |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `120` | Chunking window (chars) |
| `RAG_CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store location |
| `RAG_RETRIEVAL_TOP_K` | `5` | Chunks returned per query |
| `RAG_RRF_K` | `60` | Reciprocal Rank Fusion constant |

## Design notes

- **Hybrid retrieval**: dense (Chroma cosine similarity) and sparse (BM25
  over an in-memory index of chunk text) results are combined via
  Reciprocal Rank Fusion, so a query is well-served whether it needs
  semantic matching or exact keyword recall. The BM25 corpus is rebuilt
  from Chroma on `RAGPipeline` startup (`VectorStore.get_all()` →
  `HybridRetriever.load_corpus()`), so restarting the process doesn't lose
  sparse-search coverage over previously ingested documents.
- **Retry/error handling**: all Ollama HTTP calls (`OllamaEmbeddingProvider`,
  `OllamaGenerator`) use `tenacity` with exponential backoff on transient
  network failures, and raise typed exceptions (`rag.core.exceptions`) on
  permanent failures rather than propagating raw HTTP errors.
- **Metadata tracking**: every `Document` carries a SHA-256 checksum,
  source path/type, and ingestion timestamp; every `Chunk` inherits that
  lineage plus its character offsets within the source document, so
  retrieval results are always traceable back to their origin.
- **Typing**: all data structures (`Document`, `Chunk`, `SearchResult`,
  `GenerationResult`) are `pydantic` models with field validation
  (e.g. rejecting empty content); all components are typed against ABCs
  in `rag/core/interfaces.py`.

## Testing

```bash
pytest -q
```

The suite (`tests/`) runs fully offline: `tests/conftest.py` provides a
deterministic `FakeEmbeddingProvider` and `FakeGenerator` so chunking,
vector-store insertion/query, hybrid-retrieval fusion, and full
ingest-to-answer pipeline behavior are all verified without a live Ollama
server. Coverage includes:

- Chunking bounds (max size respected, overlap present, sequential
  indices, validation errors on bad config/empty input).
- Vector store insertion/query/delete/upsert-dedup and restart recovery
  (`get_all`).
- Hybrid retrieval fusion (keyword-exact matches rank correctly, top_k
  respected, fused scores sorted).
- End-to-end ingest → retrieve → generate against sample documents.

## Project layout

```
src/rag/
  core/          # pydantic models + ABCs shared by every stage
  ingestion/     # PDF / Markdown / Text loaders
  chunking/      # RecursiveCharacterChunker, MarkdownChunker
  embeddings/    # OllamaEmbeddingProvider
  vectorstore/   # ChromaVectorStore
  retrieval/     # HybridRetriever (dense + BM25 + RRF)
  generation/    # OllamaGenerator
  pipeline.py    # RAGPipeline orchestrator
  config.py      # Settings (env-driven)
scripts/
  ingest.py      # CLI: index a file or directory
  query.py       # CLI: ask a question
tests/           # pytest suite (offline, fakes for Ollama)
data/sample_docs/ # example Markdown/text docs for a quick demo
```
