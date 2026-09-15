#!/usr/bin/env python3
"""Ask a question against the indexed RAG corpus.

Usage:
    python scripts/query.py "What is the refund policy?"
    python scripts/query.py "What is X?" --top-k 8
"""

from __future__ import annotations

import argparse
import sys

from rag.core.exceptions import RAGError
from rag.logging_config import configure_logging
from rag.pipeline import RAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    args = parser.parse_args()

    configure_logging()

    with RAGPipeline() as pipeline:
        try:
            result = pipeline.ask(args.question, top_k=args.top_k)
        except RAGError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(f"\nAnswer ({result.model}, {result.latency_ms:.0f} ms):\n{result.answer}\n")
        print("Sources:")
        for i, source in enumerate(result.sources, start=1):
            print(
                f"  [{i}] {source.chunk.metadata.source_path} "
                f"(score={source.score:.4f}, chunk={source.chunk.metadata.chunk_index})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
