#!/usr/bin/env python3
"""Ingest a file or directory of documents into the RAG vector store.

Usage:
    python scripts/ingest.py data/sample_docs
    python scripts/ingest.py path/to/report.pdf
"""

from __future__ import annotations

import argparse
import sys

from rag.logging_config import configure_logging
from rag.pipeline import RAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--no-recursive", action="store_true", help="Do not recurse into subdirectories"
    )
    args = parser.parse_args()

    configure_logging()

    with RAGPipeline() as pipeline:
        count = pipeline.ingest_path(args.path, recursive=not args.no_recursive)
        print(f"Indexed {count} chunks from {args.path}")
        print(f"Vector store now contains {pipeline.vector_store.count()} chunks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
