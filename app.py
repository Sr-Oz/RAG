#!/usr/bin/env python3
"""Gradio front end for the RAG pipeline — no command line required.

Run with:
    python app.py

Then open the printed local URL in a browser. Upload documents on the
"Manage Documents" tab, then ask questions on the "Ask Questions" tab.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from rag.core.exceptions import RAGError
from rag.logging_config import configure_logging
from rag.pipeline import RAGPipeline

configure_logging()

_pipeline: RAGPipeline | None = None
_pipeline_error: str | None = None


def get_pipeline() -> RAGPipeline:
    """Lazily create the pipeline once and reuse it across requests."""
    global _pipeline, _pipeline_error
    if _pipeline is None and _pipeline_error is None:
        try:
            _pipeline = RAGPipeline()
        except Exception as exc:  # startup/config problems, not per-request errors
            _pipeline_error = str(exc)
    if _pipeline_error is not None:
        raise RAGError(
            f"The knowledge base could not be started: {_pipeline_error}\n"
            "Check that Ollama is installed and try restarting the app."
        )
    assert _pipeline is not None
    return _pipeline


def list_indexed_documents() -> str:
    try:
        pipeline = get_pipeline()
        chunks = pipeline.vector_store.get_all()
    except RAGError as exc:
        return f"⚠️ {exc}"

    if not chunks:
        return "_No documents indexed yet. Add some on this tab to get started._"

    counts: dict[str, int] = {}
    for chunk in chunks:
        name = Path(chunk.metadata.source_path).name
        counts[name] = counts.get(name, 0) + 1

    lines = [f"- **{name}** — {n} chunk(s)" for name, n in sorted(counts.items())]
    return (
        f"**{len(chunks)} chunk(s)** across **{len(counts)} document(s)**:\n\n"
        + "\n".join(lines)
    )


def ingest_files(
    file_paths: list[str] | None, progress: gr.Progress = gr.Progress()
) -> tuple[str, str]:
    if not file_paths:
        return "Please choose at least one file first.", list_indexed_documents()

    try:
        pipeline = get_pipeline()
    except RAGError as exc:
        return f"⚠️ {exc}", list_indexed_documents()

    total_chunks = 0
    failures: list[str] = []
    for i, path in enumerate(file_paths):
        name = Path(path).name
        progress((i, len(file_paths)), desc=f"Reading {name}...")
        try:
            total_chunks += pipeline.ingest_path(path)
        except RAGError as exc:
            failures.append(f"{name}: {exc}")

    ok_count = len(file_paths) - len(failures)
    message = f"✅ Added {total_chunks} chunk(s) from {ok_count} file(s)."
    if failures:
        message += "\n\n⚠️ These files could not be added:\n" + "\n".join(
            f"- {f}" for f in failures
        )
    return message, list_indexed_documents()


def ask_question(
    message: str, history: list[dict[str, str]]
) -> tuple[list[dict[str, str]], str]:
    if not message or not message.strip():
        return history, ""

    history = history + [{"role": "user", "content": message}]

    try:
        pipeline = get_pipeline()
        result = pipeline.ask(message)
    except RAGError as exc:
        answer = f"Sorry, I couldn't answer that: {exc}"
        history = history + [{"role": "assistant", "content": answer}]
        return history, ""

    answer = result.answer
    if result.sources:
        seen: list[str] = []
        for source in result.sources:
            name = Path(source.chunk.metadata.source_path).name
            if name not in seen:
                seen.append(name)
        answer += "\n\n*Sources: " + ", ".join(seen) + "*"

    history = history + [{"role": "assistant", "content": answer}]
    return history, ""


with gr.Blocks(title="RAG Assistant") as demo:
    gr.Markdown("# 📚 RAG Assistant\nAsk questions about your documents — answered locally, no data leaves this machine.")

    with gr.Tab("💬 Ask Questions"):
        chatbot = gr.Chatbot(height=450, label="Conversation")
        with gr.Row():
            question_box = gr.Textbox(
                placeholder="Type your question and press Enter...",
                show_label=False,
                scale=8,
            )
            ask_button = gr.Button("Ask", variant="primary", scale=1)
        clear_button = gr.Button("Clear conversation")

        question_box.submit(ask_question, [question_box, chatbot], [chatbot, question_box])
        ask_button.click(ask_question, [question_box, chatbot], [chatbot, question_box])
        clear_button.click(lambda: [], None, chatbot)

    with gr.Tab("📁 Manage Documents"):
        gr.Markdown("Upload PDF, Markdown, or text files to add them to the knowledge base.")
        file_upload = gr.File(
            file_count="multiple",
            file_types=[".pdf", ".md", ".markdown", ".txt"],
            label="Choose files",
        )
        index_button = gr.Button("Add to knowledge base", variant="primary")
        status_box = gr.Markdown()
        gr.Markdown("### Currently indexed")
        indexed_box = gr.Markdown(list_indexed_documents)
        refresh_button = gr.Button("Refresh list")

        index_button.click(ingest_files, [file_upload], [status_box, indexed_box])
        refresh_button.click(list_indexed_documents, None, indexed_box)


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
