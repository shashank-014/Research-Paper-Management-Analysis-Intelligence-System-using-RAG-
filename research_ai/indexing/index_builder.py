from __future__ import annotations

import logging
from pathlib import Path

from research_ai.indexing.chunking import chunk_papers
from research_ai.indexing.embedding_model import BaseEmbedder, create_embedder
from research_ai.indexing.vector_store import FaissVectorStore
from research_ai.models import ResearchPaper
from research_ai.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def load_papers_from_json(folder_path: str | Path) -> list[ResearchPaper]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    papers: list[ResearchPaper] = []
    for json_path in sorted(folder.glob("*.json")):
        papers.append(ResearchPaper.model_validate_json(json_path.read_text(encoding="utf-8")))

    return papers


def index_papers(
    papers: list[ResearchPaper],
    index_dir: str | Path = "data/indices",
    embedder: BaseEmbedder | None = None,
    provider: str = "sentence_transformers",
    model_name: str | None = None,
    batch_size: int = 32,
    max_tokens: int = 700,
    overlap_tokens: int = 80,
) -> FaissVectorStore:
    configure_logging()

    if not papers:
        raise ValueError("No papers provided for indexing")

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    active_embedder = embedder or create_embedder(
        provider=provider,
        model_name=model_name,
        cache_path=index_path / "embedding_cache.json",
    )
    vector_store = FaissVectorStore(index_path)
    if vector_store.index_path.exists() and vector_store.metadata_path.exists():
        vector_store.load()
    chunks = chunk_papers(papers, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    logger.info("Generated %s chunks from %s paper(s)", len(chunks), len(papers))

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        embeddings = active_embedder.embed_texts([chunk.text for chunk in batch_chunks], batch_size=batch_size)
        vector_store.add_embeddings(batch_chunks, embeddings)

    vector_store.save()
    logger.info("Indexed %s chunk(s)", vector_store.size)
    return vector_store


def index_papers_from_json(
    folder_path: str | Path,
    index_dir: str | Path = "data/indices",
    provider: str = "sentence_transformers",
    model_name: str | None = None,
    batch_size: int = 32,
    max_tokens: int = 700,
    overlap_tokens: int = 80,
) -> FaissVectorStore:
    papers = load_papers_from_json(folder_path)
    return index_papers(
        papers,
        index_dir=index_dir,
        provider=provider,
        model_name=model_name,
        batch_size=batch_size,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

