from __future__ import annotations

from pathlib import Path
from typing import Any

from research_ai.indexing.embedding_model import BaseEmbedder, create_embedder
from research_ai.indexing.vector_store import FaissVectorStore


def semantic_search(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    index_dir: str | Path = "data/indices",
    embedder: BaseEmbedder | None = None,
    provider: str = "sentence_transformers",
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    active_embedder = embedder or create_embedder(
        provider=provider,
        model_name=model_name,
        cache_path=Path(index_dir) / "embedding_cache.json",
    )
    vector_store = FaissVectorStore(index_dir).load()
    query_vector = active_embedder.embed_query(query)
    return vector_store.search(query_vector, top_k=top_k, filters=filters)
