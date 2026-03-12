from __future__ import annotations

from pathlib import Path
from typing import Any

from research_ai.indexing.embedding_model import BaseEmbedder
from research_ai.rag.prompt_templates import build_comparison_messages, format_context_by_paper, format_sources
from research_ai.rag.rag_pipeline import BaseLLMClient, GroqLLMClient, retrieve_context


def compare_papers(
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int = 8,
    index_dir: str | Path = "data/indices",
    llm_client: BaseLLMClient | None = None,
    llm_model: str = "llama-3.1-8b-instant",
    api_key: str | None = None,
    embedder: BaseEmbedder | None = None,
    provider: str = "sentence_transformers",
    model_name: str | None = None,
    paper_ids: list[str] | None = None,
    paper_titles: list[str] | None = None,
) -> dict[str, Any]:
    retrieval = retrieve_context(
        query,
        filters=filters,
        top_k=top_k,
        index_dir=index_dir,
        embedder=embedder,
        provider=provider,
        model_name=model_name,
        paper_ids=paper_ids,
        paper_titles=paper_titles,
    )

    if not retrieval["results"]:
        return {"query": query, "comparison": "Insufficient information", "sources": [], "paper_count": 0}

    active_llm = llm_client or GroqLLMClient(model_name=llm_model, api_key=api_key)
    grouped_context = format_context_by_paper(retrieval["results"])
    prompt = build_comparison_messages(query, grouped_context)
    comparison = active_llm.generate(prompt, temperature=0.1).strip() or "Insufficient information"

    paper_count = len({item.get("paper_title") for item in retrieval["results"]})
    return {
        "query": query,
        "comparison": comparison,
        "sources": format_sources(retrieval["results"]),
        "paper_count": paper_count,
    }
