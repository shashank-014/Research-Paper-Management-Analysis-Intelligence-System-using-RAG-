from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from research_ai.indexing.embedding_model import BaseEmbedder
from research_ai.indexing.semantic_search import semantic_search
from research_ai.rag.prompt_templates import build_qa_messages, format_context, format_sources
from research_ai.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


class BaseLLMClient:
    model_name: str

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError


class GroqLLMClient(BaseLLMClient):
    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str | None = None) -> None:
        self.model_name = model_name
        self._api_key = _resolve_api_key(api_key)
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise ImportError("groq package is required for RAG generation") from exc

        self._client = Groq(api_key=self._api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


def retrieve_context(
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int = 6,
    index_dir: str | Path = "data/indices",
    embedder: BaseEmbedder | None = None,
    provider: str = "sentence_transformers",
    model_name: str | None = None,
    paper_ids: list[str] | None = None,
    paper_titles: list[str] | None = None,
) -> dict[str, Any]:
    configure_logging()

    fetched = semantic_search(
        query,
        filters=filters,
        top_k=max(top_k * 3, top_k),
        index_dir=index_dir,
        embedder=embedder,
        provider=provider,
        model_name=model_name,
    )
    filtered = _apply_paper_scope(fetched, paper_ids=paper_ids, paper_titles=paper_titles)
    deduped = _dedupe_results(filtered)
    selected = deduped[:top_k]

    return {
        "query": query,
        "results": selected,
        "context": format_context(selected),
        "sources": format_sources(selected),
    }


def answer_question(
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int = 6,
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
        return {
            "query": query,
            "answer": "Information not found in retrieved papers.",
            "sources": [],
            "retrieved_chunks": 0,
        }

    active_llm = llm_client or GroqLLMClient(model_name=llm_model, api_key=api_key)
    prompt = build_qa_messages(query, retrieval["context"])
    answer = active_llm.generate(prompt, temperature=0.1).strip()
    if not answer:
        answer = "Information not found in retrieved papers."

    return {
        "query": query,
        "answer": answer,
        "sources": retrieval["sources"],
        "retrieved_chunks": len(retrieval["results"]),
    }


def parse_json_response(payload: str) -> dict[str, Any]:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end > start:
            return json.loads(payload[start : end + 1])
        raise ValueError("LLM did not return valid JSON")


def _apply_paper_scope(
    results: list[dict[str, Any]],
    *,
    paper_ids: list[str] | None,
    paper_titles: list[str] | None,
) -> list[dict[str, Any]]:
    scoped = results
    if paper_ids:
        wanted_ids = {item.lower() for item in paper_ids}
        scoped = [item for item in scoped if str(item.get("paper_id", "")).lower() in wanted_ids]
    if paper_titles:
        wanted_titles = {item.lower() for item in paper_titles}
        scoped = [item for item in scoped if str(item.get("paper_title", "")).lower() in wanted_titles]
    return scoped


def _dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        chunk_id = str(item.get("chunk_id", ""))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(item)
    return deduped


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    try:
        import streamlit as st

        return st.secrets["GROQ_API_KEY"]
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            "GROQ_API_KEY not available. Pass api_key explicitly or configure st.secrets['GROQ_API_KEY']."
        ) from exc
