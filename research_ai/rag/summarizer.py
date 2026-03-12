from __future__ import annotations

from typing import Any

from research_ai.models import ResearchPaper
from research_ai.rag.prompt_templates import build_summary_messages, format_sources
from research_ai.rag.rag_pipeline import BaseLLMClient, GroqLLMClient, parse_json_response

SECTION_PRIORITY = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]


def summarize_paper(
    paper: ResearchPaper,
    *,
    llm_client: BaseLLMClient | None = None,
    llm_model: str = "llama-3.1-8b-instant",
    api_key: str | None = None,
    include_section_summaries: bool = True,
    max_context_chars: int = 14000,
) -> dict[str, Any]:
    sections = _ordered_sections(paper)
    context_parts: list[str] = []
    sources: list[dict[str, object]] = []
    remaining = max_context_chars

    for section in sections:
        if not section.content.strip():
            continue
        snippet = section.content.strip()
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rsplit(" ", 1)[0]
        if not snippet:
            break
        context_parts.append(f"Section: {section.section_name}\nText: {snippet}")
        sources.append({"paper_title": paper.title, "section": section.section_name, "score": 1.0})
        remaining -= len(snippet)
        if remaining <= 0:
            break

    if not context_parts:
        return {
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "short_summary": ["Insufficient information"],
            "structured_summary": {
                "problem_statement": "Insufficient information",
                "proposed_approach": "Insufficient information",
                "key_contributions": ["Insufficient information"],
                "experimental_results": "Insufficient information",
                "limitations": "Insufficient information",
            },
            "section_summaries": {},
            "sources": [],
        }

    active_llm = llm_client or GroqLLMClient(model_name=llm_model, api_key=api_key)
    prompt = build_summary_messages(paper_title=paper.title, context="\n\n".join(context_parts), include_section_summaries=include_section_summaries)
    response = active_llm.generate(prompt, temperature=0.1, json_mode=True)
    parsed = parse_json_response(response)

    return {
        "paper_id": paper.paper_id,
        "paper_title": paper.title,
        "short_summary": parsed.get("short_summary", ["Insufficient information"]),
        "structured_summary": parsed.get("structured_summary", {}),
        "section_summaries": parsed.get("section_summaries", {}),
        "sources": format_sources(sources),
    }


def _ordered_sections(paper: ResearchPaper) -> list[Any]:
    priority_map = {name: index for index, name in enumerate(SECTION_PRIORITY)}
    return sorted(paper.sections, key=lambda section: (priority_map.get(section.section_name.lower(), 99), section.page_start or 9999))
