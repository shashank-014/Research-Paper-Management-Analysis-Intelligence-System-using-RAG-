from __future__ import annotations

import json
from collections import defaultdict


def format_sources(results: list[dict[str, object]]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in results:
        key = (str(item.get("paper_title", "")), str(item.get("section", "")))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "paper_title": item.get("paper_title"),
                "section": item.get("section"),
                "score": round(float(item.get("score", 0.0)), 4),
            }
        )
    return sources


def format_context(results: list[dict[str, object]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"Paper: {item.get('paper_title', 'Unknown Paper')}",
                    f"Section: {item.get('section', 'Unknown Section')}",
                    f"Score: {float(item.get('score', 0.0)):.4f}",
                    f"Text: {item.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_context_by_paper(results: list[dict[str, object]]) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in results:
        grouped[str(item.get("paper_title", "Unknown Paper"))].append(
            f"Section: {item.get('section', 'Unknown Section')}\nText: {item.get('text', '')}"
        )

    blocks: list[str] = []
    for paper_title, snippets in grouped.items():
        blocks.append(f"Paper: {paper_title}\n" + "\n\n".join(snippets))
    return "\n\n".join(blocks)


def build_summary_messages(paper_title: str, context: str, include_section_summaries: bool = True) -> list[dict[str, str]]:
    section_instruction = (
        'Include "section_summaries" for introduction, methods, and results when enough evidence exists.'
        if include_section_summaries
        else 'Set "section_summaries" to an empty object.'
    )
    schema = {
        "short_summary": ["bullet 1", "bullet 2"],
        "structured_summary": {
            "problem_statement": "",
            "proposed_approach": "",
            "key_contributions": [""],
            "experimental_results": "",
            "limitations": "",
        },
        "section_summaries": {
            "introduction": "",
            "methods": "",
            "results": "",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an academic research assistant. Use only the supplied paper context. "
                "Do not invent datasets, results, baselines, or claims. If evidence is missing, use 'Insufficient information'. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Summarize the paper titled '{paper_title}'.\n\n"
                f"Required output schema:\n{json.dumps(schema, indent=2)}\n\n"
                "Rules:\n"
                "- Keep a neutral academic tone.\n"
                "- The short summary must contain 5 or 6 bullets.\n"
                "- Key contributions must be a list.\n"
                f"- {section_instruction}\n"
                "- If the context lacks evidence for a field, set it to 'Insufficient information'.\n\n"
                f"Paper context:\n{context}"
            ),
        },
    ]


def build_qa_messages(query: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a grounded research QA assistant. Answer only from the retrieved context. "
                "Do not speculate. If the answer is missing, respond exactly with 'Information not found in retrieved papers.'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                "Answer requirements:\n"
                "- Give a concise, technically accurate answer.\n"
                "- Mention paper titles when useful.\n"
                "- Do not use knowledge outside the context.\n\n"
                f"Retrieved context:\n{context}"
            ),
        },
    ]


def build_comparison_messages(query: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You compare research papers using only retrieved evidence. "
                "Do not infer unsupported differences. If evidence is missing, state 'Insufficient information'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Comparison request: {query}\n\n"
                "Return a structured markdown comparison using this shape:\n"
                "Comparison\n\n"
                "Paper Title\n"
                "Method\n"
                "Advantages\n"
                "Limitations\n\n"
                "Include a short overall takeaway at the end. Cite paper titles directly in the text.\n\n"
                f"Retrieved context grouped by paper:\n{context}"
            ),
        },
    ]
