from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from research_ai.models import ResearchPaper

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "using", "we", "with", "our", "these", "those",
    "paper", "study", "approach", "method", "results", "show", "based", "used", "use", "proposed",
}


def extract_keywords_for_papers(papers: Iterable[ResearchPaper], top_k: int = 8) -> dict[str, list[str]]:
    paper_list = list(papers)
    corpus_terms = [_candidate_terms(_paper_text(paper)) for paper in paper_list]
    document_frequency = Counter()
    for terms in corpus_terms:
        document_frequency.update(set(terms))

    total_docs = max(len(paper_list), 1)
    extracted: dict[str, list[str]] = {}
    for paper in paper_list:
        scores = _score_terms(_candidate_terms(_paper_text(paper)), document_frequency, total_docs)
        extracted[paper.paper_id] = [term for term, _ in scores[:top_k]]
    return extracted


def enrich_papers_with_keywords(papers: Iterable[ResearchPaper], top_k: int = 8) -> dict[str, list[str]]:
    paper_list = list(papers)
    extracted = extract_keywords_for_papers(paper_list, top_k=top_k)
    for paper in paper_list:
        keywords = extracted.get(paper.paper_id, [])
        merged = list(dict.fromkeys([*paper.keywords, *keywords]))
        paper.keywords = merged[: max(len(paper.keywords), top_k)]
        paper.metadata["extracted_keywords"] = keywords
    return extracted


def _paper_text(paper: ResearchPaper) -> str:
    parts: list[str] = []
    if paper.abstract:
        parts.append(paper.abstract)
    for section in paper.sections:
        if section.section_name.lower() == "introduction":
            parts.append(section.content)
    return "\n".join(parts)


def _candidate_terms(text: str) -> list[str]:
    words = [token for token in _tokenize(text) if token not in STOPWORDS]
    bigrams = [f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1)]
    return words + bigrams


def _score_terms(terms: list[str], document_frequency: Counter[str], total_docs: int) -> list[tuple[str, float]]:
    frequencies = Counter(terms)
    scores: list[tuple[str, float]] = []
    for term, count in frequencies.items():
        if len(term) < 3:
            continue
        if term.count(" ") == 0 and count < 2:
            continue
        idf = math.log((total_docs + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
        scores.append((term, round(count * idf, 4)))
    scores.sort(key=lambda item: (item[1], len(item[0].split()), len(item[0])), reverse=True)
    return scores


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower()) if len(token) > 2]
