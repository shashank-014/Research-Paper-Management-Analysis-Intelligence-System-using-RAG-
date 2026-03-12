from __future__ import annotations

import hashlib
import re
from typing import Iterable

from pydantic import BaseModel, Field

from research_ai.models import ResearchPaper


class PaperChunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    section_name: str
    text: str
    year: int | None = None
    venue: str | None = None
    keywords: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    token_count: int
    chunk_index: int


def chunk_paper(
    paper: ResearchPaper,
    max_tokens: int = 700,
    overlap_tokens: int = 80,
) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []

    for section in paper.sections:
        section_chunks = _chunk_section_text(section.content, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for chunk_index, chunk_text in enumerate(section_chunks):
            chunk_id = _build_chunk_id(paper.paper_id, section.section_name, chunk_index, chunk_text)
            chunks.append(
                PaperChunk(
                    chunk_id=chunk_id,
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    section_name=section.section_name,
                    text=chunk_text,
                    year=paper.year,
                    venue=paper.venue,
                    keywords=paper.keywords,
                    authors=paper.authors,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    token_count=_estimate_tokens(chunk_text),
                    chunk_index=chunk_index,
                )
            )

    return chunks


def chunk_papers(
    papers: Iterable[ResearchPaper],
    max_tokens: int = 700,
    overlap_tokens: int = 80,
) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    for paper in papers:
        chunks.extend(chunk_paper(paper, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    return chunks


def _chunk_section_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    sentences = [part.strip() for part in re.split(r"(?<=[\.!?])\s+", cleaned) if part.strip()]
    if not sentences:
        return [_trim_words(cleaned, max_tokens)]

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)
        if sentence_tokens >= max_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_tokens = 0
            chunks.extend(_split_long_text(sentence, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
            continue

        if current_tokens + sentence_tokens > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences).strip())
            overlap = _tail_words(" ".join(current_sentences), overlap_tokens)
            current_sentences = [overlap] if overlap else []
            current_tokens = _estimate_tokens(overlap)

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [chunk for chunk in chunks if chunk]


def _split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(max_tokens - overlap_tokens, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_tokens]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + max_tokens >= len(words):
            break

    return chunks


def _tail_words(text: str, token_count: int) -> str:
    words = text.split()
    if token_count <= 0 or len(words) <= token_count:
        return text if len(words) <= token_count else ""
    return " ".join(words[-token_count:])


def _trim_words(text: str, max_tokens: int) -> str:
    words = text.split()
    return " ".join(words[:max_tokens])


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.15))


def _build_chunk_id(paper_id: str, section_name: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{paper_id}:{section_name}:{chunk_index}:{digest}"
