from __future__ import annotations

from pathlib import Path
from typing import Any

from research_ai.analytics.citation_graph import build_citation_graph
from research_ai.analytics.keyword_extractor import enrich_papers_with_keywords, extract_keywords_for_papers
from research_ai.analytics.trend_analysis import aggregate_by_venue, aggregate_topic_trends, identify_emerging_topics
from research_ai.config import get_settings
from research_ai.indexing.index_builder import index_papers, load_papers_from_json
from research_ai.models import ResearchPaper
from research_ai.parsing.paper_builder import batch_parse_papers

SETTINGS = get_settings()
DATA_DIR = SETTINGS.data_dir
PROCESSED_DIR = SETTINGS.processed_dir
INDEX_DIR = SETTINGS.indices_dir
RAW_PDF_DIR = SETTINGS.raw_pdf_dir


def load_library() -> list[ResearchPaper]:
    if not PROCESSED_DIR.exists():
        return []
    papers = load_papers_from_json(PROCESSED_DIR)
    enrich_papers_with_keywords(papers)
    return papers


def refresh_library(pdf_dir: str | Path | None = None, rebuild_index: bool = True) -> list[ResearchPaper]:
    source_dir = Path(pdf_dir) if pdf_dir else Path.cwd()
    papers = batch_parse_papers(source_dir, export_json=True, output_dir=PROCESSED_DIR)
    enrich_papers_with_keywords(papers)
    if rebuild_index and papers:
        index_papers(papers, index_dir=INDEX_DIR, provider=SETTINGS.embedding_provider, model_name=SETTINGS.embedding_model)
    return papers


def paper_lookup(papers: list[ResearchPaper]) -> dict[str, ResearchPaper]:
    return {paper.paper_id: paper for paper in papers}


def build_analytics_snapshot(papers: list[ResearchPaper]) -> dict[str, Any]:
    if not papers:
        return {"graph": None, "relations": [], "keyword_map": {}, "topic_trends": {}, "emerging_topics": [], "venue_counts": {}}

    keyword_map = extract_keywords_for_papers(papers)
    graph, relations = build_citation_graph(papers)
    return {
        "graph": graph,
        "relations": relations,
        "keyword_map": keyword_map,
        "topic_trends": aggregate_topic_trends(papers, extracted_keywords=keyword_map),
        "emerging_topics": identify_emerging_topics(papers, extracted_keywords=keyword_map, top_k=12),
        "venue_counts": aggregate_by_venue(papers),
    }


def paper_filter_options(papers: list[ResearchPaper]) -> dict[str, list[Any]]:
    years = sorted({paper.year for paper in papers if paper.year is not None})
    venues = sorted({paper.venue for paper in papers if paper.venue})
    keywords = sorted({keyword for paper in papers for keyword in paper.keywords if keyword})
    return {"years": years, "venues": venues, "keywords": keywords}


def filter_papers(papers: list[ResearchPaper], *, year_range: tuple[int | None, int | None] | None = None, keyword: str | None = None, venue: str | None = None) -> list[ResearchPaper]:
    filtered = papers
    if year_range and any(value is not None for value in year_range):
        min_year, max_year = year_range
        filtered = [paper for paper in filtered if paper.year is not None and (min_year is None or paper.year >= min_year) and (max_year is None or paper.year <= max_year)]
    if keyword:
        keyword_value = keyword.lower()
        filtered = [paper for paper in filtered if any(keyword_value in item.lower() for item in paper.keywords)]
    if venue:
        venue_value = venue.lower()
        filtered = [paper for paper in filtered if paper.venue and venue_value in paper.venue.lower()]
    return filtered


def system_status(papers: list[ResearchPaper]) -> dict[str, Any]:
    return {
        "paper_count": len(papers),
        "processed_ready": PROCESSED_DIR.exists() and any(PROCESSED_DIR.glob("*.json")),
        "index_ready": (INDEX_DIR / "paper_chunks.faiss").exists(),
        "groq_ready": _has_groq_secret(),
        "processed_dir": str(PROCESSED_DIR),
        "index_dir": str(INDEX_DIR),
    }


def _has_groq_secret() -> bool:
    try:
        import streamlit as st
        return bool(st.secrets.get("GROQ_API_KEY"))
    except Exception:
        return False
