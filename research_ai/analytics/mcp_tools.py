from __future__ import annotations

from pathlib import Path
from typing import Iterable

from research_ai.analytics.citation_metrics import get_citation_count
from research_ai.analytics.trend_analysis import identify_emerging_topics, topic_frequency
from research_ai.indexing.semantic_search import semantic_search
from research_ai.models import ResearchPaper

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"


def paper_metadata_lookup(query: str, papers: Iterable[ResearchPaper], graph=None, allow_external: bool = True) -> dict[str, object] | None:
    normalized_query = query.lower().strip()
    for paper in papers:
        if normalized_query == (paper.doi or "").lower() or normalized_query in paper.title.lower():
            citation_count = get_citation_count(graph, paper.paper_id) if graph is not None else 0
            return {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "venue": paper.venue,
                "citation_count": citation_count,
                "source": "local",
            }
    if allow_external:
        external = _lookup_external_metadata(query)
        if external:
            return external
    return None


def discover_related_work(
    paper_id: str,
    papers: Iterable[ResearchPaper],
    graph=None,
    *,
    index_dir: str | Path = "data/indices",
    top_k: int = 5,
) -> dict[str, object]:
    paper_lookup = {paper.paper_id: paper for paper in papers}
    paper = paper_lookup.get(paper_id)
    if paper is None:
        raise ValueError(f"Unknown paper_id: {paper_id}")

    semantic_neighbors = [item for item in semantic_search(paper.title, top_k=top_k + 3, index_dir=index_dir) if item.get("paper_id") != paper_id][:top_k]

    citation_neighbors: dict[str, list[dict[str, object]]] = {"references": [], "cited_by": []}
    if graph is not None and paper_id in graph:
        citation_neighbors["references"] = [{"paper_id": neighbor_id, "paper_title": paper_lookup.get(neighbor_id).title if neighbor_id in paper_lookup else neighbor_id} for neighbor_id in graph.successors(paper_id)]
        citation_neighbors["cited_by"] = [{"paper_id": neighbor_id, "paper_title": paper_lookup.get(neighbor_id).title if neighbor_id in paper_lookup else neighbor_id} for neighbor_id in graph.predecessors(paper_id)]

    return {
        "paper_id": paper_id,
        "paper_title": paper.title,
        "semantic_neighbors": semantic_neighbors,
        "citation_neighbors": citation_neighbors,
    }


def trend_analytics_tool(topic: str, papers: Iterable[ResearchPaper], extracted_keywords: dict[str, list[str]] | None = None) -> dict[str, object]:
    paper_list = list(papers)
    frequency = topic_frequency(paper_list, topic, extracted_keywords=extracted_keywords)
    emerging_topics = identify_emerging_topics(paper_list, extracted_keywords=extracted_keywords, top_k=20)
    matching_emerging = next((item for item in emerging_topics if topic.lower() in str(item["topic"]).lower()), None)

    example_papers = []
    for paper in paper_list:
        keywords = extracted_keywords.get(paper.paper_id, paper.keywords) if extracted_keywords else paper.keywords
        if topic.lower() in " ".join(keywords).lower():
            example_papers.append(paper.title)

    return {
        "topic": topic,
        "publication_frequency": frequency,
        "example_papers": example_papers[:5],
        "trend_growth": matching_emerging["growth_rate"] if matching_emerging else 0.0,
    }


def _lookup_external_metadata(query: str) -> dict[str, object] | None:
    if requests is None:
        return None
    return _lookup_crossref(query) or _lookup_openalex(query)


def _lookup_crossref(query: str) -> dict[str, object] | None:
    try:
        response = requests.get(CROSSREF_API, params={"query.bibliographic": query, "rows": 1}, timeout=8)
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        if not items:
            return None
        item = items[0]
        issued = item.get("issued", {}).get("date-parts", [[None]])
        year = issued[0][0] if issued and issued[0] else None
        venue = (item.get("container-title") or [None])[0]
        return {
            "paper_id": None,
            "title": (item.get("title") or [query])[0],
            "year": year,
            "venue": venue,
            "citation_count": item.get("is-referenced-by-count", 0),
            "doi": item.get("DOI"),
            "source": "crossref",
        }
    except Exception:
        return None


def _lookup_openalex(query: str) -> dict[str, object] | None:
    try:
        response = requests.get(OPENALEX_API, params={"search": query, "per-page": 1}, timeout=8)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None
        item = results[0]
        return {
            "paper_id": None,
            "title": item.get("display_name", query),
            "year": item.get("publication_year"),
            "venue": (item.get("primary_location") or {}).get("source", {}).get("display_name"),
            "citation_count": item.get("cited_by_count", 0),
            "doi": item.get("doi"),
            "source": "openalex",
        }
    except Exception:
        return None
