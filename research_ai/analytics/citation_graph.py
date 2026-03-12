from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Iterable

from research_ai.models import CitationRelation, ResearchPaper

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


def extract_citation_relations(papers: Iterable[ResearchPaper], fuzzy_threshold: float = 0.92) -> list[CitationRelation]:
    paper_list = list(papers)
    title_index = {_normalize_title(paper.title): paper for paper in paper_list}
    relations: list[CitationRelation] = []

    for paper in paper_list:
        for citation in paper.citations:
            cited_title = citation.cited_title or citation.raw_reference_text
            resolved_paper = _resolve_cited_paper(cited_title, title_index, fuzzy_threshold=fuzzy_threshold)
            relation = citation.model_copy(deep=True)
            relation.citing_paper_id = paper.paper_id
            relation.cited_title = cited_title
            if resolved_paper is not None:
                relation.cited_paper_id = resolved_paper.paper_id
                relation.resolved = True
                relation.confidence = 1.0 if _normalize_title(cited_title) == _normalize_title(resolved_paper.title) else fuzzy_threshold
            relations.append(relation)

    return relations


def build_citation_graph(papers: Iterable[ResearchPaper], fuzzy_threshold: float = 0.92):
    if nx is None:
        raise ImportError("networkx is required for citation graph analytics")

    paper_list = list(papers)
    graph = nx.DiGraph()
    relations = extract_citation_relations(paper_list, fuzzy_threshold=fuzzy_threshold)
    paper_lookup = {paper.paper_id: paper for paper in paper_list}

    for paper in paper_list:
        graph.add_node(
            paper.paper_id,
            title=paper.title,
            year=paper.year,
            venue=paper.venue,
            keywords=paper.keywords,
        )

    for relation in relations:
        if not relation.cited_paper_id:
            continue
        if relation.citing_paper_id == relation.cited_paper_id:
            continue
        cited_paper = paper_lookup.get(relation.cited_paper_id)
        graph.add_edge(
            relation.citing_paper_id,
            relation.cited_paper_id,
            cited_title=relation.cited_title,
            cited_year=cited_paper.year if cited_paper else None,
            raw_reference_text=relation.raw_reference_text,
            confidence=relation.confidence,
        )

    logger.info("Built citation graph with %s nodes and %s edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph, relations


def get_cited_by(graph, paper_id: str) -> list[str]:
    if paper_id not in graph:
        return []
    return list(graph.predecessors(paper_id))


def _resolve_cited_paper(
    cited_title: str | None,
    title_index: dict[str, ResearchPaper],
    fuzzy_threshold: float,
) -> ResearchPaper | None:
    if not cited_title:
        return None

    normalized = _normalize_title(cited_title)
    if normalized in title_index:
        return title_index[normalized]

    best_match: ResearchPaper | None = None
    best_score = 0.0
    for known_title, paper in title_index.items():
        score = SequenceMatcher(a=normalized, b=known_title).ratio()
        if score > best_score:
            best_score = score
            best_match = paper

    if best_score >= fuzzy_threshold:
        return best_match
    return None


def _normalize_title(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", title.lower())
    return re.sub(r"\s+", " ", cleaned).strip()
