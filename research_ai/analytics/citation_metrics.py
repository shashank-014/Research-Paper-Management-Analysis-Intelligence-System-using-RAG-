from __future__ import annotations

from typing import Iterable

from research_ai.models import ResearchPaper


def get_citation_count(graph, paper_id: str) -> int:
    if paper_id not in graph:
        return 0
    return int(graph.in_degree(paper_id))


def get_most_influential_papers(graph, papers: Iterable[ResearchPaper], top_k: int = 10) -> list[dict[str, object]]:
    paper_lookup = {paper.paper_id: paper for paper in papers}
    centrality = _safe_in_degree_centrality(graph)
    ranked: list[dict[str, object]] = []

    for paper_id in graph.nodes:
        paper = paper_lookup.get(paper_id)
        ranked.append(
            {
                "paper_id": paper_id,
                "paper_title": paper.title if paper else paper_id,
                "citation_count": int(graph.in_degree(paper_id)),
                "in_degree_centrality": float(centrality.get(paper_id, 0.0)),
                "year": paper.year if paper else None,
                "venue": paper.venue if paper else None,
            }
        )

    ranked.sort(key=lambda item: (item["citation_count"], item["in_degree_centrality"]), reverse=True)
    return ranked[:top_k]


def get_citation_clusters(graph) -> list[dict[str, object]]:
    undirected = graph.to_undirected()
    clusters: list[dict[str, object]] = []
    for index, component in enumerate(sorted(_connected_components(undirected), key=len, reverse=True), start=1):
        clusters.append(
            {
                "cluster_id": index,
                "size": len(component),
                "paper_ids": sorted(component),
            }
        )
    return clusters


def _safe_in_degree_centrality(graph) -> dict[str, float]:
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("networkx is required for citation metrics") from exc

    return nx.in_degree_centrality(graph)


def _connected_components(graph):
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("networkx is required for citation metrics") from exc

    return nx.connected_components(graph)
