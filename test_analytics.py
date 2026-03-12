from __future__ import annotations

from pathlib import Path

from research_ai.analytics.citation_graph import build_citation_graph
from research_ai.analytics.citation_metrics import get_most_influential_papers
from research_ai.analytics.keyword_extractor import enrich_papers_with_keywords
from research_ai.analytics.mcp_tools import discover_related_work
from research_ai.analytics.trend_analysis import identify_emerging_topics
from research_ai.indexing.index_builder import load_papers_from_json


def main() -> None:
    papers = load_papers_from_json(Path("data") / "processed")
    if not papers:
        raise ValueError("No parsed papers found in data/processed")

    enrich_papers_with_keywords(papers)
    graph, relations = build_citation_graph(papers)

    print("Citation Relationships")
    for relation in relations[:10]:
        print(f"- {relation.citing_paper_id} -> {relation.cited_title} | resolved={relation.resolved}")

    print("\nTop Influential Papers")
    for item in get_most_influential_papers(graph, papers, top_k=5):
        print(f"- {item['paper_title']} | citations={item['citation_count']} | centrality={item['in_degree_centrality']:.4f}")

    print("\nEmerging Topics")
    for item in identify_emerging_topics(papers, top_k=5):
        print(f"- {item['topic']} | growth={item['growth_rate']:.4f} | examples={item['example_papers']}")

    print("\nRelated Work")
    related = discover_related_work(papers[0].paper_id, papers, graph=graph)
    print(f"Paper: {related['paper_title']}")
    print("Semantic neighbors:")
    for item in related["semantic_neighbors"]:
        print(f"- {item['paper_title']} | {item['section']}")
    print("Citation neighbors:")
    print(related["citation_neighbors"])


if __name__ == "__main__":
    main()
