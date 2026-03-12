"""Citation intelligence and trend analytics modules."""

from .citation_graph import build_citation_graph, extract_citation_relations, get_cited_by
from .citation_metrics import get_citation_clusters, get_citation_count, get_most_influential_papers
from .keyword_extractor import enrich_papers_with_keywords, extract_keywords_for_papers
from .mcp_tools import discover_related_work, paper_metadata_lookup, trend_analytics_tool
from .trend_analysis import aggregate_topic_trends, identify_emerging_topics

__all__ = [
    "aggregate_topic_trends",
    "build_citation_graph",
    "discover_related_work",
    "enrich_papers_with_keywords",
    "extract_citation_relations",
    "extract_keywords_for_papers",
    "get_citation_clusters",
    "get_citation_count",
    "get_cited_by",
    "get_most_influential_papers",
    "identify_emerging_topics",
    "paper_metadata_lookup",
    "trend_analytics_tool",
]
