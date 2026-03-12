from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from research_ai.analytics.keyword_extractor import extract_keywords_for_papers
from research_ai.models import ResearchPaper


def aggregate_topic_trends(
    papers: Iterable[ResearchPaper],
    extracted_keywords: dict[str, list[str]] | None = None,
) -> dict[str, dict[int, int]]:
    paper_list = list(papers)
    keyword_map = extracted_keywords or extract_keywords_for_papers(paper_list)
    trends: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for paper in paper_list:
        if paper.year is None:
            continue
        for topic in _paper_topics(paper, keyword_map):
            trends[topic][paper.year] += 1

    return {topic: dict(sorted(years.items())) for topic, years in trends.items()}


def aggregate_by_venue(papers: Iterable[ResearchPaper]) -> dict[str, int]:
    venue_counts: dict[str, int] = defaultdict(int)
    for paper in papers:
        if paper.venue:
            venue_counts[paper.venue] += 1
    return dict(sorted(venue_counts.items(), key=lambda item: item[1], reverse=True))


def identify_emerging_topics(
    papers: Iterable[ResearchPaper],
    extracted_keywords: dict[str, list[str]] | None = None,
    top_k: int = 10,
    recent_window: int = 2,
) -> list[dict[str, object]]:
    paper_list = [paper for paper in papers if paper.year is not None]
    if not paper_list:
        return []

    keyword_map = extracted_keywords or extract_keywords_for_papers(paper_list)
    trends = aggregate_topic_trends(paper_list, extracted_keywords=keyword_map)
    latest_year = max(paper.year for paper in paper_list if paper.year is not None)
    previous_start = latest_year - (recent_window * 2) + 1
    recent_start = latest_year - recent_window + 1

    results: list[dict[str, object]] = []
    for topic, yearly_counts in trends.items():
        recent_count = sum(count for year, count in yearly_counts.items() if year >= recent_start)
        previous_count = sum(count for year, count in yearly_counts.items() if previous_start <= year < recent_start)
        growth_rate = round((recent_count - previous_count) / max(previous_count, 1), 4)
        if recent_count <= 0:
            continue
        example_papers = [paper.title for paper in paper_list if topic in _paper_topics(paper, keyword_map)][:3]
        results.append(
            {
                "topic": topic,
                "recent_count": recent_count,
                "previous_count": previous_count,
                "growth_rate": growth_rate,
                "example_papers": example_papers,
            }
        )

    results.sort(key=lambda item: (item["growth_rate"], item["recent_count"]), reverse=True)
    return results[:top_k]


def topic_frequency(
    papers: Iterable[ResearchPaper],
    topic: str,
    extracted_keywords: dict[str, list[str]] | None = None,
) -> dict[int, int]:
    paper_list = list(papers)
    keyword_map = extracted_keywords or extract_keywords_for_papers(paper_list)
    normalized_topic = topic.lower()
    counts: dict[int, int] = defaultdict(int)

    for paper in paper_list:
        if paper.year is None:
            continue
        if any(normalized_topic in item.lower() or item.lower() in normalized_topic for item in _paper_topics(paper, keyword_map)):
            counts[paper.year] += 1

    return dict(sorted(counts.items()))


def _paper_topics(paper: ResearchPaper, keyword_map: dict[str, list[str]]) -> list[str]:
    extracted = keyword_map.get(paper.paper_id, [])
    combined = list(dict.fromkeys([*paper.keywords, *extracted]))
    return combined[:10]
