from __future__ import annotations

from pathlib import Path

from research_ai.indexing.index_builder import index_papers, load_papers_from_json
from research_ai.indexing.semantic_search import semantic_search
from research_ai.parsing.paper_builder import batch_parse_papers


def main() -> None:
    processed_dir = Path("data") / "processed"
    if list(processed_dir.glob("*.json")):
        papers = load_papers_from_json(processed_dir)
    else:
        papers = batch_parse_papers(Path("."), export_json=True)

    index_papers(papers, index_dir=Path("data") / "indices")

    queries = [
        "transformer efficiency",
        "reinforcement learning robotics",
        "attention mechanism foundational work",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = semantic_search(query, top_k=5, index_dir=Path("data") / "indices")
        for item in results:
            print(f"- {item['paper_title']} | {item['section']} | score={item['score']:.4f}")
            print(f"  {item['text'][:220]}...")


if __name__ == "__main__":
    main()
