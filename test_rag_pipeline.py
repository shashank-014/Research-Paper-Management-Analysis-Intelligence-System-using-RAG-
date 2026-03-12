from __future__ import annotations

from pathlib import Path

from research_ai.indexing.index_builder import load_papers_from_json
from research_ai.rag.comparison_engine import compare_papers
from research_ai.rag.rag_pipeline import answer_question
from research_ai.rag.summarizer import summarize_paper


def main() -> None:
    processed_dir = Path("data") / "processed"
    papers = load_papers_from_json(processed_dir)
    if not papers:
        raise ValueError("No parsed papers found in data/processed")

    try:
        summary = summarize_paper(papers[0])
        print("Summary test")
        print(f"Paper: {summary['paper_title']}")
        for bullet in summary["short_summary"]:
            print(f"- {bullet}")

        questions = ["What datasets are used in this paper?", "What problem does this paper solve?"]
        for query in questions:
            result = answer_question(query, paper_ids=[papers[0].paper_id])
            print(f"\nQuestion: {query}")
            print(result["answer"])
            print("Sources:")
            for source in result["sources"]:
                print(f"- {source['paper_title']} | {source['section']}")

        comparison = compare_papers("Compare diffusion models and GANs.")
        print("\nComparison test")
        print(comparison["comparison"])
    except Exception as exc:
        print("RAG test could not run end-to-end.")
        print("Make sure parsed papers exist, the FAISS index is built, and st.secrets['GROQ_API_KEY'] is configured.")
        print(f"Details: {exc}")


if __name__ == "__main__":
    main()
