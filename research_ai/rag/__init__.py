"""RAG and summarization modules for research paper analysis."""

from .comparison_engine import compare_papers
from .rag_pipeline import answer_question, retrieve_context
from .summarizer import summarize_paper

__all__ = ["answer_question", "compare_papers", "retrieve_context", "summarize_paper"]
