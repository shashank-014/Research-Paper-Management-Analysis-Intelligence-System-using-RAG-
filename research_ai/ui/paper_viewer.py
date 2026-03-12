from __future__ import annotations

import streamlit as st

from research_ai.analytics.citation_metrics import get_citation_count
from research_ai.rag.summarizer import summarize_paper

SECTION_ORDER = ["abstract", "introduction", "related_work", "methods", "results", "discussion", "limitations", "conclusion", "references", "appendix"]


def render(papers, paper_lookup, analytics_snapshot):
    st.title("Paper Viewer")
    if not papers:
        st.info("Load parsed papers to view paper details.")
        return None

    current_id = st.session_state.get("selected_paper_id", papers[0].paper_id)
    labels = {paper.paper_id: f"{paper.title} ({paper.year or 'n/a'})" for paper in papers}
    selected_id = st.selectbox("Select a paper", options=list(labels.keys()), format_func=lambda item: labels[item], index=max(list(labels.keys()).index(current_id), 0) if current_id in labels else 0)
    st.session_state["selected_paper_id"] = selected_id
    paper = paper_lookup[selected_id]

    st.header(paper.title)
    st.caption(f"{', '.join(paper.authors) if paper.authors else 'Unknown authors'}")
    st.caption(f"Year: {paper.year or 'Unknown'} | Venue: {paper.venue or 'Unknown'}")
    if paper.doi or paper.arxiv_id:
        st.caption(f"DOI: {paper.doi or 'n/a'} | arXiv: {paper.arxiv_id or 'n/a'}")

    col1, col2, col3 = st.columns(3)
    graph = analytics_snapshot.get("graph")
    col1.metric("References", len(paper.citations))
    col2.metric("Cited By", get_citation_count(graph, paper.paper_id) if graph is not None else 0)
    col3.metric("Keywords", len(paper.keywords))

    if paper.abstract:
        st.subheader("Abstract")
        st.write(paper.abstract)

    with st.expander("Auto-generated summary", expanded=True):
        if st.button("Generate Summary", key=f"summary-{paper.paper_id}"):
            try:
                st.session_state[f"summary:{paper.paper_id}"] = summarize_paper(paper)
                st.session_state.pop(f"summary_error:{paper.paper_id}", None)
            except Exception as exc:
                st.session_state[f"summary_error:{paper.paper_id}"] = str(exc)
        summary = st.session_state.get(f"summary:{paper.paper_id}")
        summary_error = st.session_state.get(f"summary_error:{paper.paper_id}")
        if summary_error:
            st.error(f"Summary generation failed: {summary_error}")
        elif summary:
            st.markdown("**Short Summary**")
            for bullet in summary.get("short_summary", []):
                st.write(f"- {bullet}")
            structured = summary.get("structured_summary", {})
            if structured:
                st.markdown("**Structured Summary**")
                for label, value in structured.items():
                    title = label.replace("_", " ").title()
                    st.write(f"{title}: {', '.join(value) if isinstance(value, list) else value}")

    if graph is not None and paper.paper_id in graph:
        refs = [paper_lookup[item].title for item in graph.successors(paper.paper_id) if item in paper_lookup]
        cited_by = [paper_lookup[item].title for item in graph.predecessors(paper.paper_id) if item in paper_lookup]
        with st.expander("Citation Information", expanded=False):
            st.markdown("**References In Library**")
            if refs:
                for title in refs:
                    st.write(f"- {title}")
            else:
                st.write("No resolved references found in the current library.")
            st.markdown("**Cited By In Library**")
            if cited_by:
                for title in cited_by:
                    st.write(f"- {title}")
            else:
                st.write("No incoming citations found in the current library.")

    st.subheader("Parsed Sections")
    ordered_sections = sorted(paper.sections, key=lambda section: (SECTION_ORDER.index(section.section_name) if section.section_name in SECTION_ORDER else 99, section.page_start or 999))
    for section in ordered_sections:
        with st.expander(section.section_name.replace("_", " ").title(), expanded=section.section_name.lower() in {"introduction", "methods", "results"}):
            st.caption(f"Pages: {section.page_start or '-'} to {section.page_end or '-'}")
            st.write(section.content or "No content extracted.")

    if st.button("Ask Questions About This Paper", key=f"ask-{paper.paper_id}"):
        st.session_state["chat_paper_id"] = paper.paper_id
    return paper.paper_id
