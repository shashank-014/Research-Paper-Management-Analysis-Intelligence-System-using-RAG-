from __future__ import annotations

import streamlit as st

from research_ai.ui import citation_explorer, paper_comparison, paper_dashboard, paper_viewer, research_chat, trend_dashboard
from research_ai.ui.backend import build_analytics_snapshot, load_library, paper_lookup, refresh_library, system_status

st.set_page_config(page_title="Research Intelligence System", layout="wide")


@st.cache_data(show_spinner=False)
def _load_papers_cached():
    return load_library()


@st.cache_resource(show_spinner=False)
def _load_analytics_cached(papers_signature: tuple[str, ...]):
    papers = _load_papers_cached()
    try:
        return build_analytics_snapshot(papers)
    except Exception as exc:
        return {
            "graph": None,
            "relations": [],
            "keyword_map": {},
            "topic_trends": {},
            "emerging_topics": [],
            "venue_counts": {},
            "error": str(exc),
        }


def main() -> None:
    st.sidebar.title("Research Intelligence")
    st.sidebar.caption("Semantic discovery, grounded QA, and research analytics in one workspace.")

    papers = _load_papers_cached()
    lookup = paper_lookup(papers)
    analytics_snapshot = _load_analytics_cached(tuple(sorted(paper.paper_id for paper in papers)))
    status = system_status(papers)

    if analytics_snapshot.get("error"):
        st.sidebar.warning(f"Analytics limited: {analytics_snapshot['error']}")

    with st.sidebar.expander("System Status", expanded=True):
        st.write(f"Parsed papers: {status['paper_count']}")
        st.write(f"Processed JSON ready: {status['processed_ready']}")
        st.write(f"FAISS index ready: {status['index_ready']}")
        st.write(f"Groq key ready: {status['groq_ready']}")
        st.caption(status["processed_dir"])
        st.caption(status["index_dir"])
        if st.button("Refresh Papers And Index", key="refresh-system"):
            try:
                refresh_library(rebuild_index=True)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Refresh failed: {exc}")

    page = st.sidebar.radio("Navigate", options=["Paper Dashboard", "Paper Viewer", "Research Chat", "Paper Comparison", "Citation Explorer", "Trend Dashboard"])

    with st.sidebar.expander("Demo Flow", expanded=False):
        st.write("1. Refresh papers and index if needed.")
        st.write("2. Search papers from the dashboard.")
        st.write("3. Open a paper in the viewer.")
        st.write("4. Generate a summary and ask questions.")
        st.write("5. Compare papers and inspect citation/trend analytics.")

    if page == "Paper Dashboard":
        selected = paper_dashboard.render(papers, status)
        if selected:
            st.sidebar.success("Paper selected. Open the viewer to inspect details.")
    elif page == "Paper Viewer":
        paper_viewer.render(papers, lookup, analytics_snapshot)
    elif page == "Research Chat":
        research_chat.render(papers, lookup)
    elif page == "Paper Comparison":
        paper_comparison.render(papers, lookup)
    elif page == "Citation Explorer":
        citation_explorer.render(papers, lookup, analytics_snapshot)
    elif page == "Trend Dashboard":
        trend_dashboard.render(papers, analytics_snapshot)


if __name__ == "__main__":
    main()
