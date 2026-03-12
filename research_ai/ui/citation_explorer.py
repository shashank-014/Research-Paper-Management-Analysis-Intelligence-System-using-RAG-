from __future__ import annotations

import pandas as pd
import streamlit as st

from research_ai.analytics.citation_metrics import get_citation_clusters, get_most_influential_papers
from research_ai.analytics.mcp_tools import discover_related_work


def render(papers, paper_lookup, analytics_snapshot):
    st.title("Citation Explorer")
    graph = analytics_snapshot.get("graph")
    if analytics_snapshot.get("error"):
        st.warning(f"Citation analytics unavailable: {analytics_snapshot['error']}")
    if not papers or graph is None:
        st.info("Citation analytics are not available until parsed papers and graph dependencies are ready.")
        return

    influential = get_most_influential_papers(graph, papers, top_k=10)
    st.subheader("Top Influential Papers")
    st.dataframe(pd.DataFrame(influential), use_container_width=True)

    labels = {paper.paper_id: paper.title for paper in papers}
    selected_id = st.selectbox(
        "Inspect citation neighborhood",
        options=list(labels.keys()),
        format_func=lambda item: labels[item],
        index=max(list(labels.keys()).index(st.session_state.get("selected_paper_id")), 0) if st.session_state.get("selected_paper_id") in labels else 0,
    )

    related = discover_related_work(selected_id, papers, graph=graph)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**References**")
        for item in related["citation_neighbors"]["references"]:
            st.write(f"- {item['paper_title']}")
    with col2:
        st.markdown("**Cited By**")
        for item in related["citation_neighbors"]["cited_by"]:
            st.write(f"- {item['paper_title']}")

    st.markdown("**Semantic Neighbors**")
    for item in related["semantic_neighbors"]:
        with st.container(border=True):
            st.write(f"{item['paper_title']} | {item['section']}")
            st.caption(item['text'][:260] + ("..." if len(item['text']) > 260 else ""))

    clusters = get_citation_clusters(graph)
    st.subheader("Citation Clusters")
    st.dataframe(pd.DataFrame(clusters), use_container_width=True)
