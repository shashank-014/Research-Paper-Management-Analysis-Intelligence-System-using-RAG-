from __future__ import annotations

import streamlit as st

from research_ai.indexing.semantic_search import semantic_search
from research_ai.ui.backend import INDEX_DIR, filter_papers, paper_filter_options


def render(papers, status):
    st.title("Research Paper Library")
    st.caption("Browse indexed papers, filter the library, and run semantic discovery queries.")

    if not papers:
        st.warning("No parsed papers found yet. Use the sidebar refresh action to parse PDFs and build the index.")
        return

    info_cols = st.columns(4)
    info_cols[0].metric("Parsed Papers", status["paper_count"])
    info_cols[1].metric("Processed JSON", "Ready" if status["processed_ready"] else "Missing")
    info_cols[2].metric("FAISS Index", "Ready" if status["index_ready"] else "Missing")
    info_cols[3].metric("Groq Key", "Ready" if status["groq_ready"] else "Missing")

    options = paper_filter_options(papers)
    year_values = options["years"]
    min_year = year_values[0] if year_values else None
    max_year = year_values[-1] if year_values else None

    with st.sidebar:
        st.subheader("Library Filters")
        selected_years = (min_year, max_year)
        if min_year is not None and max_year is not None:
            selected_years = st.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
        selected_keyword = st.selectbox("Keyword", options=["All"] + options["keywords"], index=0)
        selected_venue = st.selectbox("Venue", options=["All"] + options["venues"], index=0)

    query = st.text_input("Semantic search", placeholder="recent work on transformer efficiency")

    filtered = filter_papers(
        papers,
        year_range=selected_years if min_year is not None else None,
        keyword=None if selected_keyword == "All" else selected_keyword,
        venue=None if selected_venue == "All" else selected_venue,
    )

    search_results = []
    if query.strip() and status["index_ready"]:
        search_filters = {}
        if min_year is not None:
            search_filters["year"] = {"min": selected_years[0], "max": selected_years[1]}
        if selected_keyword != "All":
            search_filters["keywords"] = [selected_keyword]
        if selected_venue != "All":
            search_filters["venue"] = selected_venue
        try:
            raw_results = semantic_search(query, filters=search_filters or None, top_k=18, index_dir=INDEX_DIR)
            search_results = _group_search_results(raw_results)
        except Exception as exc:
            st.error(f"Semantic search is unavailable: {exc}")
    elif query.strip() and not status["index_ready"]:
        st.info("Build the FAISS index from the sidebar before using semantic search.")

    if search_results:
        st.subheader("Semantic Matches")
        for item in search_results:
            with st.container(border=True):
                st.markdown(f"**{item['paper_title']}**")
                st.caption(f"Best score: {item['score']:.4f} | Year: {item.get('year') or 'Unknown'} | Venue: {item.get('venue') or 'Unknown'}")
                for section in item["sections"]:
                    st.write(f"**{section['section']}**: {section['text']}")
                if st.button("Open Paper", key=f"search-open-{item['paper_id']}"):
                    st.session_state["selected_paper_id"] = item["paper_id"]
                    st.session_state["current_page"] = "Paper Viewer"
                    st.rerun()
    else:
        st.subheader("Library Papers")
        for paper in filtered:
            with st.container(border=True):
                st.markdown(f"**{paper.title}**")
                author_line = ", ".join(paper.authors[:4]) if paper.authors else "Unknown authors"
                st.caption(f"{author_line} | {paper.year or 'Unknown year'} | {paper.venue or 'Unknown venue'}")
                if paper.abstract:
                    st.write(paper.abstract[:280] + ("..." if len(paper.abstract) > 280 else ""))
                if st.button("View Paper", key=f"library-open-{paper.paper_id}"):
                    st.session_state["selected_paper_id"] = paper.paper_id
                    st.session_state["current_page"] = "Paper Viewer"
                    st.rerun()


def _group_search_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for item in results:
        paper_id = str(item.get("paper_id"))
        record = grouped.setdefault(
            paper_id,
            {
                "paper_id": paper_id,
                "paper_title": item.get("paper_title"),
                "score": float(item.get("score", 0.0)),
                "year": item.get("year"),
                "venue": item.get("venue"),
                "sections": [],
                "_seen_sections": set(),
            },
        )
        record["score"] = max(record["score"], float(item.get("score", 0.0)))
        section_name = str(item.get("section", "Unknown Section"))
        if section_name in record["_seen_sections"]:
            continue
        record["_seen_sections"].add(section_name)
        snippet = str(item.get("text", "")).strip()
        record["sections"].append(
            {
                "section": section_name,
                "text": snippet[:240] + ("..." if len(snippet) > 240 else ""),
            }
        )

    final = []
    for record in grouped.values():
        record.pop("_seen_sections", None)
        final.append(record)
    final.sort(key=lambda item: item["score"], reverse=True)
    return final
