from __future__ import annotations

import streamlit as st

from research_ai.rag.comparison_engine import compare_papers


def render(papers, paper_lookup):
    st.title("Cross-Paper Comparison")
    if len(papers) < 2:
        st.info("Add at least two parsed papers to compare research methods and findings.")
        return

    selected_ids = st.multiselect(
        "Select papers to compare",
        options=[paper.paper_id for paper in papers],
        format_func=lambda value: paper_lookup[value].title,
        default=[paper.paper_id for paper in papers[:2]],
    )
    question = st.text_input("Comparison question", value="Compare the methods used in these papers.")

    if st.button("Compare Papers", key="run-compare"):
        if len(selected_ids) < 2:
            st.warning("Select at least two papers.")
        else:
            try:
                result = compare_papers(question, paper_ids=selected_ids)
                st.session_state["last_comparison_result"] = result
            except Exception as exc:
                st.session_state["last_comparison_error"] = str(exc)

    result = st.session_state.get("last_comparison_result")
    error = st.session_state.get("last_comparison_error")
    if error:
        st.error(f"Comparison failed: {error}")
    elif result:
        st.subheader("Comparison")
        st.markdown(result["comparison"])
        st.markdown("**Sources**")
        for source in result.get("sources", []):
            st.write(f"- {source['paper_title']} | {source['section']}")
