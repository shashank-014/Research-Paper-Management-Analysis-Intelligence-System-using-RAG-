from __future__ import annotations

import streamlit as st

from research_ai.rag.rag_pipeline import answer_question


def render(papers, paper_lookup):
    st.title("Research Chat Assistant")
    if not papers:
        st.info("Load parsed papers to start chatting over the research library.")
        return

    default_scope = st.session_state.get("chat_paper_id")
    scope_options = ["Full Library"] + [paper.paper_id for paper in papers]
    selected_scope = st.selectbox(
        "Context scope",
        options=scope_options,
        index=scope_options.index(default_scope) if default_scope in scope_options else 0,
        format_func=lambda value: "Full Library" if value == "Full Library" else paper_lookup[value].title,
    )

    query = st.text_area(
        "Ask a question",
        placeholder="What datasets are used?",
        height=120,
    )

    if st.button("Run Research QA", key="run-chat"):
        if not query.strip():
            st.warning("Enter a research question first.")
        else:
            kwargs = {}
            if selected_scope != "Full Library":
                kwargs["paper_ids"] = [selected_scope]
            try:
                result = answer_question(query, **kwargs)
                st.session_state["last_chat_result"] = result
            except Exception as exc:
                st.session_state["last_chat_error"] = str(exc)

    result = st.session_state.get("last_chat_result")
    error = st.session_state.get("last_chat_error")
    if error:
        st.error(f"Research QA failed: {error}")
    elif result:
        st.subheader("Answer")
        st.write(result["answer"])
        st.markdown("**Sources**")
        for source in result.get("sources", []):
            st.write(f"- {source['paper_title']} | {source['section']} | score={source['score']}")
