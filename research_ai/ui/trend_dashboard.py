from __future__ import annotations

import pandas as pd
import streamlit as st

from research_ai.analytics.mcp_tools import trend_analytics_tool


def render(papers, analytics_snapshot):
    st.title("Trend Analytics")
    if analytics_snapshot.get("error"):
        st.warning(f"Trend analytics unavailable: {analytics_snapshot['error']}")
    if not papers:
        st.info("Load parsed papers to inspect publication trends and emerging topics.")
        return

    topic_trends = analytics_snapshot.get("topic_trends", {})
    emerging_topics = analytics_snapshot.get("emerging_topics", [])
    venue_counts = analytics_snapshot.get("venue_counts", {})

    st.subheader("Emerging Topics")
    if emerging_topics:
        st.dataframe(pd.DataFrame(emerging_topics), use_container_width=True)
    else:
        st.write("No topic trends available yet.")

    st.subheader("Publication Frequency by Venue")
    if venue_counts:
        venue_df = pd.DataFrame([{"venue": key, "papers": value} for key, value in venue_counts.items()])
        st.bar_chart(venue_df.set_index("venue"))

    st.subheader("Topic Growth Over Time")
    available_topics = sorted(topic_trends.keys())
    if available_topics:
        selected_topic = st.selectbox("Topic", options=available_topics, index=0)
        topic_df = pd.DataFrame(
            [{"year": year, "papers": count} for year, count in topic_trends[selected_topic].items()]
        )
        if not topic_df.empty:
            st.line_chart(topic_df.set_index("year"))
        insight = trend_analytics_tool(selected_topic, papers, extracted_keywords=analytics_snapshot.get("keyword_map"))
        st.markdown("**Topic Snapshot**")
        st.write(f"Growth rate: {insight['trend_growth']}")
        st.write(f"Example papers: {', '.join(insight['example_papers']) if insight['example_papers'] else 'None'}")
