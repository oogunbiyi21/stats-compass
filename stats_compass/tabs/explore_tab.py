"""
Explore tab for Stats Compass application.

Provides exploratory data analysis tools including:
- Correlation matrix for numeric columns
- Suggested visualizations based on data types
"""

import streamlit as st
from utils.analysis import key_trends_numeric_only, suggest_visualisations


def render_explore_tab() -> None:
    """Render the explore tab with correlation matrix and suggested visualizations."""
    st.header("Explore")
    st.markdown("**Numeric correlation matrix (table view)**")
    corr = key_trends_numeric_only(st.session_state.df)
    if corr is None:
        st.info("Not enough numeric columns to compute correlations.")
    else:
        st.dataframe(corr, use_container_width=True)

    st.markdown("---")
    st.markdown("**Suggested visualisations**")
    suggestions = suggest_visualisations(st.session_state.df)
    if not suggestions:
        st.info("No clear suggestions from this schema. Try uploading a different dataset.")
    else:
        for title, render in suggestions:
            st.markdown(f"**{title}**")
            try:
                render()  # calls the provided lambda
            except Exception as e:
                st.error(f"Failed to render: {e}")
