"""
Summary tab for Stats Compass application.

Displays dataset summary statistics including:
- Row/column counts
- Missing value analysis
- Numeric column statistics
- Categorical column top values
"""

import streamlit as st
from utils.analysis import summarise_dataset


def render_summary_tab() -> None:
    """Render the summary tab with dataset statistics and visualizations."""
    st.header("Summary")
    summary, missing_by_col, numeric_desc, top_cats = summarise_dataset(st.session_state.df)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{summary['rows']:,}")
    with c2:
        st.metric("Columns", f"{summary['cols']:,}")
    with c3:
        st.metric("Missing (total)", f"{summary['missing_values_total']:,}")

    st.markdown("**Numeric summary (first 10 columns)**")
    st.dataframe(numeric_desc.head(10), use_container_width=True)

    if len(missing_by_col) > 0:
        st.markdown("**Most missing values by column**")
        st.bar_chart(missing_by_col, use_container_width=True)

    if top_cats:
        st.markdown("**Top categories (up to 5 columns)**")
        for col, vc in top_cats.items():
            st.write(f"• {col}")
            st.bar_chart(vc, use_container_width=True)
