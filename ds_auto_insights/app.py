# ds_auto_insights/app.py
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---- Local modules ----
from util import (
    load_table,
    summarise_dataset,
    key_trends_numeric_only,
    suggest_visualisations,
)
from planner_mcp import run_mcp_planner

# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="DS Auto Insights", layout="wide")

with st.sidebar:
    st.header("⚙️ Diagnostics")
    st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
    st.caption("Tip: create a `.env` with OPENAI_API_KEY=sk-...")

st.title("📊 DS Auto Insights")
st.subheader("Turn your raw datasets into structured insights instantly.")

# ---------- Session State ----------
if "df" not in st.session_state:
    st.session_state.df = None

if "chat_history" not in st.session_state:
    # store as simple dict messages for Streamlit chat
    st.session_state.chat_history = []  # [{"role": "user"/"assistant", "content": "..."}]

if "chart_data" not in st.session_state:
    st.session_state.chart_data = {}

# ---------- File Uploader ----------
uploaded_file = st.file_uploader("Upload your dataset (CSV/XLSX)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = load_table(uploaded_file)
        st.session_state.df = df
        st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        st.dataframe(df.head(), use_container_width=True)
        mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
        st.caption(f"Approx. memory usage: {mem_mb:.2f} MB")

        with st.expander("Columns & dtypes"):
            info = pd.DataFrame({
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "nulls": df.isna().sum().values
            })
            st.dataframe(info, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")

# Guard
if st.session_state.df is None:
    st.info("📂 Upload a CSV/XLSX file to get started.")
    st.stop()

df_use = st.session_state.df

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Chat (LLM)", "Summary", "Explore"])

with tab1:
    st.header("Chat (tool-calling)")

    # 1) Replay existing chat history first (before processing new message)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 2) If we have a message queued from the previous run, process it
    queued = st.session_state.pop("to_process", None)
    if queued is not None and st.session_state.df is not None:
        # Show the user's message
        with st.chat_message("user"):
            st.markdown(queued)
        st.session_state.chat_history.append({"role": "user", "content": queued})

        # Assistant "thinking…" placeholder
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ Thinking...")

            # Call your agent WITH chat history for context
            try:
                result = run_mcp_planner(
                    queued, 
                    df_use, 
                    chat_history=st.session_state.chat_history[:-1]  # Exclude the current user message
                )
                final_text = result.get("output", "(No output)")
            except Exception as e:
                final_text = f"❌ Agent error: {e}"
                result = {}

            # Replace "thinking" with the actual answer
            placeholder.empty()
            st.markdown(final_text)

            # Display charts if any were created
            if hasattr(st.session_state, 'chart_data') and st.session_state.chart_data:
                for chart_name, chart_info in st.session_state.chart_data.items():
                    chart_type = chart_info.get('type')
                    data = chart_info.get('data')
                    title = chart_info.get('title', 'Chart')
                    
                    if chart_type == 'histogram':
                        st.subheader(f"📊 {title}")
                        st.bar_chart(data.set_index('bin_range')['count'], use_container_width=True)
                        
                    elif chart_type == 'bar':
                        st.subheader(f"📊 {title}")
                        st.bar_chart(data.set_index('category')['count'], use_container_width=True)
                        
                    elif chart_type == 'scatter':
                        st.subheader(f"📊 {title}")
                        if chart_info.get('color_column'):
                            st.scatter_chart(
                                data, 
                                x=chart_info['x_column'], 
                                y=chart_info['y_column'],
                                color=chart_info['color_column'],
                                use_container_width=True
                            )
                        else:
                            st.scatter_chart(
                                data, 
                                x=chart_info['x_column'], 
                                y=chart_info['y_column'],
                                use_container_width=True
                            )
                        # Show correlation info
                        corr = chart_info.get('correlation', 0)
                        if abs(corr) > 0.7:
                            st.success(f"🔍 Strong correlation: {corr:.3f}")
                        elif abs(corr) > 0.3:
                            st.info(f"📈 Moderate correlation: {corr:.3f}")
                        else:
                            st.caption(f"📊 Weak correlation: {corr:.3f}")
                            
                    elif chart_type == 'line':
                        st.subheader(f"📊 {title}")
                        st.line_chart(
                            data.set_index(chart_info['x_column'])[chart_info['y_column']], 
                            use_container_width=True
                        )
                
                # Clear chart data after displaying
                st.session_state.chart_data = {}

            # Optional: show intermediate steps
            if isinstance(result, dict) and result.get("intermediate_steps"):
                with st.expander("🔎 Intermediate steps"):
                    for i, step in enumerate(result["intermediate_steps"], 1):
                        try:
                            action, observation = step
                            st.markdown(f"**Step {i}: {getattr(action, 'tool', 'tool')}**")
                            st.code(getattr(action, "tool_input", {}), language="json")
                            st.text_area("Observation", value=str(observation), height=120, key=f"obs_{i}")
                        except Exception:
                            st.text(str(step))

        # Persist assistant reply
        st.session_state.chat_history.append({"role": "assistant", "content": final_text})

    # 3) Place the chat input at the END. When user submits, queue it + rerun.
    user_query = st.chat_input("Ask a question about your data", key="chat_input_bottom")
    if user_query:
        st.session_state.to_process = user_query
        st.rerun()


with tab2:
    st.header("Summary")
    summary, missing_by_col, numeric_desc, top_cats = summarise_dataset(df_use)

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

with tab3:
    st.header("Explore")
    st.markdown("**Numeric correlation matrix (table view)**")
    corr = key_trends_numeric_only(df_use)
    if corr is None:
        st.info("Not enough numeric columns to compute correlations.")
    else:
        st.dataframe(corr, use_container_width=True)
        st.caption("We can add a heatmap later if useful.")

    st.markdown("---")
    st.markdown("**Suggested visualisations**")
    suggestions = suggest_visualisations(df_use)
    if not suggestions:
        st.info("No clear suggestions from this schema. Try uploading a different dataset.")
    else:
        for title, render in suggestions:
            st.markdown(f"**{title}**")
            try:
                render()  # calls the provided lambda
            except Exception as e:
                st.error(f"Failed to render: {e}")
