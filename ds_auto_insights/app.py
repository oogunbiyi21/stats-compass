# ds_auto_insights/app.py
import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# ---- Local modules ----
from util import (
    load_table,
    summarise_dataset,
    key_trends_numeric_only,
    suggest_visualisations,
)
from planner_mcp import run_mcp_planner
from export_utils import (
    generate_markdown_report,
    export_pdf_report,
    create_narrative_summary,
    export_session_data,
    export_chart_as_image
)


def display_single_chart(chart_info):
    """Display a single chart based on chart_info dictionary"""
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
    st.session_state.chart_data = []
if "current_response_charts" not in st.session_state:
    st.session_state.current_response_charts = []

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
tab1, tab2, tab3, tab4 = st.tabs(["Chat (LLM)", "Summary", "Explore", "📄 Reports"])

with tab1:
    st.header("Chat (tool-calling)")

    # 1) Replay existing chat history first (before processing new message)
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display charts that were created with this message
            if msg["role"] == "assistant" and "charts" in msg:
                for chart_info in msg["charts"]:
                    display_single_chart(chart_info)

    # 2) If we have a message queued from the previous run, process it
    queued = st.session_state.pop("to_process", None)
    if queued is not None and st.session_state.df is not None:
        # Clear current response charts to start fresh
        st.session_state.current_response_charts = []
        
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

            # Display any charts that were created during this response
            current_charts = []
            if hasattr(st.session_state, 'current_response_charts') and st.session_state.current_response_charts:
                for chart_info in st.session_state.current_response_charts:
                    display_single_chart(chart_info)
                    current_charts.append(chart_info)
                
                # Clear the current response charts since they're now displayed
                st.session_state.current_response_charts = []

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

        # Persist assistant reply with any charts that were created
        assistant_message = {"role": "assistant", "content": final_text}
        if current_charts:
            assistant_message["charts"] = current_charts
        st.session_state.chat_history.append(assistant_message)

    # Export Section
    if st.session_state.chat_history:
        st.divider()
        st.subheader("📄 Export & Reports")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📝 Export Markdown", help="Download conversation as Markdown"):
                try:
                    filename = uploaded_file.name if uploaded_file else "dataset"
                    markdown_content = generate_markdown_report(
                        st.session_state.chat_history, 
                        filename
                    )
                    st.download_button(
                        label="⬇️ Download Markdown",
                        data=markdown_content,
                        file_name=f"analysis_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"Error generating markdown: {e}")
        
        with col2:
            if st.button("📊 Export PDF", help="Download full report with charts as PDF"):
                try:
                    filename = uploaded_file.name if uploaded_file else "dataset"
                    with st.spinner("Generating PDF report..."):
                        pdf_bytes = export_pdf_report(
                            st.session_state.chat_history,
                            filename
                        )
                        if pdf_bytes:
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=pdf_bytes,
                                file_name=f"analysis_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf"
                            )
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
        
        with col3:
            if st.button("📈 Export Charts", help="Download all charts as images"):
                try:
                    import zipfile
                    from io import BytesIO
                    
                    # Create a zip file with all charts
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        chart_count = 1
                        for msg in st.session_state.chat_history:
                            if msg["role"] == "assistant" and "charts" in msg:
                                for chart in msg["charts"]:
                                    try:
                                        chart_bytes = export_chart_as_image(chart, format='png')
                                        if chart_bytes:
                                            chart_title = chart.get('title', f'Chart_{chart_count}')
                                            safe_title = "".join(c for c in chart_title if c.isalnum() or c in (' ', '-', '_')).strip()
                                            zip_file.writestr(f"{chart_count:02d}_{safe_title}.png", chart_bytes)
                                            chart_count += 1
                                    except Exception as e:
                                        st.warning(f"Could not export chart: {e}")
                    
                    zip_buffer.seek(0)
                    filename = uploaded_file.name if uploaded_file else "dataset"
                    st.download_button(
                        label="⬇️ Download Charts ZIP",
                        data=zip_buffer.getvalue(),
                        file_name=f"charts_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                        mime="application/zip"
                    )
                except Exception as e:
                    st.error(f"Error exporting charts: {e}")
        
        with col4:
            if st.button("📋 Export Data", help="Download session data as JSON"):
                try:
                    # Extract all charts from chat history
                    all_charts = []
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "assistant" and "charts" in msg:
                            all_charts.extend(msg["charts"])
                    
                    session_data = export_session_data(st.session_state.chat_history, all_charts)
                    filename = uploaded_file.name if uploaded_file else "dataset"
                    
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json.dumps(session_data, indent=2),
                        file_name=f"session_data_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error exporting session data: {e}")

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


with tab4:
    st.header("📄 Reports & Export")
    
    if not st.session_state.chat_history:
        st.info("💬 Start a conversation in the Chat tab to generate reports.")
    else:
        # Summary stats about the session
        user_questions = [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
        assistant_responses = [msg for msg in st.session_state.chat_history if msg["role"] == "assistant"]
        total_charts = sum(len(msg.get("charts", [])) for msg in assistant_responses)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Questions Asked", len(user_questions))
        with col2:
            st.metric("AI Responses", len(assistant_responses))
        with col3:
            st.metric("Charts Created", total_charts)
        
        st.divider()
        
        # Report Generation Section
        st.subheader("📊 Generate Reports")
        
        filename = uploaded_file.name if uploaded_file else "dataset"
        
        # Narrative Summary
        with st.expander("📖 View Narrative Summary", expanded=True):
            try:
                narrative = create_narrative_summary(st.session_state.chat_history)
                st.markdown(narrative)
            except Exception as e:
                st.error(f"Error generating narrative: {e}")
        
        # Export Options
        st.subheader("⬇️ Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 Text Reports**")
            
            # Markdown Export
            if st.button("Export Markdown Report", key="md_export_tab"):
                try:
                    markdown_content = generate_markdown_report(st.session_state.chat_history, filename)
                    st.download_button(
                        label="⬇️ Download Markdown",
                        data=markdown_content,
                        file_name=f"analysis_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown",
                        key="md_download"
                    )
                    st.success("✅ Markdown report ready for download!")
                except Exception as e:
                    st.error(f"Error generating markdown: {e}")
            
            # PDF Export
            if st.button("Export PDF Report", key="pdf_export_tab"):
                try:
                    with st.spinner("🔄 Generating PDF with embedded charts..."):
                        pdf_bytes = export_pdf_report(st.session_state.chat_history, filename)
                        if pdf_bytes:
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=pdf_bytes,
                                file_name=f"analysis_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                key="pdf_download"
                            )
                            st.success("✅ PDF report ready for download!")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
        
        with col2:
            st.markdown("**📊 Data & Charts**")
            
            # Charts Export
            if st.button("Export All Charts", key="charts_export_tab"):
                try:
                    import zipfile
                    from io import BytesIO
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        chart_count = 1
                        for msg in st.session_state.chat_history:
                            if msg["role"] == "assistant" and "charts" in msg:
                                for chart in msg["charts"]:
                                    try:
                                        chart_bytes = export_chart_as_image(chart, format='png')
                                        if chart_bytes:
                                            chart_title = chart.get('title', f'Chart_{chart_count}')
                                            safe_title = "".join(c for c in chart_title if c.isalnum() or c in (' ', '-', '_')).strip()
                                            zip_file.writestr(f"{chart_count:02d}_{safe_title}.png", chart_bytes)
                                            chart_count += 1
                                    except Exception as e:
                                        st.warning(f"Could not export chart: {e}")
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="⬇️ Download Charts ZIP",
                        data=zip_buffer.getvalue(),
                        file_name=f"charts_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                        mime="application/zip",
                        key="charts_download"
                    )
                    st.success("✅ Charts ZIP ready for download!")
                except Exception as e:
                    st.error(f"Error exporting charts: {e}")
            
            # Session Data Export
            if st.button("Export Session Data", key="data_export_tab"):
                try:
                    all_charts = []
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "assistant" and "charts" in msg:
                            all_charts.extend(msg["charts"])
                    
                    session_data = export_session_data(st.session_state.chat_history, all_charts)
                    
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json.dumps(session_data, indent=2),
                        file_name=f"session_data_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        key="data_download"
                    )
                    st.success("✅ Session data ready for download!")
                except Exception as e:
                    st.error(f"Error exporting session data: {e}")
        
        # Preview Section
        st.divider()
        st.subheader("👀 Report Preview")
        
        preview_type = st.radio(
            "Choose preview type:",
            ["Markdown", "Session Summary"],
            horizontal=True
        )
        
        if preview_type == "Markdown":
            try:
                markdown_content = generate_markdown_report(st.session_state.chat_history, filename)
                st.markdown("**Preview (first 1000 characters):**")
                st.code(markdown_content[:1000] + "..." if len(markdown_content) > 1000 else markdown_content, language="markdown")
            except Exception as e:
                st.error(f"Error generating preview: {e}")
        
        elif preview_type == "Session Summary":
            st.markdown("**Chat Session Overview:**")
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    st.markdown(f"**Q{i//2 + 1}:** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                elif msg["role"] == "assistant":
                    charts_info = f" (+{len(msg.get('charts', []))} charts)" if msg.get('charts') else ""
                    st.markdown(f"**A{i//2 + 1}:** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}{charts_info}")
                    
            if total_charts > 0:
                st.info(f"💡 This session created {total_charts} visualizations that will be included in exported reports.")
