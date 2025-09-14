# ds_auto_insights/app.py
import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# ---- Local modules ----
from util import (
    summarise_dataset,
    key_trends_numeric_only,
    suggest_visualisations,
    process_uploaded_file,
    display_single_chart
)
from planner_mcp import run_mcp_planner
from export_utils import (
    generate_markdown_report,
    export_pdf_report,
    create_narrative_summary,
    export_session_data,
    export_chart_as_image
)
from smart_suggestions import generate_smart_suggestions

# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="DS Auto Insights", layout="wide")

# Add authentication check
try:
    from auth import check_password
    if not check_password():
        st.stop()
except ImportError:
    # If auth.py doesn't exist, continue without authentication
    pass

with st.sidebar:
    st.header("⚙️ Diagnostics")
    
    # Environment detection
    is_cloud = hasattr(st, "secrets") and "localhost" not in st.context.headers.get("host", "")
    env_type = "☁️ Streamlit Cloud" if is_cloud else "💻 Local Dev"
    st.caption(f"Environment: {env_type}")
    
    # API Key status
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    st.write("OPENAI_API_KEY set:", api_key_set)
    if not api_key_set:
        if is_cloud:
            st.warning("⚠️ Add OPENAI_API_KEY to Streamlit Cloud secrets")
        else:
            st.caption("Tip: create a `.env` with OPENAI_API_KEY=sk-...")
    
    st.divider()
    
    
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
# Show in main area when no dataset loaded, move to sidebar when dataset loaded
if "df" not in st.session_state or st.session_state.df is None:
    # Main area file uploader when no dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    
    # Process the uploaded file
    if process_uploaded_file(uploaded_file):
        with st.expander("📊Dataset preview", expanded=False):
            st.dataframe(st.session_state.df.head(), use_container_width=True)
        with st.sidebar:
            st.success(f"✅ Loaded {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns")
            mem_mb = st.session_state.df.memory_usage(deep=True).sum() / (1024**2)
            st.caption(f"Approx. memory usage: {mem_mb:.2f} MB")
            st.divider()
            st.rerun()
else:
    uploaded_file = None
    with st.sidebar:
        # Show usage stats
        if hasattr(st.session_state, 'query_count'):
            st.caption(f"Queries this session: {st.session_state.query_count}")
            
        # Show what the AI knows about this dataset
        with st.expander("🧠 What I know about your dataset", expanded=False):
            from planner_mcp import generate_dataset_context
            context = generate_dataset_context(st.session_state.df)
            st.code(context.strip(), language=None)
            st.info("💡 I have immediate knowledge of all these columns and can suggest analysis without needing to explore first!")

        # Detailed smart suggestions (different from the compact sidebar ones)
        with st.expander("� Columns & Data Types"):
            info = pd.DataFrame({
                "column": st.session_state.df.columns,
                "dtype": st.session_state.df.dtypes.astype(str).values,
                "nulls": st.session_state.df.isna().sum().values
            })
            st.dataframe(info, use_container_width=True)
            info = pd.DataFrame({
                "column": st.session_state.df.columns,
                "dtype": st.session_state.df.dtypes.astype(str).values,
                "nulls": st.session_state.df.isna().sum().values
            })
            st.dataframe(info, use_container_width=True)

        # File uploader in sidebar when dataset is loaded
        st.divider()
        st.header("📁 Change Dataset")
        sidebar_uploaded_file = st.file_uploader(
            "Upload a different dataset",
            type=["csv", "xlsx", "xls"],
            help="Replace current dataset with a new file",
            key="sidebar_uploader"
        )
        
        # Process sidebar file upload using the shared function
        if process_uploaded_file(sidebar_uploaded_file, clear_history=True):
            st.success(f"✅ New dataset loaded: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns")
            # Force refresh to update dataset context
            st.rerun()

        # Export & Reports section (only show when dataset is loaded)
        if hasattr(st.session_state, 'df') and st.session_state.df is not None:
            st.divider()
            st.header("📄 Export & Reports")
            
            # Check if there's any analysis to export
            has_analysis = hasattr(st.session_state, 'chat_history') and st.session_state.chat_history
            
            if has_analysis:
                # Summary stats about the session
                user_questions = [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
                assistant_responses = [msg for msg in st.session_state.chat_history if msg["role"] == "assistant"]
                total_charts = sum(len(msg.get("charts", [])) for msg in assistant_responses)
                
                st.caption(f"💬 {len(user_questions)} questions • 📊 {total_charts} charts")
            else:
                st.caption("💡 Start analyzing to generate reports")
            
            filename = st.session_state.get('uploaded_filename', "dataset")
            
            # Export buttons in sidebar
            if st.button("📝 Markdown", help="Export conversation as Markdown", use_container_width=True):
                if not has_analysis:
                    st.warning("⚠️ Can't generate report without analysis. Ask questions about your data first!")
                else:
                    try:
                        markdown_content = generate_markdown_report(st.session_state.chat_history, filename)
                        st.download_button(
                            label="⬇️ Download Markdown",
                            data=markdown_content,
                            file_name=f"analysis_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown",
                            key="sidebar_md_download",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            if st.button("📊 PDF Report", help="Export full report with charts as PDF", use_container_width=True):
                if not has_analysis:
                    st.warning("⚠️ Can't generate report without analysis. Ask questions about your data first!")
                else:
                    try:
                        with st.spinner("Generating PDF..."):
                            pdf_bytes = export_pdf_report(st.session_state.chat_history, filename)
                            if pdf_bytes:
                                st.download_button(
                                    label="⬇️ Download PDF",
                                    data=pdf_bytes,
                                    file_name=f"analysis_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                    mime="application/pdf",
                                    key="sidebar_pdf_download",
                                    use_container_width=True
                                )
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            if st.button("📈 Charts ZIP", help="Download all charts as images", use_container_width=True):
                if not has_analysis:
                    st.warning("⚠️ Can't export charts without analysis. Ask questions to generate charts first!")
                else:
                    # Check if any charts actually exist
                    total_charts = sum(len(msg.get("charts", [])) for msg in st.session_state.chat_history if msg["role"] == "assistant")
                    
                    if total_charts == 0:
                        st.warning("⚠️ No charts found to export! Ask questions that generate visualizations first.")
                    else:
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
                                            except Exception:
                                                pass
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                label="⬇️ Download Charts",
                                data=zip_buffer.getvalue(),
                                file_name=f"charts_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                                mime="application/zip",
                                key="sidebar_charts_download",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            # Dataset context and detailed analysis
            st.divider()
            
            

if uploaded_file is not None:
    try:
        # File processing is handled above, this section is no longer needed
        pass
            
    except Exception as e:
        # Handle other unexpected errors
        st.error(f"❌ **Unexpected error loading file:** {type(e).__name__}")
        st.info("💡 Please check your file format and try again. Supported formats: CSV, XLSX, XLS")
        with st.expander("🔍 Technical details (for debugging)"):
            st.code(str(e))
        
        # Reset session state
        st.session_state.df = None
        uploaded_file = None

# Guard
if not hasattr(st.session_state, 'df') or st.session_state.df is None:
    st.info("📂 Upload a CSV/XLSX file to get started.")
    st.stop()

df_use = st.session_state.df

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Summary", "Explore", "Reports"])

with tab1:
    st.header("Chat")
    
    # Smart Suggestions Grid (2x3 format)
    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        suggestions = generate_smart_suggestions(st.session_state.df)
        
        if suggestions:
            st.markdown("💡 **Quick Analysis Suggestions**")
            
            # Create 2x3 grid of suggestion buttons
            col1, col2, col3 = st.columns(3)
            
            # First row
            if len(suggestions) > 0:
                with col1:
                    if st.button(
                        suggestions[0]['title'], 
                        key="grid_suggest_0",
                        help=suggestions[0]['description'],
                        use_container_width=True
                    ):
                        st.session_state.to_process = suggestions[0]['query']
                        st.rerun()
            
            if len(suggestions) > 1:
                with col2:
                    if st.button(
                        suggestions[1]['title'], 
                        key="grid_suggest_1",
                        help=suggestions[1]['description'],
                        use_container_width=True
                    ):
                        st.session_state.to_process = suggestions[1]['query']
                        st.rerun()
            
            if len(suggestions) > 2:
                with col3:
                    if st.button(
                        suggestions[2]['title'], 
                        key="grid_suggest_2",
                        help=suggestions[2]['description'],
                        use_container_width=True
                    ):
                        st.session_state.to_process = suggestions[2]['query']
                        st.rerun()
            
            # Second row
            if len(suggestions) > 3:
                with col1:
                    if st.button(
                        suggestions[3]['title'], 
                        key="grid_suggest_3",
                        help=suggestions[3]['description'],
                        use_container_width=True
                    ):
                        st.session_state.to_process = suggestions[3]['query']
                        st.rerun()
            
            if len(suggestions) > 4:
                with col2:
                    if st.button(
                        suggestions[4]['title'], 
                        key="grid_suggest_4",
                        help=suggestions[4]['description'],
                        use_container_width=True
                    ):
                        st.session_state.to_process = suggestions[4]['query']
                        st.rerun()
            
            if len(suggestions) > 5:
                with col3:
                    if st.button(
                        suggestions[5]['title'], 
                        key="grid_suggest_5",
                        help=suggestions[5]['description'],
                        use_container_width=True
                    ):
                        st.session_state.to_process = suggestions[5]['query']
                        st.rerun()
            
            st.divider()

    # 1) Replay existing chat history first (before processing new message)
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display charts that were created with this message
            if msg["role"] == "assistant" and "charts" in msg:
                for j, chart_info in enumerate(msg["charts"]):
                    display_single_chart(chart_info, f"history_{i}_{j}")

    # 2) If we have a message queued from the previous run, process it
    queued = st.session_state.pop("to_process", None)
    if queued is not None and hasattr(st.session_state, 'df') and st.session_state.df is not None:
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
                st.info(f"📊 Displaying {len(st.session_state.current_response_charts)} charts from this response")
                for i, chart_info in enumerate(st.session_state.current_response_charts):
                    st.caption(f"Chart {i+1}: {chart_info.get('type', 'unknown')} - {chart_info.get('title', 'untitled')}")
                    display_single_chart(chart_info, f"current_{i}")
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
