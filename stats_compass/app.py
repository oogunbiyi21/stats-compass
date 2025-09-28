# stats_compass/app.py
import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# ---- Local modules ----
from utils.data_loading import process_uploaded_file
from utils.visualization import display_single_chart
from utils.analysis import summarise_dataset, key_trends_numeric_only, suggest_visualisations
from utils.token_tracking import (
    track_usage,
    update_session_usage, 
    get_usage_summary,
    check_usage_limits
)
from utils.export_utils import (
    render_session_summary,
    render_export_buttons,
    render_text_export_buttons,
    render_data_export_buttons,
    render_dataframe_export_buttons,
    render_model_export_buttons,
    render_report_preview,
    create_narrative_summary
)
from utils.agent_transcript import (
    AgentTranscriptLogger,
    AgentTranscriptDisplay,
    AgentTranscriptExporter,
    store_session_transcripts,
    render_transcript_history
)
from planner_mcp import run_mcp_planner
from smart_suggestions import generate_smart_suggestions

# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="Stats Compass", layout="wide")

# Add authentication check
try:
    from auth import check_password
    if not check_password():
        st.stop()
except ImportError:
    # If auth.py doesn't exist, continue without authentication
    pass

with st.sidebar:

    # Diagnostics section (always visible)
    st.markdown("**⚙️ Diagnostics**")
    
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

    # Check for usage warnings
    total_tokens, total_cost, usage_display = get_usage_summary()
    st.markdown(f"**{usage_display}**")
    usage_warning = check_usage_limits(total_tokens, total_cost)
    if usage_warning:
        st.warning(usage_warning)
    
    # Show detailed breakdown if there's usage
    if total_tokens > 0:
        st.markdown("**📊 Usage Details**")
        usage_history = st.session_state.get("usage_history", [])
        if usage_history:
            st.caption(f"Total interactions: {len(usage_history)}")
            
            # Show last few interactions
            recent = usage_history[-3:] if len(usage_history) > 3 else usage_history
            for i, usage in enumerate(recent, 1):
                st.caption(f"Query {len(usage_history) - len(recent) + i}: {usage['total_tokens']} tokens (${usage['cost']:.4f})")

    

st.title("🧭 Stats Compass")
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
        with st.sidebar:
            st.success(f"✅ Loaded {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns")
            mem_mb = st.session_state.df.memory_usage(deep=True).sum() / (1024**2)
            st.caption(f"Approx. memory usage: {mem_mb:.2f} MB")
            with st.expander("📊Dataset preview", expanded=False):
                st.dataframe(st.session_state.df.head(), use_container_width=True)
            st.rerun()
else:
    uploaded_file = None
    with st.sidebar:
        # Show usage stats
        if hasattr(st.session_state, 'query_count'):
            st.caption(f"Queries this session: {st.session_state.query_count}")
            
        # Quick Export Actions at the top (always visible)
        st.divider()
        st.markdown("**⚡ Quick Export**")
        filename = st.session_state.get('uploaded_filename', "dataset")
        
        # Compact export buttons for immediate access
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="📊 CSV",
                data=csv_data,
                file_name=f"{filename}_cleaned.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download current dataset as CSV"
            )
        with col2:
            # Create Excel data using BytesIO buffer
            import io
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                st.session_state.df.to_excel(writer, sheet_name='Data', index=False)
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📈 Excel", 
                data=excel_data,
                file_name=f"{filename}_cleaned.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Download current dataset as Excel"
            )

        # Export & Reports section (moved higher and made collapsible)
        with st.expander("📄 Export & Reports", expanded=False):
            # Show session summary
            render_session_summary(st.session_state.chat_history, location="sidebar")
            
            # Render export buttons
            render_export_buttons(st.session_state.chat_history, filename, location="sidebar")

        # Show what the AI knows about this dataset
        with st.expander("🧠 Dataset Context", expanded=False):
            from planner_mcp import generate_dataset_context
            context = generate_dataset_context(st.session_state.df)
            st.code(context.strip(), language=None)
            st.info("💡 I have immediate knowledge of all these columns and can suggest analysis without needing to explore first!")

        # File uploader in sidebar when dataset is loaded (also collapsible)
        with st.expander("📁 Change Dataset", expanded=False):
            sidebar_uploaded_file = st.file_uploader(
                "Upload a different dataset",
                type=["csv", "xlsx", "xls"],
                help="Replace current dataset with a new file",
                key="sidebar_uploader"
            )

            filename = st.session_state.get('uploaded_filename', 'Unknown file')
            st.success(f"📊 **Current Dataset**")
            st.markdown(f"**📁 {filename}**")
            st.caption(f"{st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns")
            mem_mb = st.session_state.df.memory_usage(deep=True).sum() / (1024**2)
            st.caption(f"Memory: {mem_mb:.2f} MB")
            
            # Process sidebar file upload using the shared function
            if process_uploaded_file(sidebar_uploaded_file, clear_history=True):
                st.success(f"✅ New dataset loaded: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns")
                # Force refresh to update dataset context
                st.rerun()
            
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Reports", "Explore", "Summary", "Agent Logs"])

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
            
            # Display intermediate steps for assistant messages
            if msg["role"] == "assistant" and "intermediate_steps" in msg:
                st.markdown("---")
                st.markdown("🔧 **Analysis Steps:**")
                
                for step_i, step in enumerate(msg["intermediate_steps"], 1):
                    try:
                        action, observation = step
                        tool_name = getattr(action, 'tool', 'unknown_tool')
                        tool_input = getattr(action, "tool_input", {})
                        
                        # Create a more readable display
                        with st.expander(f"Step {step_i}: {tool_name}", expanded=False):
                            if tool_input:
                                st.markdown("**Input:**")
                                st.json(tool_input)
                            st.markdown("**Result:**")
                            # Truncate very long observations
                            obs_str = str(observation)
                            if len(obs_str) > 1000:
                                st.text_area(
                                    "Tool output (truncated)", 
                                    value=obs_str[:1000] + "...", 
                                    height=120, 
                                    key=f"history_obs_{i}_{step_i}_truncated",
                                    label_visibility="hidden"
                                )
                                st.caption("(Output truncated for display)")
                            else:
                                st.text_area(
                                    "Tool output", 
                                    value=obs_str, 
                                    height=120, 
                                    key=f"history_obs_{i}_{step_i}_full",
                                    label_visibility="hidden"
                                )
                    except Exception as e:
                        with st.expander(f"Step {step_i}: (parsing error)", expanded=False):
                            st.text(f"Error: {e}")
                            st.text(str(step))
                
                st.markdown("---")
            
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

        # Assistant response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data and generating insights..."):
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

            # Display the actual response
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

        # Persist assistant reply with any charts and intermediate steps that were created
        assistant_message = {"role": "assistant", "content": final_text}
        if current_charts:
            assistant_message["charts"] = current_charts
        # Store intermediate steps for replay in chat history
        if isinstance(result, dict) and result.get("intermediate_steps"):
            assistant_message["intermediate_steps"] = result["intermediate_steps"]
        st.session_state.chat_history.append(assistant_message)
        
        # Track token usage and cost for this interaction
        try:
            usage_stats = track_usage(queued, final_text, model="gpt-4o")  # Adjust model as needed
            update_session_usage(usage_stats)
            
            
            # Force rerun to update sidebar usage display
            st.rerun()
        except Exception as e:
            st.caption(f"Usage tracking error: {e}")

    # 3) Place the chat input at the END. When user submits, queue it + rerun.
    user_query = st.chat_input("Ask a question about your data", key="chat_input_bottom")
    if user_query:
        st.session_state.to_process = user_query
        st.rerun()

with tab2:
    st.header("📄 Reports & Export")
    
    if not st.session_state.chat_history:
        st.info("💬 Start a conversation in the Chat tab to generate reports.")
    else:
        # Session statistics
        render_session_summary(st.session_state.chat_history, location="tab")
        
        st.divider()

        # Export Options
        st.subheader("⬇️ Export Options")
        
        # Create tabs for different export types
        export_tab1, export_tab2, export_tab3, export_tab4 = st.tabs([
            "📝 Reports", "📊 Charts", "💾 Data", "🤖 Models"
        ])
        
        with export_tab1:
            st.markdown("**Text Reports & Analysis**")
            render_text_export_buttons(st.session_state.chat_history, filename, location="tab")
        
        with export_tab2:
            st.markdown("**Charts & Session Data**")  
            render_data_export_buttons(st.session_state.chat_history, filename, location="tab")
        
        with export_tab3:
            st.markdown("**Raw & Processed Data**")
            render_dataframe_export_buttons(filename, location="tab")
        
        with export_tab4:
            st.markdown("**Trained Models & Analysis Objects**")
            render_model_export_buttons(filename, location="tab")
        

        # Report Generation Section
        st.divider()
        st.subheader("📊 Generate Reports")
        
        filename = uploaded_file.name if uploaded_file else "dataset"
        
        # Narrative Summary
        with st.expander("📖 View Narrative Summary", expanded=True):
            try:
                narrative = create_narrative_summary(st.session_state.chat_history)
                st.markdown(narrative)
            except Exception as e:
                st.error(f"Error generating narrative: {e}")

        # Preview Section
        st.divider()
        render_report_preview(st.session_state.chat_history, filename)

with tab3:
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

with tab4:
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


with tab5:
    st.header("🤖 AI Agent Reasoning & Logs")
    
    st.markdown("""
    This section shows how the AI agent thinks, what tools it uses, and the step-by-step reasoning 
    behind each analysis. This is valuable for understanding the agent's decision-making process 
    and for learning about data analysis workflows.
    """)
    
    # Check if there are any transcripts
    if "agent_transcripts" not in st.session_state or not st.session_state.agent_transcripts:
        st.info("🤖 No agent activity yet. Start a conversation in the Chat tab to see agent reasoning!")
        st.markdown("""
        **What you'll see here:**
        - 🧠 **Agent Reasoning**: The thought process behind each analysis
        - 🔧 **Tool Usage**: Which data analysis tools were used and why
        - 📊 **Step-by-Step Process**: Detailed breakdown of each analysis step
        - 📥 **Export Options**: Download transcripts for documentation or learning
        """)
    else:
        # Summary statistics
        total_transcripts = len(st.session_state.agent_transcripts)
        latest_transcript = st.session_state.agent_transcripts[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📜 Total Sessions", total_transcripts)
        with col2:
            st.metric("🔧 Tools in Latest", len(latest_transcript.get("tools_used", [])))
        with col3:
            st.metric("⚡ Steps in Latest", latest_transcript.get("total_steps", 0))
        
        st.divider()
        
        # Display option tabs
        transcript_tab1, transcript_tab2 = st.tabs(["📖 Latest Session", "📚 All Sessions"])
        
        with transcript_tab1:
            if st.session_state.agent_transcripts:
                latest = st.session_state.agent_transcripts[-1]
                st.subheader("🤖 Most Recent Agent Session")
                AgentTranscriptDisplay.render_transcript_expander(latest, "Latest Agent Analysis")
                
                # Export options for latest
                st.subheader("📥 Export Latest Session")
                col1, col2 = st.columns(2)
                with col1:
                    AgentTranscriptExporter.create_downloadable_transcript(latest, "json", "latest")
                with col2:
                    AgentTranscriptExporter.create_downloadable_transcript(latest, "markdown", "latest")
        
        with transcript_tab2:
            st.subheader("📜 Complete Session History")
            render_transcript_history()
