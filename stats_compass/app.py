"""
Stats Compass - AI-Powered Data Analysis Application

This is the main entry point for the Stats Compass application, which provides
an AI-powered chat interface for exploring and analyzing datasets.

Architecture:
- config.py: Application constants and configuration
- components/: Reusable UI components (sidebar, etc.)
- tabs/: Individual tab modules (chat, reports, summary, explore, logs)
- utils/: Helper functions for data loading, visualization, analysis, export
- tools/: LangChain tools for ML, charts, and statistical analysis
- planner_mcp.py: Agent orchestration with Model Context Protocol

The app follows a modular design where:
1. Helper functions (render_chat_message, process_user_query, etc.) handle core logic
2. Component modules (sidebar) manage reusable UI sections
3. Tab modules encapsulate each tab's functionality
4. app.py orchestrates everything with minimal code (~280 lines)

Usage:
    streamlit run stats_compass/app.py
"""

# stats_compass/app.py
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---- Config ----
from config import (
    PAGE_TITLE,
    PAGE_LAYOUT,
    DEFAULT_MODEL,
    MAX_OBSERVATION_LENGTH,
)

# ---- Local modules ----
from utils.session import initialize_session_state
from utils.visualization import display_single_chart
from utils.token_tracking import track_usage, update_session_usage
from utils.agent_transcript import (
    AgentTranscriptLogger,
    store_session_transcripts,
)
from planner_mcp import run_mcp_planner_stream
from auth import check_password

# ---------- Components ----------
from components.sidebar import render_sidebar
from components.file_uploader import render_file_uploader

# ---------- Tabs ----------
from tabs.chat_tab import render_chat_tab
from tabs.reports_tab import render_reports_tab
from tabs.summary_tab import render_summary_tab
from tabs.explore_tab import render_explore_tab
from tabs.logs_tab import render_logs_tab

# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)

# ---------- Helper Functions ----------
def render_intermediate_steps(steps: list, message_index: int) -> None:
    """
    Render intermediate steps from an agent execution.
    
    Args:
        steps: List of (action, observation) tuples from agent execution
        message_index: Index of the message in chat history (for unique keys)
    """
    st.markdown("---")
    st.markdown("🔧 **Analysis Steps:**")
    
    for step_i, step in enumerate(steps, 1):
        try:
            # Step should be a tuple: (AgentAction, observation_string)
            action, observation = step
            
            # Try different ways to get tool name (robust extraction)
            tool_name = None
            if hasattr(action, 'tool'):
                tool_name = action.tool
            if not tool_name and hasattr(action, 'tool_name'):
                tool_name = action.tool_name
            if not tool_name and isinstance(action, dict) and 'tool' in action:
                tool_name = action['tool']
            if not tool_name and isinstance(action, dict) and 'name' in action:
                tool_name = action['name']
            
            # Try string parsing as fallback
            if not tool_name:
                action_str = str(action)
                if 'tool=' in action_str:
                    import re
                    match = re.search(r"tool='([^']+)'", action_str)
                    if match:
                        tool_name = match.group(1)
            
            if not tool_name:
                tool_name = 'unknown_tool'
            
            # Get tool input
            tool_input = {}
            if hasattr(action, 'tool_input'):
                tool_input = action.tool_input
            elif hasattr(action, 'input'):
                tool_input = action.input
            elif isinstance(action, dict) and 'tool_input' in action:
                tool_input = action['tool_input']
            
            # Create a more readable display
            with st.expander(f"Step {step_i}: {tool_name}", expanded=False):
                if tool_input:
                    st.markdown("**Input:**")
                    st.json(tool_input)
                st.markdown("**Result:**")
                # Truncate very long observations
                obs_str = str(observation)
                if len(obs_str) > MAX_OBSERVATION_LENGTH:
                    st.text_area(
                        "Tool output (truncated)", 
                        value=obs_str[:MAX_OBSERVATION_LENGTH] + "...", 
                        height=120, 
                        key=f"history_obs_{message_index}_{step_i}_truncated",
                        label_visibility="hidden"
                    )
                    st.caption("(Output truncated for display)")
                else:
                    st.text_area(
                        "Tool output", 
                        value=obs_str, 
                        height=120, 
                        key=f"history_obs_{message_index}_{step_i}_full",
                        label_visibility="hidden"
                    )
        except Exception as e:
            with st.expander(f"Step {step_i}: (parsing error)", expanded=False):
                st.text(f"Error: {e}")
                st.text(f"Step type: {type(step)}")
                st.text(f"Step content: {str(step)[:500]}")
    
    st.markdown("---")


def render_chat_message(msg: dict, message_index: int, key_prefix: str) -> None:
    """
    Render a single chat message with content, intermediate steps, and charts.
    
    Args:
        msg: Message dict with 'role', 'content', optionally 'intermediate_steps' and 'charts'
        message_index: Index of the message in chat history
        key_prefix: Prefix for unique keys (e.g., 'history' or 'current')
    """
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display intermediate steps for assistant messages
        if msg["role"] == "assistant" and "intermediate_steps" in msg:
            render_intermediate_steps(msg["intermediate_steps"], message_index)
        
        # Display charts that were created with this message
        if msg["role"] == "assistant" and "charts" in msg:
            for j, chart_info in enumerate(msg["charts"]):
                display_single_chart(chart_info, f"{key_prefix}_{message_index}_{j}")


def process_user_query(query: str, df: pd.DataFrame) -> None:
    """
    Process a user query by calling the agent and updating session state.
    Uses streaming to display responses as they arrive (ChatGPT-style).
    Modifies st.session_state.chat_history and st.session_state.current_response_charts.
    
    Args:
        query: User's question/request
        df: Current dataframe to analyze
    """
    # Clear current response charts to start fresh
    st.session_state.current_response_charts = []
    
    # Show the user's message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        tool_status_placeholder = st.empty()  # For showing tool execution
        full_response = ""
        intermediate_steps = []
        result = {}
        current_tool_name = None
        
        try:
            # Stream the response
            for chunk in run_mcp_planner_stream(
                query, 
                df, 
                chat_history=st.session_state.chat_history[:-1],  # Exclude current user message
                api_key=None  # Will use OPENAI_API_KEY environment variable (private beta)
            ):
                if chunk["type"] == "thinking":
                    # Show initial thinking indicator
                    tool_status_placeholder.info(f"💭 {chunk['content']}")
                
                elif chunk["type"] == "token":
                    # Clear thinking indicator once we have output
                    if not full_response:
                        tool_status_placeholder.empty()
                    # Append LLM token and display with cursor
                    full_response += chunk["content"]
                    message_placeholder.markdown(full_response + "▌")
                
                elif chunk["type"] == "tool_start":
                    # Show tool execution starting
                    current_tool_name = chunk["name"]
                    tool_status_placeholder.info(f"🔧 Running `{current_tool_name}`...")
                
                elif chunk["type"] == "tool_end":
                    # Tool completed
                    tool_name = chunk["name"]
                    tool_status_placeholder.success(f"✅ `{tool_name}` completed")
                    current_tool_name = None
                
                elif chunk["type"] == "final":
                    # Final result arrived
                    full_response = chunk["output"]
                    intermediate_steps = chunk.get("intermediate_steps", [])
                    result = chunk
                    break
            
            # Clear tool status and remove cursor from response
            tool_status_placeholder.empty()
            message_placeholder.markdown(full_response)
            final_text = full_response
            
        except Exception as e:
            final_text = f"❌ Agent error: {e}"
            message_placeholder.markdown(final_text)
            result = {}

        # Display intermediate steps immediately
        if intermediate_steps:
            current_msg_index = len(st.session_state.chat_history)
            render_intermediate_steps(intermediate_steps, current_msg_index)

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
    if intermediate_steps:
        assistant_message["intermediate_steps"] = intermediate_steps
        
        # Create and store agent transcript for Agent Logs tab
        try:
            formatted_steps = AgentTranscriptLogger.format_intermediate_steps(intermediate_steps)
            transcript_summary = AgentTranscriptLogger.create_transcript_summary(formatted_steps, final_text)
            store_session_transcripts(transcript_summary)
        except Exception as e:
            # Silently fail - don't break the chat if transcript creation fails
            pass
            
    st.session_state.chat_history.append(assistant_message)
    
    # Track token usage and cost for this interaction
    try:
        usage_stats = track_usage(query, final_text, model=DEFAULT_MODEL)
        update_session_usage(usage_stats)
    except Exception as e:
        st.caption(f"Usage tracking error: {e}")


# ---------- Password Authentication (Private Beta) ----------
if not check_password():
    st.stop()

# Initialize all session state variables
initialize_session_state()

# Render sidebar (API key, usage, dataset info)
render_sidebar()

st.title("🧭 Stats Compass")
st.subheader("Turn your raw datasets into structured insights instantly.")

# ---------- File Uploader ----------
# Main area uploader when no dataset, sidebar uploader when dataset exists
render_file_uploader(location="main")

# Guard
if not hasattr(st.session_state, 'df') or st.session_state.df is None:
    st.info("📂 Upload a CSV/XLSX file to get started.")
    st.stop()


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Reports", "Summary", "Explore", "Agent Logs"])

with tab1:
    render_chat_tab(render_chat_message, process_user_query)

with tab2:
    render_reports_tab()

with tab3:
    render_summary_tab()

with tab4:
    render_explore_tab()

with tab5:
    render_logs_tab()
