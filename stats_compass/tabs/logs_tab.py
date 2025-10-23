"""
Agent Logs tab for Stats Compass application.

Displays AI agent reasoning and execution logs including:
- Agent thinking process
- Tool usage breakdown
- Step-by-step analysis details
- Transcript export options
"""

import streamlit as st
from utils.agent_transcript import (
    AgentTranscriptDisplay,
    AgentTranscriptExporter,
    render_transcript_history
)


def render_logs_tab() -> None:
    """Render the agent logs tab with reasoning details and transcripts."""
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
