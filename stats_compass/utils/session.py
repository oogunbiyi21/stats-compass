"""
Session State Initialization

Centralizes all session state initialization for the Stats Compass application.
Provides a single source of truth for session variable defaults and structure.
"""

import streamlit as st


def initialize_session_state() -> None:
    """
    Initialize all session state variables with their default values.
    
    This function is idempotent - it only sets values that don't already exist.
    Call this early in app.py to ensure all required state is available.
    
    Session State Variables:
        - df: Current working dataframe (None until file uploaded)
        - original_df: Preserved copy of uploaded file (for exports)
        - uploaded_filename: Name of uploaded file
        - chat_history: List of chat messages [{"role": "user"/"assistant", "content": "..."}]
        - chart_data: List of all charts created during session
        - current_response_charts: Temporary storage for charts in current response
        - session_transcripts: List of agent execution transcripts for Agent Logs tab
        - total_tokens: Cumulative token usage across session
        - total_cost: Cumulative cost in USD
        - usage_history: List of per-query usage stats
    """
    # Dataset state
    if "df" not in st.session_state:
        st.session_state.df = None
    
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
        
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None
    
    # Chat state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chart state
    if "chart_data" not in st.session_state:
        st.session_state.chart_data = []
        
    if "current_response_charts" not in st.session_state:
        st.session_state.current_response_charts = []
    
    # Agent logs state
    if "session_transcripts" not in st.session_state:
        st.session_state.session_transcripts = []
    
    # Usage tracking state
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
        
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
        
    if "usage_history" not in st.session_state:
        st.session_state.usage_history = []
