import streamlit as st
from datetime import datetime, timedelta

def check_usage_limits():
    """Simple usage limiting - max queries per session"""
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
        st.session_state.session_start = datetime.now()
    
    # Limit: 20 queries per session
    MAX_QUERIES = 20
    
    if st.session_state.query_count >= MAX_QUERIES:
        st.error(f"🚫 Session limit reached ({MAX_QUERIES} queries). Please refresh to start a new session.")
        return False
    
    return True

def increment_query_count():
    """Call this each time a query is made"""
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    st.session_state.query_count += 1
