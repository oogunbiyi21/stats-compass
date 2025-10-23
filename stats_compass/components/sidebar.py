"""
Sidebar component for Stats Compass application.

This module handles all sidebar rendering logic including:
- API key management
- Token usage tracking
- Dataset information and export options
- File upload for changing datasets
"""

import io
import pandas as pd
import streamlit as st

from config import RECENT_USAGE_DISPLAY_COUNT
from api_key_auth import render_sidebar_api_key_widget
from utils.token_tracking import get_usage_summary, check_usage_limits
from utils.export_utils import render_session_summary, render_export_buttons
from utils.data_loading import process_uploaded_file
from planner_mcp import generate_dataset_context


def render_api_key_section() -> None:
    """Render the API key management widget."""
    render_sidebar_api_key_widget()
    st.divider()


def render_usage_section() -> None:
    """
    Render token usage information in an expandable section.
    Shows total usage, warnings, and recent interaction breakdown.
    """
    with st.expander("💰 Token Usage", expanded=False):
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
                recent = usage_history[-RECENT_USAGE_DISPLAY_COUNT:] if len(usage_history) > RECENT_USAGE_DISPLAY_COUNT else usage_history
                for i, usage in enumerate(recent, 1):
                    st.caption(f"Query {len(usage_history) - len(recent) + i}: {usage['total_tokens']} tokens (${usage['cost']:.4f})")
        else:
            st.info("💡 No usage yet. Start a conversation to see token costs.")


def render_dataset_section() -> None:
    """
    Render dataset information, quick export options, and change dataset controls.
    Only shown when a dataset is loaded.
    """
    if not hasattr(st.session_state, 'df') or st.session_state.df is None:
        return
    
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


def render_sidebar() -> None:
    """
    Main function to render the complete sidebar.
    Orchestrates all sidebar sections in the correct order.
    """
    with st.sidebar:
        render_api_key_section()
        render_usage_section()
        render_dataset_section()
