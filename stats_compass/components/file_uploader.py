"""
File Upload Component

Provides a single canonical file uploader that adapts to context:
- Main area: When no dataset is loaded (initial upload)
- Sidebar: When dataset exists (change dataset)

Eliminates duplication by handling all success/failure UI in one place.
"""

import streamlit as st
from utils.data_loading import process_uploaded_file


def render_file_uploader(location: str = "main") -> None:
    """
    Render file uploader with context-aware UI.
    
    Args:
        location: Where uploader is rendered - "main" (no dataset) or "sidebar" (has dataset)
    
    Location Behavior:
        - main: Large uploader in main area when no dataset loaded
        - sidebar: Compact uploader in sidebar expander when dataset exists
    """
    has_dataset = hasattr(st.session_state, 'df') and st.session_state.df is not None
    
    # Initialize file uploader counter if not exists (for cache clearing)
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    
    if location == "main":
        # Main area uploader when no dataset
        if has_dataset:
            return  # Don't render main uploader if dataset already loaded
            
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV/XLSX)", 
            type=["csv", "xlsx", "xls"],
            help="Upload a CSV or Excel file to start analyzing",
            key=f"main_uploader_{st.session_state.file_uploader_key}"
        )
        
        # Process the uploaded file
        if process_uploaded_file(uploaded_file):
            # Show success info in sidebar (where dataset info lives)
            with st.sidebar:
                st.success(f"✅ Loaded {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns")
                mem_mb = st.session_state.df.memory_usage(deep=True).sum() / (1024**2)
                st.caption(f"Approx. memory usage: {mem_mb:.2f} MB")
                with st.expander("📊 Dataset preview", expanded=False):
                    st.dataframe(st.session_state.df.head(), use_container_width=True)
            st.rerun()
            
    elif location == "sidebar":
        # Sidebar uploader when dataset exists
        if not has_dataset:
            return  # Don't render sidebar uploader if no dataset
            
        with st.expander("📁 Change Dataset", expanded=False):
            sidebar_uploaded_file = st.file_uploader(
                "Upload a different dataset",
                type=["csv", "xlsx", "xls"],
                help="Replace current dataset with a new file",
                key=f"sidebar_uploader_{st.session_state.file_uploader_key}"
            )
            
            # Show current dataset info
            filename = st.session_state.get('uploaded_filename', 'Unknown file')
            st.success("📊 **Current Dataset**")
            st.markdown(f"**📁 {filename}**")
            st.caption(f"{st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns")
            mem_mb = st.session_state.df.memory_usage(deep=True).sum() / (1024**2)
            st.caption(f"Memory: {mem_mb:.2f} MB")
            
            # Process sidebar file upload (clears chat history for new dataset)
            if process_uploaded_file(sidebar_uploaded_file, clear_history=True):
                # Increment key to clear file uploader cache and prevent Streamlit file reference errors
                st.session_state.file_uploader_key += 1
                st.success(f"✅ New dataset loaded: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns")
                # Force refresh to update dataset context
                st.rerun()
