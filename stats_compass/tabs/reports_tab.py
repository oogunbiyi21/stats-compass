"""
Reports tab for Stats Compass application.

Handles report generation and export functionality including:
- Session statistics
- Multi-format export options (text, charts, data, models)
- Narrative summary generation
- Report previews
"""

import streamlit as st
from utils.export_utils import (
    render_session_summary,
    render_text_export_buttons,
    render_data_export_buttons,
    render_dataframe_export_buttons,
    render_model_export_buttons,
    render_report_preview,
    create_narrative_summary
)


def render_reports_tab() -> None:
    """Render the reports and export tab."""
    st.header("📄 Reports & Export")
    
    if not st.session_state.chat_history:
        st.info("💬 Start a conversation in the Chat tab to generate reports.")
    else:
        # Get filename for exports
        filename = st.session_state.get('uploaded_filename', 'dataset')
        
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
