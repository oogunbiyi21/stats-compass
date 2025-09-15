# ds_auto_insights/utils/export_utils.py

import streamlit as st
import json
import zipfile
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import functions that this module needs
from .token_tracking import get_session_stats
# Import from the main export_utils module at package level
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from export_utils import (
    generate_markdown_report, 
    export_pdf_report, 
    export_session_data,
    export_chart_as_image
)


def render_session_summary(chat_history: list, location: str = "sidebar"):
    """
    Render session statistics summary.
    
    Args:
        chat_history: List of chat messages
        location: Where this is being rendered ("sidebar" or "tab")
    """
    stats = get_session_stats(chat_history)
    
    if location == "sidebar":
        if stats["has_analysis"]:
            st.caption(f"💬 {stats['user_questions']} questions • 📊 {stats['total_charts']} charts")
        else:
            st.caption("💡 Start analyzing to generate reports")
    else:  # tab
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Questions Asked", stats['user_questions'])
        with col2:
            st.metric("AI Responses", stats['assistant_responses'])
        with col3:
            st.metric("Charts Created", stats['total_charts'])


def create_charts_zip(chat_history: list, filename: str):
    """
    Create a ZIP file containing all charts from the chat history.
    
    Args:
        chat_history: List of chat messages
        filename: Base filename for the ZIP
    
    Returns:
        Bytes of the ZIP file, or None if no charts found
    """
    # Check if any charts actually exist
    total_charts = sum(len(msg.get("charts", [])) for msg in chat_history if msg["role"] == "assistant")
    
    if total_charts == 0:
        return None
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        chart_count = 1
        for msg in chat_history:
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
    return zip_buffer.getvalue()


def render_text_export_buttons(chat_history: list, filename: str, location: str = "sidebar"):
    """
    Render text export buttons (Markdown, PDF).
    
    Args:
        chat_history: List of chat messages
        filename: Base filename for exports
        location: Where buttons are rendered ("sidebar" or "tab")
    """
    stats = get_session_stats(chat_history)
    key_suffix = f"_{location}"
    
    # Markdown Export
    md_button_text = "📝 Markdown" if location == "sidebar" else "Export Markdown Report"
    md_key = f"md_export{key_suffix}"
    
    if st.button(md_button_text, help="Export conversation as Markdown", 
                 use_container_width=True, key=md_key):
        if not stats["has_analysis"]:
            st.warning("⚠️ Can't generate report without analysis. Ask questions about your data first!")
        else:
            try:
                markdown_content = generate_markdown_report(chat_history, filename)
                st.download_button(
                    label="⬇️ Download Markdown",
                    data=markdown_content,
                    file_name=f"analysis_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    key=f"md_download{key_suffix}",
                    use_container_width=True
                )
                if location == "tab":
                    st.success("✅ Markdown report ready for download!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # PDF Export
    pdf_button_text = "📊 PDF Report" if location == "sidebar" else "Export PDF Report"
    pdf_key = f"pdf_export{key_suffix}"
    
    if st.button(pdf_button_text, help="Export full report with charts as PDF", 
                 use_container_width=True, key=pdf_key):
        if not stats["has_analysis"]:
            st.warning("⚠️ Can't generate report without analysis. Ask questions about your data first!")
        else:
            try:
                spinner_text = "Generating PDF..." if location == "sidebar" else "🔄 Generating PDF with embedded charts..."
                with st.spinner(spinner_text):
                    pdf_bytes = export_pdf_report(chat_history, filename)
                    if pdf_bytes:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=f"analysis_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            key=f"pdf_download{key_suffix}",
                            use_container_width=True
                        )
                        if location == "tab":
                            st.success("✅ PDF report ready for download!")
            except Exception as e:
                st.error(f"Error: {e}")


def render_data_export_buttons(chat_history: list, filename: str, location: str = "sidebar"):
    """
    Render data export buttons (Charts ZIP, Session Data).
    
    Args:
        chat_history: List of chat messages
        filename: Base filename for exports
        location: Where buttons are rendered ("sidebar" or "tab")
    """
    stats = get_session_stats(chat_history)
    key_suffix = f"_{location}"
    
    # Charts ZIP Export
    charts_button_text = "📈 Charts ZIP" if location == "sidebar" else "Export All Charts"
    charts_key = f"charts_export{key_suffix}"
    
    if st.button(charts_button_text, help="Download all charts as images", 
                 use_container_width=True, key=charts_key):
        if not stats["has_analysis"]:
            warning_text = ("⚠️ Can't export charts without analysis. Ask questions to generate charts first!" 
                          if location == "sidebar" 
                          else "⚠️ Can't export charts without analysis. Ask questions that generate visualizations first!")
            st.warning(warning_text)
        else:
            if stats["total_charts"] == 0:
                st.warning("⚠️ No charts found to export! Ask questions that generate visualizations first.")
            else:
                try:
                    zip_bytes = create_charts_zip(chat_history, filename)
                    if zip_bytes:
                        download_label = "⬇️ Download Charts" if location == "sidebar" else "⬇️ Download Charts ZIP"
                        st.download_button(
                            label=download_label,
                            data=zip_bytes,
                            file_name=f"charts_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                            mime="application/zip",
                            key=f"charts_download{key_suffix}",
                            use_container_width=True
                        )
                        if location == "tab":
                            st.success("✅ Charts ZIP ready for download!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Session Data Export (only in tab view)
    if location == "tab":
        if st.button("Export Session Data", key=f"data_export{key_suffix}", use_container_width=True):
            try:
                all_charts = []
                for msg in chat_history:
                    if msg["role"] == "assistant" and "charts" in msg:
                        all_charts.extend(msg["charts"])
                
                session_data = export_session_data(chat_history, all_charts)
                
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json.dumps(session_data, indent=2),
                    file_name=f"session_data_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    key="data_download",
                    use_container_width=True
                )
                st.success("✅ Session data ready for download!")
            except Exception as e:
                st.error(f"Error exporting session data: {e}")


def render_export_buttons(chat_history: list, filename: str, location: str = "sidebar"):
    """
    Render all export buttons with unified logic.
    
    Args:
        chat_history: List of chat messages
        filename: Base filename for exports
        location: Where buttons are rendered ("sidebar" or "tab")
    """
    # For sidebar, render all buttons together
    if location == "sidebar":
        render_text_export_buttons(chat_history, filename, location)
        render_data_export_buttons(chat_history, filename, location)
    else:
        # For tab, this function is not used - use specific functions in columns
        pass


def render_report_preview(chat_history: list, filename: str):
    """
    Render report preview section.
    
    Args:
        chat_history: List of chat messages
        filename: Base filename for preview
    """
    st.subheader("👀 Report Preview")
    
    preview_type = st.radio(
        "Choose preview type:",
        ["Markdown", "Session Summary"],
        horizontal=True
    )
    
    if preview_type == "Markdown":
        try:
            markdown_content = generate_markdown_report(chat_history, filename)
            st.markdown("**Preview (first 1000 characters):**")
            st.code(markdown_content[:1000] + "..." if len(markdown_content) > 1000 else markdown_content, language="markdown")
        except Exception as e:
            st.error(f"Error generating preview: {e}")
    
    elif preview_type == "Session Summary":
        st.markdown("**Chat Session Overview:**")
        stats = get_session_stats(chat_history)
        
        for i, msg in enumerate(chat_history):
            if msg["role"] == "user":
                st.markdown(f"**Q{i//2 + 1}:** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
            elif msg["role"] == "assistant":
                charts_info = f" (+{len(msg.get('charts', []))} charts)" if msg.get('charts') else ""
                st.markdown(f"**A{i//2 + 1}:** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}{charts_info}")
                
        if stats["total_charts"] > 0:
            st.info(f"💡 This session created {stats['total_charts']} visualizations that will be included in exported reports.")
