# stats_compass/utils/export_utils.py

import streamlit as st
import json
import zipfile
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional
import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors

# Import functions that this module needs
from .token_tracking import get_session_stats


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


def create_narrative_summary(chat_history: List[Dict]) -> str:
    """Create an AI-generated narrative summary of the analysis"""
    
    # Extract key insights from the conversation
    insights = []
    charts_created = []
    
    for msg in chat_history:
        if msg["role"] == "assistant":
            # Extract key findings (simplified heuristic)
            content = msg["content"]
            if "correlation" in content.lower():
                insights.append("Correlation analysis was performed")
            if "distribution" in content.lower():
                insights.append("Data distribution was analyzed")
            if "top" in content.lower() and ("categories" in content.lower() or "values" in content.lower()):
                insights.append("Top categories were identified")
            
            # Count charts
            if "charts" in msg:
                charts_created.extend([chart.get('type') for chart in msg["charts"]])
    
    # Create narrative
    narrative = f"""## Key Insights Summary

This analysis session generated {len(insights)} key insights and {len(charts_created)} visualizations:

"""
    
    if insights:
        narrative += "**Analysis performed:**\n"
        for insight in insights:
            narrative += f"- {insight}\n"
        narrative += "\n"
    
    if charts_created:
        narrative += "**Visualizations created:**\n"
        chart_types = list(set(charts_created))
        for chart_type in chart_types:
            count = charts_created.count(chart_type)
            narrative += f"- {count} {chart_type} chart(s)\n"
        narrative += "\n"
    
    narrative += """**Recommendations:**
- Review the visualizations to identify patterns and outliers
- Consider the relationships discovered between variables
- Use these insights to inform business decisions
- Share this report with stakeholders for further discussion
"""
    
    return narrative


def export_chart_as_image(chart_info: Dict[str, Any], format: str = "png") -> bytes:
    """Export a single chart as an image (PNG/SVG)"""
    try:
        chart_type = chart_info.get('type')
        data = chart_info.get('data')
        title = chart_info.get('title', 'Chart')
        
        if chart_type == 'histogram':
            fig = px.bar(
                data, 
                x='bin_range', 
                y='count',
                title=title,
                template='plotly_white'
            )
        elif chart_type == 'bar':
            fig = px.bar(
                data, 
                x='category', 
                y='count',
                title=title,
                template='plotly_white'
            )
        elif chart_type == 'scatter':
            fig = px.scatter(
                data, 
                x=chart_info['x_column'], 
                y=chart_info['y_column'],
                color=chart_info.get('color_column'),
                title=title,
                template='plotly_white'
            )
        elif chart_type == 'line':
            fig = px.line(
                data, 
                x=chart_info['x_column'], 
                y=chart_info['y_column'],
                title=title,
                template='plotly_white'
            )
        elif chart_type == 'time_series':
            # Recreate the chart from stored data
            fig = px.line(
                data, 
                x='Date', 
                y='Value',
                title=title,
                template='plotly_white'
            )
            
            # Apply any stored styling
            chart_config = chart_info.get('chart_config', {})
            line_width = chart_config.get('line_width', 2)
            ylabel = chart_config.get('ylabel', 'Value')
            
            fig.update_traces(line=dict(width=line_width))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=ylabel,
                height=500
            )
            
        elif chart_type == 'correlation_heatmap':
            # Recreate the chart from correlation matrix
            corr_matrix = chart_info.get('correlation_matrix', {})
            if corr_matrix:
                df_corr = pd.DataFrame(corr_matrix)
                fig = px.imshow(
                    df_corr,
                    text_auto=True,
                    title=title,
                    template='plotly_white',
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                
                fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textfont_size=10
                )
                
                cols_count = len(chart_info.get('columns', []))
                fig.update_layout(
                    height=max(400, cols_count * 40),
                    width=max(400, cols_count * 40)
                )
            else:
                raise ValueError("No correlation matrix data available")
        elif chart_type == 't_test':
            # Use the stored plotly figure directly but ensure proper template for export
            fig = chart_info.get('fig')
            if not fig:
                raise ValueError("No statistical test figure available")
            
            # Apply export-friendly template and colors
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=12)
            )
            
            # Apply vibrant colors that work well in PDF
            colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            # Update traces with better colors and borders for PDF
            for i, trace in enumerate(fig.data):
                color = colors_list[i % len(colors_list)]
                
                if hasattr(trace, 'marker'):
                    # For scatter plots, histograms
                    fig.data[i].marker.update(
                        color=color,
                        line=dict(width=1, color='black')
                    )
                elif hasattr(trace, 'fillcolor'):
                    # For box plots
                    fig.data[i].update(
                        fillcolor=color,
                        line=dict(color='black', width=2),
                        marker=dict(color='black', size=4)  # outlier markers
                    )
        elif chart_type == 'z_test':
            # Use the stored plotly figure directly but ensure proper template for export
            fig = chart_info.get('fig')
            if not fig:
                raise ValueError("No statistical test figure available")
            
            # Apply export-friendly template and colors
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=12)
            )
            
            # Apply vibrant colors that work well in PDF
            colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            # Update traces with better colors and borders for PDF
            for i, trace in enumerate(fig.data):
                color = colors_list[i % len(colors_list)]
                
                if hasattr(trace, 'marker'):
                    # For scatter plots, histograms
                    fig.data[i].marker.update(
                        color=color,
                        line=dict(width=1, color='black')
                    )
                elif hasattr(trace, 'fillcolor'):
                    # For box plots
                    fig.data[i].update(
                        fillcolor=color,
                        line=dict(color='black', width=2),
                        marker=dict(color='black', size=4)  # outlier markers
                    )
        elif chart_type == 'chi_square_test':
            # Use the stored plotly figure directly but ensure proper template for export
            fig = chart_info.get('fig')
            if not fig:
                raise ValueError("No statistical test figure available")
            
            # Apply export-friendly template and colors
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=12)
            )
            
            # Apply vibrant colors that work well in PDF
            colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            # Update traces with better colors and borders for PDF
            for i, trace in enumerate(fig.data):
                color = colors_list[i % len(colors_list)]
                
                if hasattr(trace, 'marker'):
                    # For scatter plots, histograms
                    fig.data[i].marker.update(
                        color=color,
                        line=dict(width=1, color='black')
                    )
                elif hasattr(trace, 'fillcolor'):
                    # For box plots
                    fig.data[i].update(
                        fillcolor=color,
                        line=dict(color='black', width=2),
                        marker=dict(color='black', size=4)  # outlier markers
                    )
        elif chart_type == 'regression_plot':
            # Use the stored plotly figure directly
            fig = chart_info.get('figure')
            if not fig:
                raise ValueError("No regression plot figure available")
            
            # Apply export-friendly template
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=12)
            )
        elif chart_type == 'residual_plot':
            # Use the stored plotly figure directly
            fig = chart_info.get('figure')
            if not fig:
                raise ValueError("No residual plot figure available")
            
            # Apply export-friendly template
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=12)
            )
        elif chart_type == 'coefficient_chart':
            # Use the stored plotly figure directly
            fig = chart_info.get('figure')
            if not fig:
                raise ValueError("No coefficient chart figure available")
            
            # Apply export-friendly template
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=12)
            )
        elif chart_type == 'feature_importance_chart':
            # Use the stored plotly figure directly
            fig = chart_info.get('figure')
            if not fig:
                raise ValueError("No feature importance chart figure available")
            
            # Apply export-friendly template
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=12)
            )
        elif chart_type == 'roc_curve':
            # Use the stored plotly figure directly
            fig = chart_info.get('figure')
            if not fig:
                raise ValueError("No ROC curve figure available")
            
            # Apply export-friendly template
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=12)
            )
        elif chart_type == 'precision_recall_curve':
            # Use the stored plotly figure directly
            fig = chart_info.get('figure')
            if not fig:
                raise ValueError("No precision-recall curve figure available")
            
            # Apply export-friendly template
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=12)
            )
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Export as image
        if format.lower() == 'png':
            img_bytes = pio.to_image(fig, format='png', width=800, height=600)
        elif format.lower() == 'svg':
            img_bytes = pio.to_image(fig, format='svg', width=800, height=600)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return img_bytes
        
    except Exception as e:
        st.error(f"Error exporting chart: {e}")
        return None


def generate_markdown_report(chat_history: List[Dict], dataset_name: str = "Dataset") -> str:
    """Generate a comprehensive Markdown report from chat history"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown_content = f"""# Data Analysis Report

**Dataset:** {dataset_name}  
**Generated:** {timestamp}  
**Tool:** DS Auto Insights

---

## Executive Summary

This report contains the analysis and insights generated from an interactive data exploration session. The analysis includes various visualizations and statistical findings that provide comprehensive insights into the dataset.

---

## Analysis Session

"""
    
    chart_counter = 1
    
    for i, msg in enumerate(chat_history):
        if msg["role"] == "user":
            markdown_content += f"\n### 🔍 Question {i//2 + 1}\n\n"
            markdown_content += f"{msg['content']}\n\n"
            
        elif msg["role"] == "assistant":
            markdown_content += f"### 💡 Analysis\n\n"
            markdown_content += f"{msg['content']}\n\n"
            
            # Add charts if they exist
            if "charts" in msg:
                for chart in msg["charts"]:
                    chart_title = chart.get('title', f'Chart {chart_counter}')
                    chart_type = chart.get('type', 'chart')
                    
                    markdown_content += f"#### 📊 {chart_title}\n\n"
                    markdown_content += f"*Chart Type:* {chart_type.title()}\n\n"
                    
                    # Add chart placeholder (will be replaced with actual image in PDF)
                    markdown_content += f"![{chart_title}](chart_{chart_counter}.png)\n\n"
                    chart_counter += 1
            
            markdown_content += "---\n\n"
    
    # Add footer
    markdown_content += f"""
## About This Report

This report was automatically generated by DS Auto Insights, a tool that enables natural language data analysis and visualization. All charts and insights were created through conversational queries with an AI-powered data analysis assistant.

**Report Generated:** {timestamp}  
**Tool Version:** DS Auto Insights v1.0
"""
    
    return markdown_content


def export_pdf_report(chat_history: List[Dict], dataset_name: str = "Dataset") -> bytes:
    """Generate a PDF report with embedded charts using reportlab"""
    
    # Create a BytesIO buffer to store the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Build the story (content)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1f2937'),
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#374151')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.HexColor('#4b5563')
    )
    
    # Title page
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("Data Analysis Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Dataset:</b> {dataset_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
    story.append(Paragraph(f"<b>Tool:</b> DS Auto Insights", styles['Normal']))
    story.append(Spacer(1, 24))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        "This report contains the analysis and insights generated from an interactive "
        "data exploration session. The analysis includes various visualizations and "
        "statistical findings that provide comprehensive insights into the dataset.",
        styles['Normal']
    ))
    story.append(Spacer(1, 24))
    
    # Analysis Session
    story.append(Paragraph("Analysis Session", heading_style))
    story.append(Spacer(1, 12))
    
    question_counter = 1
    chart_counter = 1
    
    for i, msg in enumerate(chat_history):
        if msg["role"] == "user":
            story.append(Paragraph(f"Question {question_counter}", subheading_style))
            story.append(Paragraph(msg['content'], styles['Normal']))
            story.append(Spacer(1, 12))
            
        elif msg["role"] == "assistant":
            story.append(Paragraph("Analysis", subheading_style))
            story.append(Paragraph(msg['content'], styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Add charts if they exist
            if "charts" in msg:
                for chart in msg["charts"]:
                    try:
                        # Export chart as PNG
                        chart_bytes = export_chart_as_image(chart, format='png')
                        if chart_bytes:
                            # Create image from bytes
                            chart_img = Image(io.BytesIO(chart_bytes))
                            chart_img.drawHeight = 4 * inch
                            chart_img.drawWidth = 6 * inch
                            
                            chart_title = chart.get('title', f'Chart {chart_counter}')
                            story.append(Paragraph(f"Chart: {chart_title}", subheading_style))
                            story.append(chart_img)
                            story.append(Spacer(1, 12))
                            chart_counter += 1
                    except Exception as e:
                        story.append(Paragraph(f"[Chart could not be generated: {e}]", styles['Normal']))
                        story.append(Spacer(1, 12))
            
            story.append(Spacer(1, 18))
            question_counter += 1
    
    # Add narrative summary
    narrative = create_narrative_summary(chat_history)
    story.append(PageBreak())
    story.append(Paragraph("Report Summary", heading_style))
    
    # Parse the narrative markdown-style content
    for line in narrative.split('\n'):
        if line.strip():
            if line.startswith('##'):
                story.append(Paragraph(line.replace('##', '').strip(), heading_style))
            elif line.startswith('**') and line.endswith('**'):
                # Properly handle bold markdown
                content = line.strip()
                if content.startswith('**') and content.endswith('**'):
                    content = f"<b>{content[2:-2]}</b>"
                story.append(Paragraph(content, styles['Normal']))
            elif line.startswith('- '):
                story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
            else:
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 6))
    
    # Footer
    story.append(Spacer(1, 24))
    story.append(Paragraph("About This Report", heading_style))
    story.append(Paragraph(
        f"This report was automatically generated by DS Auto Insights on {timestamp}. "
        "All charts and insights were created through conversational queries with an "
        "AI-powered data analysis assistant.",
        styles['Normal']
    ))
    
    # Build the PDF
    try:
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None


def export_session_data(chat_history: List[Dict], charts_data: List[Dict]) -> Dict[str, Any]:
    """Export session data as JSON for later analysis"""
    
    # Clean chart data to remove DataFrames
    clean_charts_metadata = []
    for chart in charts_data:
        if chart:
            clean_chart = {
                "type": chart.get("type"),
                "title": chart.get("title"), 
                "x_column": chart.get("x_column", ""),
                "y_column": chart.get("y_column", ""),
                "color_column": chart.get("color_column", ""),
                "correlation": chart.get("correlation", 0),
                # Don't include the actual DataFrame data, just metadata
                "data_shape": list(chart.get("data").shape) if hasattr(chart.get("data"), 'shape') else None
            }
            clean_charts_metadata.append(clean_chart)
    
    # Clean chat history to remove any DataFrame references
    clean_chat_history = []
    for msg in chat_history:
        clean_msg = {
            "role": msg["role"],
            "content": msg["content"],
            # Don't include charts data since it contains DataFrames
            "has_charts": "charts" in msg and len(msg.get("charts", [])) > 0
        }
        clean_chat_history.append(clean_msg)
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "tool_version": "1.0",
        "session_data": {
            "chat_history": clean_chat_history,
            "charts_metadata": clean_charts_metadata
        }
    }
    
    return export_data
