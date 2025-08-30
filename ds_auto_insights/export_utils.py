# ds_auto_insights/export_utils.py

import os
import io
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import streamlit as st
import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors


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
            # Use the stored chart object if available, otherwise create new
            if 'chart_object' in chart_info:
                fig = chart_info['chart_object']
                fig.update_layout(template='plotly_white')
            else:
                fig = px.line(
                    data, 
                    x='Date', 
                    y='Value',
                    title=title,
                    template='plotly_white'
                )
        elif chart_type == 'correlation_heatmap':
            # Use the stored chart object if available, otherwise create new
            if 'chart_object' in chart_info:
                fig = chart_info['chart_object']
                fig.update_layout(template='plotly_white')
            else:
                # Reconstruct from correlation matrix
                corr_matrix = chart_info.get('correlation_matrix', {})
                if corr_matrix:
                    import pandas as pd
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
                else:
                    raise ValueError("No correlation matrix data available")
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
