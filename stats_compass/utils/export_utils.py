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
import joblib
import pickle
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors

# Import functions that this module needs
from .token_tracking import get_session_stats
from .agent_transcript import get_latest_transcript_summary


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


def create_dataframe_exports(filename: str) -> Dict[str, bytes]:
    """
    Create downloadable versions of dataframes from session state.
    
    Args:
        filename: Base filename for exports
        
    Returns:
        Dictionary of export formats and their byte data
    """
    exports = {}
    
    try:
        # Current dataframe (with all transformations)
        if hasattr(st.session_state, 'df') and st.session_state.df is not None:
            df = st.session_state.df
            
            # CSV export
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            exports['current_csv'] = csv_buffer.getvalue().encode('utf-8')
            
            # Excel export
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            exports['current_excel'] = excel_buffer.getvalue()
            
            # JSON export
            json_buffer = io.StringIO()
            df.to_json(json_buffer, orient='records', indent=2)
            exports['current_json'] = json_buffer.getvalue().encode('utf-8')
        
        # Original dataframe (before transformations)
        if hasattr(st.session_state, 'original_df') and st.session_state.original_df is not None:
            orig_df = st.session_state.original_df
            
            # CSV export
            csv_buffer = io.StringIO()
            orig_df.to_csv(csv_buffer, index=False)
            exports['original_csv'] = csv_buffer.getvalue().encode('utf-8')
            
            # Excel export
            excel_buffer = io.BytesIO()
            orig_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            exports['original_excel'] = excel_buffer.getvalue()
    
    except Exception as e:
        st.error(f"Error creating dataframe exports: {e}")
    
    return exports


def create_model_exports(filename: str) -> Dict[str, bytes]:
    """
    Create downloadable versions of trained models from session state.
    
    Args:
        filename: Base filename for exports
        
    Returns:
        Dictionary of model files and their byte data
    """
    exports = {}
    
    try:
        # Linear regression models
        if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
            models = st.session_state.trained_models
            
            for model_id, model_data in models.items():
                if 'model_type' in model_data:
                    model_type = model_data['model_type']
                    model_obj = model_data.get('model')
                    
                    if model_obj is not None:
                        # Serialize model using joblib (better for sklearn models)
                        model_buffer = io.BytesIO()
                        joblib.dump(model_data, model_buffer)
                        model_buffer.seek(0)
                        
                        # Create filename based on model type and timestamp
                        timestamp = model_data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
                        model_filename = f"{model_type}_{model_id}_{timestamp}.joblib"
                        
                        exports[model_filename] = model_buffer.getvalue()
                        
                        # Also create a metadata file
                        metadata = {
                            'model_type': model_type,
                            'model_id': model_id,
                            'timestamp': timestamp,
                            'features': model_data.get('features', []),
                            'target': model_data.get('target', ''),
                            'metrics': model_data.get('metrics', {}),
                            'parameters': model_data.get('parameters', {}),
                            'usage_instructions': {
                                'load_model': "import joblib; model_data = joblib.load('filename.joblib')",
                                'access_model': "model = model_data['model']",
                                'make_predictions': "predictions = model.predict(X_new)"
                            }
                        }
                        
                        metadata_buffer = io.StringIO()
                        json.dump(metadata, metadata_buffer, indent=2, default=str)
                        metadata_filename = f"{model_type}_{model_id}_{timestamp}_metadata.json"
                        exports[metadata_filename] = metadata_buffer.getvalue().encode('utf-8')
        
        # ARIMA models (special handling due to different serialization needs)
        if hasattr(st.session_state, 'arima_models') and st.session_state.arima_models:
            arima_models = st.session_state.arima_models
            
            for model_id, arima_data in arima_models.items():
                try:
                    # Create a comprehensive ARIMA export package
                    arima_export = {
                        'model_id': model_id,
                        'timestamp': arima_data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S')),
                        'arima_order': arima_data.get('arima_order', (0, 0, 0)),
                        'fitted_model': arima_data.get('fitted_model'),  # This might be a large object
                        'forecast_results': arima_data.get('forecast_results', {}),
                        'model_summary': arima_data.get('model_summary', ''),
                        'data_info': arima_data.get('data_info', {}),
                        'usage_instructions': {
                            'load_data': "import pickle; arima_data = pickle.load(open('filename.pkl', 'rb'))",
                            'access_model': "fitted_model = arima_data['fitted_model']",
                            'forecast': "forecast = fitted_model.forecast(steps=n_periods)"
                        }
                    }
                    
                    # Use pickle for ARIMA models (they may not be joblib compatible)
                    arima_buffer = io.BytesIO()
                    pickle.dump(arima_export, arima_buffer)
                    arima_buffer.seek(0)
                    
                    timestamp = arima_data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
                    arima_filename = f"arima_{model_id}_{timestamp}.pkl"
                    exports[arima_filename] = arima_buffer.getvalue()
                    
                except Exception as e:
                    st.warning(f"Could not export ARIMA model {model_id}: {e}")
    
    except Exception as e:
        st.error(f"Error creating model exports: {e}")
    
    return exports


def create_combined_export_zip(filename: str) -> bytes:
    """
    Create a comprehensive ZIP file containing all data and models.
    
    Args:
        filename: Base filename for the ZIP
        
    Returns:
        ZIP file as bytes
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add dataframes
        df_exports = create_dataframe_exports(filename)
        for export_name, export_data in df_exports.items():
            if 'csv' in export_name:
                zip_file.writestr(f"data/{export_name}.csv", export_data)
            elif 'excel' in export_name:
                zip_file.writestr(f"data/{export_name}.xlsx", export_data)
            elif 'json' in export_name:
                zip_file.writestr(f"data/{export_name}.json", export_data)
        
        # Add models
        model_exports = create_model_exports(filename)
        for model_filename, model_data in model_exports.items():
            zip_file.writestr(f"models/{model_filename}", model_data)
        
        # Add a README file with instructions
        readme_content = f"""# Data Science Export Package
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: Stats Compass Analysis

## Contents

### Data Files (./data/)
- *_current_*: Transformed data after all analysis steps
- *_original_*: Original uploaded data before transformations
- Formats: CSV (recommended), Excel, JSON

### Model Files (./models/)
- *.joblib: Scikit-learn models (linear/logistic regression)
- *.pkl: ARIMA time series models
- *_metadata.json: Model information and usage instructions

## Loading Models

### Scikit-learn Models (.joblib)
```python
import joblib
model_data = joblib.load('model_file.joblib')
model = model_data['model']
predictions = model.predict(X_new)
```

### ARIMA Models (.pkl)
```python
import pickle
arima_data = pickle.load(open('arima_file.pkl', 'rb'))
fitted_model = arima_data['fitted_model']
forecast = fitted_model.forecast(steps=10)
```

## Support
For questions about this export, refer to the original Stats Compass session
or consult the metadata files for specific model details.
"""
        zip_file.writestr("README.md", readme_content)
    
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


def render_dataframe_export_buttons(filename: str, location: str = "sidebar"):
    """
    Render dataframe export buttons for current and original data.
    
    Args:
        filename: Base filename for exports
        location: Where buttons are rendered ("sidebar" or "tab")
    """
    key_suffix = f"_{location}"
    
    # Check if dataframes are available
    has_current_df = hasattr(st.session_state, 'df') and st.session_state.df is not None
    has_original_df = hasattr(st.session_state, 'original_df') and st.session_state.original_df is not None
    
    if not has_current_df and not has_original_df:
        st.info("📊 No data available for export. Upload a dataset first!")
        return
    
    if location == "tab":
        st.write("**Available Data:**")
        if has_current_df:
            rows, cols = st.session_state.df.shape
            st.write(f"• Current Data: {rows:,} rows × {cols} columns (after transformations)")
        if has_original_df:
            rows, cols = st.session_state.original_df.shape
            st.write(f"• Original Data: {rows:,} rows × {cols} columns (as uploaded)")
        st.write("")
    
    # Current dataframe exports
    if has_current_df:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Current CSV", key=f"current_csv{key_suffix}", 
                        help="Download transformed data as CSV", use_container_width=True):
                try:
                    df_exports = create_dataframe_exports(filename)
                    if 'current_csv' in df_exports:
                        st.download_button(
                            label="⬇️ Download Current CSV",
                            data=df_exports['current_csv'],
                            file_name=f"current_data_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            key=f"current_csv_download{key_suffix}",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            if st.button("📊 Current Excel", key=f"current_excel{key_suffix}", 
                        help="Download transformed data as Excel", use_container_width=True):
                try:
                    df_exports = create_dataframe_exports(filename)
                    if 'current_excel' in df_exports:
                        st.download_button(
                            label="⬇️ Download Current Excel",
                            data=df_exports['current_excel'],
                            file_name=f"current_data_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"current_excel_download{key_suffix}",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col3:
            if st.button("🔗 Current JSON", key=f"current_json{key_suffix}", 
                        help="Download transformed data as JSON", use_container_width=True):
                try:
                    df_exports = create_dataframe_exports(filename)
                    if 'current_json' in df_exports:
                        st.download_button(
                            label="⬇️ Download Current JSON",
                            data=df_exports['current_json'],
                            file_name=f"current_data_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json",
                            key=f"current_json_download{key_suffix}",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Original dataframe exports (only if different from current)
    if has_original_df and location == "tab":
        st.write("**Original Data (before transformations):**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Original CSV", key=f"original_csv{key_suffix}", 
                        help="Download original data as CSV", use_container_width=True):
                try:
                    df_exports = create_dataframe_exports(filename)
                    if 'original_csv' in df_exports:
                        st.download_button(
                            label="⬇️ Download Original CSV",
                            data=df_exports['original_csv'],
                            file_name=f"original_data_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            key=f"original_csv_download{key_suffix}",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            if st.button("📊 Original Excel", key=f"original_excel{key_suffix}", 
                        help="Download original data as Excel", use_container_width=True):
                try:
                    df_exports = create_dataframe_exports(filename)
                    if 'original_excel' in df_exports:
                        st.download_button(
                            label="⬇️ Download Original Excel",
                            data=df_exports['original_excel'],
                            file_name=f"original_data_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"original_excel_download{key_suffix}",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error: {e}")


def render_model_export_buttons(filename: str, location: str = "sidebar"):
    """
    Render model export buttons for trained models.
    
    Args:
        filename: Base filename for exports
        location: Where buttons are rendered ("sidebar" or "tab")
    """
    key_suffix = f"_{location}"
    
    # Check for available models
    has_models = False
    model_count = 0
    arima_count = 0
    
    if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
        model_count = len(st.session_state.trained_models)
        has_models = True
    
    if hasattr(st.session_state, 'arima_models') and st.session_state.arima_models:
        arima_count = len(st.session_state.arima_models)
        has_models = True
    
    if not has_models:
        st.info("🤖 No trained models available. Run linear/logistic regression or ARIMA analysis first!")
        return
    
    if location == "tab":
        st.write("**Available Models:**")
        if model_count > 0:
            st.write(f"• ML Models: {model_count} (Linear/Logistic Regression)")
        if arima_count > 0:
            st.write(f"• Time Series Models: {arima_count} (ARIMA)")
        st.write("")
    
    # Individual model exports
    if model_count > 0:
        if st.button("🤖 Download ML Models", key=f"ml_models{key_suffix}", 
                    help="Download all linear/logistic regression models", use_container_width=True):
            try:
                model_exports = create_model_exports(filename)
                ml_exports = {k: v for k, v in model_exports.items() if not k.startswith('arima_')}
                
                if ml_exports:
                    # Create a ZIP with ML models
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for model_filename, model_data in ml_exports.items():
                            zip_file.writestr(model_filename, model_data)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="⬇️ Download ML Models ZIP",
                        data=zip_buffer.getvalue(),
                        file_name=f"ml_models_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                        mime="application/zip",
                        key=f"ml_models_download{key_suffix}",
                        use_container_width=True
                    )
                    st.success(f"✅ {len(ml_exports)} ML model files ready for download!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    if arima_count > 0:
        if st.button("📈 Download ARIMA Models", key=f"arima_models{key_suffix}", 
                    help="Download all ARIMA time series models", use_container_width=True):
            try:
                model_exports = create_model_exports(filename)
                arima_exports = {k: v for k, v in model_exports.items() if k.startswith('arima_')}
                
                if arima_exports:
                    # Create a ZIP with ARIMA models
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for model_filename, model_data in arima_exports.items():
                            zip_file.writestr(model_filename, model_data)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="⬇️ Download ARIMA Models ZIP",
                        data=zip_buffer.getvalue(),
                        file_name=f"arima_models_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                        mime="application/zip",
                        key=f"arima_models_download{key_suffix}",
                        use_container_width=True
                    )
                    st.success(f"✅ {len(arima_exports)} ARIMA model files ready for download!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Combined export
    if location == "tab":
        st.write("**Complete Package:**")
        if st.button("📦 Download Everything", key=f"everything{key_suffix}", 
                    help="Download all data and models in one ZIP file", use_container_width=True):
            try:
                combined_zip = create_combined_export_zip(filename)
                st.download_button(
                    label="⬇️ Download Complete Package",
                    data=combined_zip,
                    file_name=f"complete_export_{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                    mime="application/zip",
                    key=f"everything_download{key_suffix}",
                    use_container_width=True
                )
                st.success("✅ Complete export package ready for download!")
            except Exception as e:
                st.error(f"Error: {e}")


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
**Tool:** Stats Compass

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
    
    # Add AI Agent Analysis Summary
    try:
        agent_transcript = get_latest_transcript_summary()
        if agent_transcript and agent_transcript.get('summary'):
            markdown_content += "\n---\n\n## 🤖 AI Agent Analysis Summary\n\n"
            
            # Agent session overview
            session_info = agent_transcript.get('session_info', {})
            markdown_content += "### Analysis Session Overview\n\n"
            markdown_content += f"- **Session ID:** {agent_transcript.get('session_id', 'N/A')}\n"
            markdown_content += f"- **Duration:** {session_info.get('duration', 'N/A')}\n"
            markdown_content += f"- **Total Actions:** {session_info.get('total_actions', 'N/A')}\n\n"
            
            # Agent reasoning and approach
            if agent_transcript.get('approach_summary'):
                markdown_content += "### Analysis Approach\n\n"
                markdown_content += f"{agent_transcript['approach_summary']}\n\n"
            
            # Key decisions and tools used
            if agent_transcript.get('tools_used'):
                markdown_content += "### Tools and Methods Used\n\n"
                tools_summary = agent_transcript.get('tools_summary', 'Various analytical tools were employed during the analysis.')
                markdown_content += f"{tools_summary}\n\n"
                
                tools_list = agent_transcript.get('tools_used', [])
                if tools_list:
                    markdown_content += "**Tools Utilized:**\n"
                    for tool in tools_list[:15]:  # Limit to first 15 tools for markdown
                        markdown_content += f"- {tool}\n"
                    markdown_content += "\n"
            
            # Agent insights and reasoning
            if agent_transcript.get('insights'):
                markdown_content += "### Key Insights from AI Analysis\n\n"
                markdown_content += f"{agent_transcript.get('insights', '')}\n\n"
                
            markdown_content += "---\n\n"
    except Exception as e:
        # If agent transcript fails, continue without it
        markdown_content += "\n## 🤖 AI Agent Analysis Summary\n\n*Agent analysis summary not available.*\n\n---\n\n"
    
    # Add footer
    markdown_content += f"""
## About This Report

This report was automatically generated by Stats Compass, a tool that enables natural language data analysis and visualization. All charts and insights were created through conversational queries with an AI-powered data analysis assistant.

**Report Generated:** {timestamp}  
**Tool Version:** Stats Compass v1.0
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
    story.append(Paragraph(f"<b>Tool:</b> Stats Compass", styles['Normal']))
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
    
    # Add AI Agent Analysis Summary
    try:
        agent_transcript = get_latest_transcript_summary()
        if agent_transcript and agent_transcript.get('summary'):
            story.append(PageBreak())
            story.append(Paragraph("AI Agent Analysis Summary", heading_style))
            story.append(Spacer(1, 12))
            
            # Agent session overview
            story.append(Paragraph("Analysis Session Overview", subheading_style))
            session_info = agent_transcript.get('session_info', {})
            story.append(Paragraph(f"Session ID: {agent_transcript.get('session_id', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"Duration: {session_info.get('duration', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"Total Actions: {session_info.get('total_actions', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Agent reasoning and approach
            if agent_transcript.get('approach_summary'):
                story.append(Paragraph("Analysis Approach", subheading_style))
                approach_lines = agent_transcript['approach_summary'].split('\n')
                for line in approach_lines:
                    if line.strip():
                        if line.strip().startswith('•') or line.strip().startswith('-'):
                            story.append(Paragraph(line.strip(), styles['Normal']))
                        else:
                            story.append(Paragraph(line.strip(), styles['Normal']))
                        story.append(Spacer(1, 3))
                story.append(Spacer(1, 12))
            
            # Key decisions and tools used
            if agent_transcript.get('tools_used'):
                story.append(Paragraph("Tools and Methods Used", subheading_style))
                tools_summary = agent_transcript.get('tools_summary', 'Various analytical tools were employed during the analysis.')
                story.append(Paragraph(tools_summary, styles['Normal']))
                story.append(Spacer(1, 6))
                
                tools_list = agent_transcript.get('tools_used', [])
                for tool in tools_list[:10]:  # Limit to first 10 tools
                    story.append(Paragraph(f"• {tool}", styles['Normal']))
                    story.append(Spacer(1, 3))
                story.append(Spacer(1, 12))
            
            # Agent insights and reasoning
            if agent_transcript.get('insights'):
                story.append(Paragraph("Key Insights from AI Analysis", subheading_style))
                insights_text = agent_transcript.get('insights', '')
                # Split long insights into paragraphs
                insights_lines = insights_text.split('\n')
                for line in insights_lines:
                    if line.strip():
                        story.append(Paragraph(line.strip(), styles['Normal']))
                        story.append(Spacer(1, 6))
                story.append(Spacer(1, 12))
    except Exception as e:
        # If agent transcript fails, continue without it
        story.append(Paragraph("AI Agent Summary: Not available", styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Footer
    story.append(Spacer(1, 24))
    story.append(Paragraph("About This Report", heading_style))
    story.append(Paragraph(
        f"This report was automatically generated by Stats Compass on {timestamp}. "
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
