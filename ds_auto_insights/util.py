# ds_auto_insights/utils.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
from typing import Dict, Tuple, Optional
from datetime import datetime

# ---------- Data loading ----------
def load_table(uploaded_file):
    name = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else ""
    is_excel = name.endswith(".xlsx") or name.endswith(".xls")

    if is_excel:
        try:
            # Check if Excel file has multiple sheets
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            # Load the first sheet
            df = pd.read_excel(uploaded_file, sheet_name=0)
            
            # Show warning if multiple sheets detected
            if len(sheet_names) > 1:
                first_sheet = sheet_names[0]
                other_sheets = sheet_names[1:]
                other_sheets_str = ", ".join([f"'{sheet}'" for sheet in other_sheets])
                
                st.warning(
                    f"⚠️ **Multiple sheets detected in Excel file!**\n\n"
                    f"📊 **Currently analyzing:** '{first_sheet}' (first sheet)\n\n"
                    f"📋 **Other sheets found:** {other_sheets_str}\n\n"
                    f"💡 **Tip:** To analyze other sheets, save them as separate Excel files or CSV files and upload individually."
                )
            
            return df
            
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")

    for opts in ({}, {"encoding": "utf-8-sig"}, {"sep": ";"}, {"on_bad_lines": "skip"}):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, **opts)
        except Exception:
            continue

    try:
        raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        for opts in ({}, {"encoding": "utf-8-sig"}, {"sep": ";"}, {"on_bad_lines": "skip"}):
            try:
                df = pd.read_csv(io.BytesIO(raw), **opts)
                # Check if DataFrame is empty or has no columns
                if df.empty or df.shape[1] == 0:
                    raise ValueError("The uploaded file appears to be empty or contains no data columns.")
                return df
            except ValueError as ve:
                # Re-raise ValueError to maintain the empty file message
                if "empty" in str(ve).lower():
                    raise ve
            except Exception:
                continue
    except Exception:
        pass

    raise ValueError("Unable to parse file as CSV or Excel. Try re-exporting or cleaning the file.")

#---------- Display functions ----------
def process_uploaded_file(uploaded_file, clear_history=False):
    """
    Process an uploaded file and update session state.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        clear_history: Whether to clear chat history (for new file uploads)
    
    Returns:
        bool: True if file was processed successfully, False otherwise
    """
    if uploaded_file is None:
        return False
        
    try:
        df = load_table(uploaded_file)
        
        # Check if the loaded DataFrame is empty or has no meaningful data
        if df.empty or df.shape[1] == 0:
            st.warning("⚠️ **Empty file detected!**\n\nThe uploaded file appears to be empty or contains no data columns. Please upload a file with actual data to proceed.")
            st.info("💡 **Tips for a successful upload:**\n- Ensure your file has at least one column with data\n- Check that your Excel sheet isn't blank\n- Verify your CSV has headers and data rows")
            st.session_state.df = None
            return False
            
        elif df.shape[0] == 0:
            st.warning("⚠️ **No data rows found!**\n\nYour file has column headers but no data rows. Please upload a file with actual data to analyze.")
            st.info("💡 Make sure your file contains data rows below the headers.")
            st.session_state.df = None
            return False
            
        else:
            # File is valid - update session state
            st.session_state.df = df
            st.session_state.uploaded_filename = uploaded_file.name
            
            # Clear previous analysis if requested (for new file uploads)
            if clear_history:
                st.session_state.chat_history = []
                st.session_state.chart_data = []
                
            return True
            
    except Exception as e:
        st.error(f"❌ **Unexpected error loading file:** {type(e).__name__}")
        st.info("💡 Please check your file format and try again. Supported formats: CSV, XLSX, XLS")
        with st.expander("🔍 Technical details (for debugging)"):
            st.code(str(e))
        st.session_state.df = None
        return False


def display_single_chart(chart_info, chart_id=None):
    """Display a single chart based on chart_info dictionary"""
    chart_type = chart_info.get('type')
    data = chart_info.get('data')
    title = chart_info.get('title', 'Chart')
    
    # Generate unique ID for this chart
    if chart_id is None:
        chart_id = f"{chart_type}_{hash(title) % 10000}"
    
    # Debug information
    st.caption(f"🔍 Chart Type: {chart_type} | Data Shape: {data.shape if hasattr(data, 'shape') else 'N/A'} | Title: {title}")
    
    if chart_type == 'histogram':
        st.subheader(f"📊 {title}")
        st.bar_chart(data.set_index('bin_range')['count'], use_container_width=True)
        
    elif chart_type == 'bar':
        st.subheader(f"📊 {title}")
        st.bar_chart(data.set_index('category')['count'], use_container_width=True)
        
    elif chart_type == 'scatter':
        st.subheader(f"📊 {title}")
        if chart_info.get('color_column'):
            st.scatter_chart(
                data, 
                x=chart_info['x_column'], 
                y=chart_info['y_column'],
                color=chart_info['color_column'],
                use_container_width=True
            )
        else:
            st.scatter_chart(
                data, 
                x=chart_info['x_column'], 
                y=chart_info['y_column'],
                use_container_width=True
            )
        # Show correlation info
        corr = chart_info.get('correlation', 0)
        if abs(corr) > 0.7:
            st.success(f"🔍 Strong correlation: {corr:.3f}")
        elif abs(corr) > 0.3:
            st.info(f"📈 Moderate correlation: {corr:.3f}")
        else:
            st.caption(f"📊 Weak correlation: {corr:.3f}")
            
    elif chart_type == 'line':
        st.subheader(f"📊 {title}")
        st.line_chart(
            data.set_index(chart_info['x_column'])[chart_info['y_column']], 
            use_container_width=True
        )
        
    elif chart_type == 'time_series':
        st.subheader(f"📈 {title}")
        # Recreate the chart from data to avoid Plotly object serialization issues
        try:
            import plotly.express as px
            # Use the stored data to recreate the chart
            fig = px.line(
                data, 
                x='Date', 
                y='Value',
                title=title
            )
            
            # Apply styling from chart config if available
            chart_config = chart_info.get('chart_config', {})
            line_width = chart_config.get('line_width', 2)
            ylabel = chart_config.get('ylabel', 'Value')
            
            fig.update_traces(
                line=dict(width=line_width),
                hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>'
            )
            
            fig.update_layout(
                title=dict(x=0.5, font=dict(size=16)),
                xaxis_title="Date",
                yaxis_title=ylabel,
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"timeseries_{chart_id}")
        except ImportError:
            # Fallback if plotly not available
            st.line_chart(data.set_index('Date')['Value'], use_container_width=True)
            
    elif chart_type == 'correlation_heatmap':
        st.subheader(f"🔥 {title}")
        try:
            import plotly.express as px
            import pandas as pd
            
            # Recreate correlation matrix from stored data
            corr_matrix = chart_info.get('correlation_matrix', {})
            if corr_matrix:
                df_corr = pd.DataFrame(corr_matrix)
                
                # Recreate the heatmap
                fig = px.imshow(
                    df_corr,
                    text_auto=True,
                    aspect="auto",
                    title=title,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                
                # Apply styling
                fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textfont_size=10,
                    hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
                )
                
                cols_count = len(chart_info.get('columns', []))
                fig.update_layout(
                    title=dict(x=0.5, font=dict(size=16)),
                    height=max(400, cols_count * 40),
                    width=max(400, cols_count * 40)
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{chart_id}")
            else:
                st.error("No correlation matrix data available")
                
        except ImportError:
            # Fallback to simple dataframe display
            corr_matrix = chart_info.get('correlation_matrix', {})
            if corr_matrix:
                import pandas as pd
                df_corr = pd.DataFrame(corr_matrix)
                st.dataframe(df_corr.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))


# ---------- Context extraction ----------
def extract_goal_kpis(text: str):
    if not text or not text.strip():
        return {"goal": "", "kpis": []}

    goal = text.strip().split(".")[0][:160].strip()
    common_kpis = [
        "retention", "churn", "activation", "conversion", "revenue", "arpu",
        "ltv", "cac", "mau", "wau", "dau", "engagement", "trial",
        "trial-to-paid", "cohort", "drop-off", "signup", "onboarding",
        "nps", "csat", "aov", "gmv"
    ]
    lower = text.lower()
    kpis = sorted({k for k in common_kpis if k in lower})
    return {"goal": goal, "kpis": kpis}

# ---------- Analytics helpers ----------
def summarise_dataset(df: pd.DataFrame):
    # Handle empty DataFrame case
    
    
    summary = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "numeric_cols": df.select_dtypes(include=np.number).shape[1],
        "non_numeric_cols": df.select_dtypes(exclude=np.number).shape[1],
        "missing_values_total": int(df.isna().sum().sum()),
    }
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    missing_by_col = missing_by_col[missing_by_col > 0].head(20)
    numeric_desc = df.select_dtypes(include=np.number).describe().T

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    top_cats = {}
    for c in cat_cols[:5]:
        vc = df[c].astype(str).value_counts(dropna=False).head(10)
        top_cats[c] = vc

    return summary, missing_by_col, numeric_desc, top_cats

def key_trends_numeric_only(df: pd.DataFrame):
    num = df.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        return num.corr(numeric_only=True)
    return None

def suggest_visualisations(df: pd.DataFrame):
    suggestions = []
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        def _render_hist(col):
            st.bar_chart(df[col].dropna().value_counts().sort_index(), use_container_width=True)
        suggestions.append((f"Distribution of {num_cols[0]}", lambda col=num_cols[0]: _render_hist(col)))

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if cat_cols:
        def _render_topcats(col):
            st.bar_chart(df[col].astype(str).value_counts().head(15), use_container_width=True)
        suggestions.append((f"Top categories in {cat_cols[0]}", lambda col=cat_cols[0]: _render_topcats(col)))

    return suggestions


# ---------- Token Usage & Cost Tracking ----------

# OpenAI GPT-4 pricing (as of 2024)
OPENAI_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
}

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: The text to count tokens for
        model: The model name to get the correct encoding
    
    Returns:
        Number of tokens
    """
    try:
        # Get the encoding for the model
        if "gpt-4" in model:
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            # Fallback to cl100k_base encoding (used by GPT-4)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
    """
    Calculate the cost for a given number of input and output tokens.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name for pricing
    
    Returns:
        Total cost in USD
    """
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-4o"])
    
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    
    return input_cost + output_cost

def track_usage(user_message: str, assistant_response: str, model: str = "gpt-4o") -> Dict:
    """
    Track token usage and cost for an interaction.
    
    Args:
        user_message: The user's input message
        assistant_response: The AI's response
        model: Model name for accurate counting and pricing
    
    Returns:
        Dictionary with usage statistics
    """
    input_tokens = count_tokens(user_message, model)
    output_tokens = count_tokens(assistant_response, model)
    total_tokens = input_tokens + output_tokens
    cost = calculate_cost(input_tokens, output_tokens, model)
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "model": model,
        "timestamp": datetime.now().isoformat()
    }

def update_session_usage(usage_stats: Dict):
    """
    Update the session state with cumulative usage statistics.
    
    Args:
        usage_stats: Usage statistics from track_usage()
    """
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "usage_history" not in st.session_state:
        st.session_state.usage_history = []
    
    # Update totals
    st.session_state.total_tokens_used += usage_stats["total_tokens"]
    st.session_state.total_cost += usage_stats["cost"]
    
    # Add to history
    st.session_state.usage_history.append(usage_stats)

def format_cost_display(cost: float) -> str:
    """
    Format cost for display with appropriate precision.
    
    Args:
        cost: Cost in USD
    
    Returns:
        Formatted cost string
    """
    if cost >= 1:
        return f"${cost:.2f}"
    elif cost >= 0.01:
        return f"${cost:.3f}"
    else:
        return f"${cost:.4f}"

def get_usage_summary() -> Tuple[int, float, str]:
    """
    Get current session usage summary.
    
    Returns:
        Tuple of (total_tokens, total_cost, formatted_display)
    """
    total_tokens = st.session_state.get("total_tokens_used", 0)
    total_cost = st.session_state.get("total_cost", 0.0)
    
    if total_tokens == 0:
        display = "💰 No usage yet"
    else:
        cost_str = format_cost_display(total_cost)
        display = f"💰 {total_tokens:,} tokens | {cost_str}"
    
    return total_tokens, total_cost, display

def check_usage_limits(tokens: int, cost: float) -> Optional[str]:
    """
    Check if usage is approaching limits and return warning message.
    
    Args:
        tokens: Current token count
        cost: Current cost
    
    Returns:
        Warning message if approaching limits, None otherwise
    """
    # Define warning thresholds
    TOKEN_WARNING = 50000  # 50K tokens
    TOKEN_LIMIT = 100000   # 100K tokens
    COST_WARNING = 5.0     # $5
    COST_LIMIT = 10.0      # $10
    
    if tokens >= TOKEN_LIMIT or cost >= COST_LIMIT:
        return f"🚨 **Usage Limit Reached!** ({tokens:,} tokens, {format_cost_display(cost)})"
    elif tokens >= TOKEN_WARNING or cost >= COST_WARNING:
        return f"⚠️ **High Usage Warning** ({tokens:,} tokens, {format_cost_display(cost)})"
    else:
        return None


# ---------- Export & Reporting Functions ----------

def get_session_stats(chat_history: list) -> dict:
    """
    Calculate session statistics for reporting.
    
    Args:
        chat_history: List of chat messages
    
    Returns:
        Dictionary with session statistics
    """
    user_questions = [msg for msg in chat_history if msg["role"] == "user"]
    assistant_responses = [msg for msg in chat_history if msg["role"] == "assistant"]
    total_charts = sum(len(msg.get("charts", [])) for msg in assistant_responses)
    
    return {
        "user_questions": len(user_questions),
        "assistant_responses": len(assistant_responses),
        "total_charts": total_charts,
        "has_analysis": bool(chat_history)
    }

def render_session_summary(chat_history: list, location: str = "sidebar"):
    """
    Render session statistics summary.
    
    Args:
        chat_history: List of chat messages
        location: Where this is being rendered ("sidebar" or "tab")
    """
    import streamlit as st
    
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
    import zipfile
    from io import BytesIO
    from export_utils import export_chart_as_image
    
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
    import streamlit as st
    from datetime import datetime
    from export_utils import generate_markdown_report, export_pdf_report
    
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
    import streamlit as st
    import json
    from datetime import datetime
    from export_utils import export_session_data
    
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
    import streamlit as st
    from export_utils import generate_markdown_report
    
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


# ---------- Data Cleaning Functions (Phase 1 Implementation) ----------

def analyze_missing_data(df: pd.DataFrame) -> Dict:
    """
    Comprehensive missing data analysis - our competitive advantage over ChatGPT.
    Returns detailed insights about missing data patterns.
    """
    missing_info = {}
    
    # Basic missing data stats
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_info['total_rows'] = len(df)
    missing_info['columns_with_missing'] = missing_counts[missing_counts > 0].to_dict()
    missing_info['missing_percentages'] = missing_percentages[missing_percentages > 0].to_dict()
    
    # Advanced pattern analysis
    missing_info['completely_empty_columns'] = missing_counts[missing_counts == len(df)].index.tolist()
    missing_info['mostly_missing'] = missing_percentages[missing_percentages > 80].index.tolist()
    missing_info['partially_missing'] = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 80)].index.tolist()
    
    # Missing data correlation (which columns tend to be missing together)
    if len(missing_info['columns_with_missing']) > 1:
        missing_matrix = df.isnull()
        missing_corr = missing_matrix.corr()
        # Find high correlations in missing patterns
        high_corr_pairs = []
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    high_corr_pairs.append({
                        'col1': missing_corr.columns[i],
                        'col2': missing_corr.columns[j], 
                        'correlation': round(corr_val, 3)
                    })
        missing_info['correlated_missing_patterns'] = high_corr_pairs
    
    # Row-wise missing analysis
    rows_missing_counts = df.isnull().sum(axis=1)
    missing_info['rows_with_no_missing'] = (rows_missing_counts == 0).sum()
    missing_info['rows_completely_empty'] = (rows_missing_counts == len(df.columns)).sum()
    missing_info['avg_missing_per_row'] = round(rows_missing_counts.mean(), 2)
    
    return missing_info

def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict:
    """
    Advanced outlier detection using multiple methods.
    This systematic approach gives us advantage over ChatGPT's basic suggestions.
    """
    outlier_info = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return {'message': 'No numeric columns found for outlier detection'}
    
    outlier_info['method_used'] = method
    outlier_info['numeric_columns_analyzed'] = list(numeric_cols)
    outlier_info['outliers_by_column'] = {}
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = col_data[z_scores > 3]
            
        elif method == 'modified_zscore':
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / mad
            outliers = col_data[np.abs(modified_z_scores) > 3.5]
        
        outlier_info['outliers_by_column'][col] = {
            'count': len(outliers),
            'percentage': round((len(outliers) / len(col_data)) * 100, 2),
            'values': outliers.tolist()[:10] if len(outliers) <= 10 else outliers.tolist()[:10] + ['...'],
            'min_outlier': outliers.min() if len(outliers) > 0 else None,
            'max_outlier': outliers.max() if len(outliers) > 0 else None
        }
    
    # Summary statistics
    total_outliers = sum([info['count'] for info in outlier_info['outliers_by_column'].values()])
    outlier_info['total_outliers_found'] = total_outliers
    outlier_info['columns_with_outliers'] = [col for col, info in outlier_info['outliers_by_column'].items() if info['count'] > 0]
    
    return outlier_info

def find_duplicates(df: pd.DataFrame, subset_cols: Optional[list] = None) -> Dict:
    """
    Comprehensive duplicate detection and analysis.
    More thorough than ChatGPT's basic duplicate checking.
    """
    duplicate_info = {}
    
    # Complete row duplicates
    complete_duplicates = df.duplicated()
    duplicate_info['complete_duplicates'] = {
        'count': complete_duplicates.sum(),
        'percentage': round((complete_duplicates.sum() / len(df)) * 100, 2),
        'duplicate_rows': df[complete_duplicates].index.tolist() if complete_duplicates.sum() <= 20 else df[complete_duplicates].index.tolist()[:20] + ['...']
    }
    
    # Subset duplicates if specified
    if subset_cols:
        valid_cols = [col for col in subset_cols if col in df.columns]
        if valid_cols:
            subset_duplicates = df.duplicated(subset=valid_cols)
            duplicate_info['subset_duplicates'] = {
                'columns_checked': valid_cols,
                'count': subset_duplicates.sum(),
                'percentage': round((subset_duplicates.sum() / len(df)) * 100, 2),
                'duplicate_rows': df[subset_duplicates].index.tolist() if subset_duplicates.sum() <= 20 else df[subset_duplicates].index.tolist()[:20] + ['...']
            }
    
    # Column-wise duplicate analysis
    duplicate_info['column_analysis'] = {}
    for col in df.columns:
        col_duplicates = df[col].duplicated()
        unique_vals = df[col].nunique()
        duplicate_info['column_analysis'][col] = {
            'unique_values': unique_vals,
            'duplicate_count': col_duplicates.sum(),
            'uniqueness_ratio': round(unique_vals / len(df), 3)
        }
    
    # Identify potential key columns (high uniqueness)
    high_uniqueness_cols = [col for col, info in duplicate_info['column_analysis'].items() 
                           if info['uniqueness_ratio'] > 0.95]
    duplicate_info['potential_key_columns'] = high_uniqueness_cols
    
    # Identify low uniqueness columns (may need attention)
    low_uniqueness_cols = [col for col, info in duplicate_info['column_analysis'].items() 
                          if info['uniqueness_ratio'] < 0.1]
    duplicate_info['low_uniqueness_columns'] = low_uniqueness_cols
    
    return duplicate_info

def suggest_data_cleaning_actions(df: pd.DataFrame) -> Dict:
    """
    AI-powered data cleaning suggestions based on comprehensive analysis.
    This strategic recommendation system is our key differentiator.
    """
    suggestions = {
        'priority_actions': [],
        'optional_improvements': [],
        'data_quality_score': 0,
        'analysis_summary': {}
    }
    
    # Run all analyses
    missing_analysis = analyze_missing_data(df)
    outlier_analysis = detect_outliers(df)
    duplicate_analysis = find_duplicates(df)
    
    suggestions['analysis_summary'] = {
        'missing_data': missing_analysis,
        'outliers': outlier_analysis,
        'duplicates': duplicate_analysis
    }
    
    # Generate priority actions
    priority_score = 100
    
    # Critical issues (high priority)
    if missing_analysis.get('completely_empty_columns'):
        suggestions['priority_actions'].append({
            'action': 'Remove completely empty columns',
            'reason': f"Found {len(missing_analysis['completely_empty_columns'])} columns with no data",
            'columns': missing_analysis['completely_empty_columns'],
            'impact': 'High - reduces noise and improves analysis efficiency'
        })
        priority_score -= 20
    
    if duplicate_analysis['complete_duplicates']['count'] > 0:
        suggestions['priority_actions'].append({
            'action': 'Remove duplicate rows',
            'reason': f"Found {duplicate_analysis['complete_duplicates']['count']} duplicate rows ({duplicate_analysis['complete_duplicates']['percentage']}%)",
            'impact': 'High - prevents skewed analysis results'
        })
        priority_score -= 15
    
    if missing_analysis.get('mostly_missing'):
        suggestions['priority_actions'].append({
            'action': 'Consider removing high-missing columns',
            'reason': f"Columns with >80% missing data: {missing_analysis['mostly_missing']}",
            'columns': missing_analysis['mostly_missing'],
            'impact': 'Medium - may not provide reliable insights'
        })
        priority_score -= 10
    
    # Optional improvements
    if outlier_analysis.get('total_outliers_found', 0) > 0:
        suggestions['optional_improvements'].append({
            'action': 'Investigate outliers',
            'reason': f"Found {outlier_analysis['total_outliers_found']} outliers across {len(outlier_analysis.get('columns_with_outliers', []))} columns",
            'impact': 'Medium - may indicate data quality issues or interesting patterns'
        })
        priority_score -= 5
    
    if missing_analysis.get('partially_missing'):
        suggestions['optional_improvements'].append({
            'action': 'Consider imputation strategies',
            'reason': f"Columns with partial missing data: {missing_analysis['partially_missing']}",
            'impact': 'Medium - can improve dataset completeness'
        })
        priority_score -= 5
    
    if duplicate_analysis.get('low_uniqueness_columns'):
        suggestions['optional_improvements'].append({
            'action': 'Review low-uniqueness columns',
            'reason': f"Columns with <10% unique values: {duplicate_analysis['low_uniqueness_columns']}",
            'impact': 'Low - may indicate categorical data that needs encoding'
        })
    
    suggestions['data_quality_score'] = max(0, priority_score)
    
    return suggestions

def apply_basic_cleaning(df: pd.DataFrame, actions: list) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply selected cleaning actions and return cleaned dataframe with summary.
    This automated cleaning capability sets us apart from ChatGPT.
    """
    cleaning_summary = {
        'actions_applied': [],
        'rows_before': len(df),
        'cols_before': len(df.columns),
        'rows_after': 0,
        'cols_after': 0,
        'changes_made': {}
    }
    
    cleaned_df = df.copy()
    
    for action in actions:
        if action == 'remove_empty_columns':
            empty_cols = cleaned_df.columns[cleaned_df.isnull().all()].tolist()
            cleaned_df = cleaned_df.drop(columns=empty_cols)
            cleaning_summary['actions_applied'].append(f"Removed {len(empty_cols)} empty columns")
            cleaning_summary['changes_made']['removed_empty_columns'] = empty_cols
            
        elif action == 'remove_duplicates':
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            after_count = len(cleaned_df)
            removed_count = before_count - after_count
            cleaning_summary['actions_applied'].append(f"Removed {removed_count} duplicate rows")
            cleaning_summary['changes_made']['removed_duplicates'] = removed_count
            
        elif action == 'remove_high_missing_columns':
            missing_percentages = (cleaned_df.isnull().sum() / len(cleaned_df)) * 100
            high_missing_cols = missing_percentages[missing_percentages > 80].index.tolist()
            cleaned_df = cleaned_df.drop(columns=high_missing_cols)
            cleaning_summary['actions_applied'].append(f"Removed {len(high_missing_cols)} high-missing columns")
            cleaning_summary['changes_made']['removed_high_missing_columns'] = high_missing_cols
    
    cleaning_summary['rows_after'] = len(cleaned_df)
    cleaning_summary['cols_after'] = len(cleaned_df.columns)
    cleaning_summary['rows_removed'] = cleaning_summary['rows_before'] - cleaning_summary['rows_after']
    cleaning_summary['cols_removed'] = cleaning_summary['cols_before'] - cleaning_summary['cols_after']
    
    return cleaned_df, cleaning_summary
