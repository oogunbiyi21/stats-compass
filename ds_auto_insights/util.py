# ds_auto_insights/utils.py
import io
import numpy as np
import pandas as pd
import streamlit as st

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


# ---------- Context extraction (heuristic) ----------
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

# ---------- Analytics helpers (no time series) ----------
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
